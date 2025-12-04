# metai/model/met_mamba/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple

# ==============================================================================
# 工具函数 & 常量
# ==============================================================================

# 物理参数：必须与 Dataset 的归一化逻辑严格一致
# 假设最大降水为 30mm/h，用于对数归一化反算
MM_MAX = 30.0  
LOG_NORM_FACTOR = math.log(MM_MAX + 1)

def mm_to_lognorm(mm_val: float) -> float:
    """将物理降水值 (mm) 转换为对数归一化值 (0-1)。"""
    return math.log(mm_val + 1) / LOG_NORM_FACTOR

# ==============================================================================
# 核心 Loss 组件 (Atomic Loss Components)
# ==============================================================================

class BalancedMSELoss(nn.Module):
    """
    [强度回归 Loss]
    针对气象数据的长尾分布设计。保留了激进的权重策略，
    强迫模型关注稀有但重要的强降水事件。
    """
    def __init__(self):
        super().__init__()
        # 定义阈值 (mm) -> Log Space
        self.thresh_light = mm_to_lognorm(0.1)
        self.thresh_mod   = mm_to_lognorm(2.0)
        self.thresh_heavy = mm_to_lognorm(5.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, extra_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 使用 L1 Loss 替代 MSE，对异常值更鲁棒
        diff = torch.abs(pred - target)
        
        # 激进权重分配 (Aggressive Weighting)
        weights = torch.ones_like(target)
        weights[target >= self.thresh_light] = 5.0   
        weights[target >= self.thresh_mod]   = 20.0  
        weights[target >= self.thresh_heavy] = 50.0  

        loss_map = diff * weights
        
        # 叠加额外权重 (如时间衰减权重)
        if extra_weights is not None:
            loss_map = loss_map * extra_weights
            weights = weights * extra_weights

        # 应用有效区域 Mask
        if mask is not None:
            loss_map = loss_map * mask
            weights = weights * mask

        # 归一化：除以权重的总和，保持梯度数值稳定
        return loss_map.sum() / (weights.sum() + 1e-8)


class CSILoss(nn.Module):
    """
    [结构与指标 Loss]
    替代了传统的 DiceLoss。直接优化多个阈值下的 CSI (Critical Success Index)，
    能更好地约束降水区域的形状和拓扑结构。
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        # 定义关键业务阈值
        thresholds_mm = [0.1, 1.0, 2.0, 5.0, 8.0]
        # 给予高阈值更高的关注度
        weights_intensity = [1.0, 1.0, 2.0, 5.0, 10.0] 
        
        self.register_buffer('thresholds', torch.tensor([mm_to_lognorm(t) for t in thresholds_mm]))
        self.register_buffer('intensity_weights', torch.tensor(weights_intensity))
        self.smooth = smooth
        self.temperature = 50.0 # 控制 Sigmoid 的陡峭程度，越大越接近阶跃函数

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        total_loss = 0.0
        total_w = 0.0
        
        for i, thresh in enumerate(self.thresholds):
            w = self.intensity_weights[i]
            # 软 CSI 计算 (Differentiable approximation)
            pred_score = torch.sigmoid((pred - thresh) * self.temperature)
            target_score = (target > thresh).float()
            
            if mask is not None:
                pred_score = pred_score * mask
                target_score = target_score * mask
            
            intersection = (pred_score * target_score).sum(dim=(-2, -1))
            union = pred_score.sum(dim=(-2, -1)) + target_score.sum(dim=(-2, -1)) - intersection
            
            csi = (intersection + self.smooth) / (union + self.smooth)
            # Loss = (1 - CSI) * weight
            total_loss += (1.0 - csi).mean() * w
            total_w += w
            
        return total_loss / (total_w + 1e-8)


class SpectralLoss(nn.Module):
    """
    [纹理与频域 Loss]
    在频域 (FFT) 约束预测结果。这对于防止预测图像模糊 (Blurring) 至关重要，
    迫使模型生成合理的高频纹理细节。
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            pred = pred * mask
            target = target * mask

        # 2D 实数 FFT 变换
        fft_pred = torch.fft.rfft2(pred, dim=(-2, -1), norm='ortho')
        fft_target = torch.fft.rfft2(target, dim=(-2, -1), norm='ortho')

        # 幅度 Loss (L1): 约束频谱能量分布
        amp_pred = torch.abs(fft_pred)
        amp_target = torch.abs(fft_target)
        loss_amp = F.l1_loss(amp_pred, amp_target)
        
        # (可选) 相位/相关性 Loss 可以在此添加，但仅幅度通常已足够
        return loss_amp


class PhysicsConstraintsLoss(nn.Module):
    """
    [物理约束 Loss]
    包含两个子约束：
    1. 局部质量守恒 (Local Conservation): 防止大范围弱降水欺骗全局 Sum。
    2. 光流一致性 (Optical Flow/Consistency): 约束运动平滑性。
    """
    def __init__(self, pool_size=4):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # --- 1. 局部质量守恒约束 (改进) ---
        # 使用 AvgPool 将图像分块，约束每个局部块内的降水总量
        if mask is not None:
            pred = pred * mask
            target = target * mask
            
        # [修改] 全局 Sum -> 局部 Avg (等价于局部 Sum 约束)
        # Input: [B, T, H, W] -> [B*T, 1, H, W] for pooling
        b, t, h, w = pred.shape
        p_reshaped = pred.view(b * t, 1, h, w)
        t_reshaped = target.view(b * t, 1, h, w)
        
        # 计算局部平均 (Local Average)
        p_local = F.avg_pool2d(p_reshaped, kernel_size=self.pool_size, stride=self.pool_size)
        t_local = F.avg_pool2d(t_reshaped, kernel_size=self.pool_size, stride=self.pool_size)
        
        loss_cons = F.l1_loss(p_local, t_local)
        
        # --- 2. 变化率/光流一致性约束 (保持不变) ---
        p_diff = pred[:, 1:] - pred[:, :-1]
        t_diff = target[:, 1:] - target[:, :-1]
        loss_flow = F.l1_loss(p_diff, t_diff)
        
        return loss_cons, loss_flow

# ==============================================================================
# 智能混合 Loss (Automatic Weighted Hybrid Loss)
# ==============================================================================

class HybridLoss(nn.Module):
    """
    [智能混合损失 - 最终版]
    基于 Kendall's Multi-Task Learning (CVPR 2018) 策略。
    
    自动学习 5 个不同 Loss 分量的权重，解决梯度冲突和超参调节难题。
    公式: Total_Loss = Sum( 0.5 * exp(-s) * Loss + 0.5 * s )
    其中 s = log(sigma^2) 是可学习的不确定性参数。
    """
    def __init__(self, 
                 use_temporal_weight: bool = True,
                 **kwargs): 
        # kwargs 用于接收 config 中的旧参数并安全忽略，保持接口兼容
        super().__init__()
        self.use_temporal_weight = use_temporal_weight
        
        # 1. 初始化各子 Loss
        self.loss_mse = BalancedMSELoss()
        self.loss_csi = CSILoss()
        self.loss_spectral = SpectralLoss()
        self.loss_physics = PhysicsConstraintsLoss()
        
        # 2. 定义可学习参数 (Learnable Weights)
        # s 代表 log(variance)。初始化为 0 意味着初始方差为 1，初始权重为 0.5。
        # 形状为 5，分别对应: [MSE, CSI, Spectral, Conservation, Flow]
        self.params = nn.Parameter(torch.zeros(5)) 
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                current_epoch: int = 100, total_epochs: int = 100) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        # 维度兼容性处理
        if pred.ndim == 5: pred = pred.squeeze(2)
        if target.ndim == 5: target = target.squeeze(2)
        if mask is not None and mask.ndim == 5: mask = mask.squeeze(2)

        # 1. 准备时间权重 (仅增强 MSE)
        extra_weights = None
        if self.use_temporal_weight:
            T = pred.shape[1]
            # 线性增加权重 (1.0 -> 2.0)，越远的时刻越重要
            extra_weights = torch.linspace(1.0, 2.0, steps=T, device=pred.device).view(1, T, 1, 1)

        # 2. 计算各项原始 Loss (Raw Values)
        l_mse = self.loss_mse(pred, target, mask, extra_weights)
        l_csi = self.loss_csi(pred, target, mask)
        l_spec = self.loss_spectral(pred, target, mask)
        l_cons, l_flow = self.loss_physics(pred, target, mask)
        
        # 将所有 Loss 堆叠: [5]
        losses = torch.stack([l_mse, l_csi, l_spec, l_cons, l_flow])
        
        # 3. 自动加权计算 (Automatic Weighting)
        # s: 可学习的不确定性参数
        s = self.params
        # precision: 相当于权重 (1 / sigma^2)
        precision = torch.exp(-s)
        
        # 贝叶斯损失公式: L = 0.5 * (precision * raw_loss + log_variance)
        weighted_losses = 0.5 * (precision * losses + s)
        
        total_loss = weighted_losses.sum()
        
        # 4. 构建返回字典 (用于监控)
        loss_dict = {
            'total': total_loss,
            # --- 原始 Loss (用于观察物理指标) ---
            'mse_raw': l_mse.detach(),
            'csi_raw': l_csi.detach(),
            'spec_raw': l_spec.detach(),
            'cons_raw': l_cons.detach(),
            'flow_raw': l_flow.detach(),
            # --- 学习到的实际权重 (0.5 * exp(-s)) ---
            # 观察这些值的变化可以看出模型当前关注哪些任务
            'w_mse': 0.5 * precision[0].detach(),
            'w_csi': 0.5 * precision[1].detach(),
            'w_spec': 0.5 * precision[2].detach(),
            'w_cons': 0.5 * precision[3].detach(),
            'w_flow': 0.5 * precision[4].detach(),
            # --- 不确定性参数 s ---
            's_mse': s[0].detach(),
            's_csi': s[1].detach()
        }
        
        return total_loss, loss_dict