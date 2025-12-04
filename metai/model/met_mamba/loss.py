# metai/model/met_mamba/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple

# ==============================================================================
# 工具函数 & 物理常量
# ==============================================================================

# 物理参数：必须与 Dataset 的归一化逻辑严格一致
# 假设最大降水为 30mm/h (对于极端强对流，建议在 Dataset 中调整此值，如 100mm/h)
MM_MAX = 30.0  
LOG_NORM_FACTOR = math.log(MM_MAX + 1)

def mm_to_lognorm(mm_val: float) -> float:
    """
    将物理降水值 (mm) 转换为对数归一化值 (0-1)。
    用于在特征空间定义具有物理意义的阈值。
    """
    return math.log(mm_val + 1) / LOG_NORM_FACTOR

# ==============================================================================
# 核心 Loss 组件 (Atomic Loss Components)
# ==============================================================================

class BalancedMSELoss(nn.Module):
    """
    [强度回归 Loss - 长尾分布优化版]
    
    科学原理：
    气象数据遵循极端的长尾分布（大量零值/弱降水，极少量强降水）。
    标准 MSE 会被占据主导地位的零值“淹没”，导致模型倾向于预测平滑的低值。
    
    改进策略：
    引入“阶梯式加权”（Aggressive Weighting），根据 Target 的物理强度
    动态调整梯度权重，强迫模型关注稀有但高危的极端天气事件。
    """
    def __init__(self):
        super().__init__()
        # 定义关键业务阈值 (mm -> Log Space)
        self.thresh_light = mm_to_lognorm(0.1)  # 小雨
        self.thresh_mod   = mm_to_lognorm(2.0)  # 中雨
        self.thresh_heavy = mm_to_lognorm(5.0)  # 大/暴雨

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, extra_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 使用 L1 Loss 替代 MSE，对异常值更鲁棒，且梯度恒定
        diff = torch.abs(pred - target)
        
        # --- 阶梯式权重分配 ---
        weights = torch.ones_like(target)
        # 给予强回波极高的关注度 (5x -> 20x -> 50x)
        weights[target >= self.thresh_light] = 5.0   
        weights[target >= self.thresh_mod]   = 20.0  
        weights[target >= self.thresh_heavy] = 50.0  

        loss_map = diff * weights
        
        # 叠加额外权重 (如时间衰减权重：越远的时刻越难预测，权重可适当调高)
        if extra_weights is not None:
            loss_map = loss_map * extra_weights
            weights = weights * extra_weights

        # 应用有效区域 Mask (如雷达扫描边界掩码)
        if mask is not None:
            loss_map = loss_map * mask
            weights = weights * mask

        # 归一化：除以权重的总和，保持梯度数值稳定，防止 Loss 随 Batch 内容剧烈波动
        return loss_map.sum() / (weights.sum() + 1e-8)


class CSILoss(nn.Module):
    """
    [拓扑结构 Loss - 软化 CSI 指标]
    
    科学原理：
    Dice Loss 或 IoU Loss 的气象学变体。直接优化 Critical Success Index (CSI)，
    这比像素级回归更能约束降水区域的“形状”和“位置”准确性。
    
    改进策略：
    使用 Sigmoid 温度缩放来近似阶跃函数，使其可微。
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        # 定义关键业务阈值
        thresholds_mm = [0.1, 1.0, 2.0, 5.0, 8.0]
        # 给予高阈值更高的权重
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
            # pred > thresh => sigmoid((pred - thresh) * T) -> 1.0
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
    [频域纹理 Loss - FFT]
    
    科学原理：
    基于 MSE 的模型倾向于生成模糊的图像（平均效应），导致高频信息（纹理细节）丢失。
    在频域约束幅度谱，迫使模型生成具有合理高频分量的预测结果。
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            pred = pred * mask
            target = target * mask

        # 2D 实数 FFT 变换
        # norm='ortho' 保证能量守恒
        fft_pred = torch.fft.rfft2(pred, dim=(-2, -1), norm='ortho')
        fft_target = torch.fft.rfft2(target, dim=(-2, -1), norm='ortho')

        # 幅度谱 Loss (L1)
        amp_pred = torch.abs(fft_pred)
        amp_target = torch.abs(fft_target)
        loss_amp = F.l1_loss(amp_pred, amp_target)
        
        return loss_amp


class PhysicsConstraintsLoss(nn.Module):
    """
    [物理约束 Loss - 增强版]
    
    包含两个子约束：
    1. 非对称局部质量守恒 (Asymmetric Local Conservation): 
       防止模型通过降低总降水量来规避风险。
       对于“漏报爆发”（Under-prediction）给予比“虚报”更重的惩罚。
    2. 显式平流一致性 (Warp Loss): 
       利用模型预测的流场对前一帧进行 Warp，强制流场符合物理运动规律。
    """
    def __init__(self, pool_size=4, under_penalty=2.0):
        super().__init__()
        self.pool_size = pool_size
        self.under_penalty = under_penalty # 漏报惩罚系数

    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                flow: Optional[torch.Tensor] = None, 
                prev_frame: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # --- 1. 局部质量守恒约束 (非对称版) ---
        if mask is not None:
            pred_masked = pred * mask
            target_masked = target * mask
        else:
            pred_masked = pred
            target_masked = target
            
        # Reshape for pooling: [B, T, H, W] -> [B*T, 1, H, W]
        b, t, h, w = pred.shape
        p_reshaped = pred_masked.view(b * t, 1, h, w)
        t_reshaped = target_masked.view(b * t, 1, h, w)
        
        # 计算局部平均 (Local Average)，模拟降水通量
        p_local = F.avg_pool2d(p_reshaped, kernel_size=self.pool_size, stride=self.pool_size)
        t_local = F.avg_pool2d(t_reshaped, kernel_size=self.pool_size, stride=self.pool_size)
        
        # 计算差异: 预测 - 真实
        # diff < 0 意味着漏报 (预测值 < 真实值)，施加更重惩罚
        diff = p_local - t_local
        weight_map = torch.where(diff < 0, self.under_penalty, 1.0)
        
        loss_cons = (torch.abs(diff) * weight_map).mean()
        
        # --- 2. 显式 Warp Loss ---
        # 如果提供了流场和前一帧，计算：| I_{t+1} - Warp(I_t, Flow) |
        loss_warp = torch.tensor(0.0, device=pred.device)
        
        if flow is not None and prev_frame is not None:
            # 构造推演源序列 (Source Images)
            # T0_src = prev_frame (观测最后一帧) -> 预测 T0_tgt (target 第一帧)
            # T1_src = target[0] -> 预测 T1_tgt (target 第二帧) (类似 Teacher Forcing)
            src_imgs = torch.cat([prev_frame, target[:, :-1]], dim=1) # [B, T, C, H, W]
            tgt_imgs = target # [B, T, C, H, W]
            
            B, T, C, H, W = tgt_imgs.shape
            
            # 1. 上采样流场: Flow 是在 Latent 空间计算的，通常分辨率较低
            # flow shape: [B, T, 2, H_lat, W_lat]
            flow_up = F.interpolate(
                flow.view(B*T, 2, flow.shape[-2], flow.shape[-1]), 
                size=(H, W), mode='bilinear', align_corners=False
            ) # -> [B*T, 2, H, W]
            
            # 2. 构建采样 Grid
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=flow.device),
                torch.linspace(-1, 1, W, device=flow.device),
                indexing='ij'
            )
            base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B*T, -1, -1, -1) # [B*T, H, W, 2]
            
            # 3. 叠加流场 (注意 permute 顺序)
            # flow_up 是 delta (dx, dy)
            sampling_grid = base_grid + flow_up.permute(0, 2, 3, 1)
            
            # 4. 执行 Warping
            src_flat = src_imgs.reshape(B*T, C, H, W)
            # 使用 reflection padding 缓解边界消失问题，与模型推理时一致
            warped_flat = F.grid_sample(src_flat, sampling_grid, mode='bilinear', padding_mode='reflection', align_corners=False)
            warped = warped_flat.view(B, T, C, H, W)
            
            # 5. 计算 L1 Loss
            if mask is not None:
                loss_warp = F.l1_loss(warped * mask, tgt_imgs * mask)
            else:
                loss_warp = F.l1_loss(warped, tgt_imgs)
        
        return loss_cons, loss_warp

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
        # [修改] 启用非对称物理约束 + Warp Loss
        self.loss_physics = PhysicsConstraintsLoss(under_penalty=2.0)
        
        # 2. 定义可学习参数 (Learnable Weights)
        # s 代表 log(variance)。初始化为 0 意味着初始方差为 1，初始权重为 0.5。
        # 形状为 5，分别对应: [MSE, CSI, Spectral, Conservation, Warp]
        self.params = nn.Parameter(torch.zeros(5)) 
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                flow: Optional[torch.Tensor] = None, 
                prev_frame: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None, 
                **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        # 维度兼容性处理
        if pred.ndim == 5: pred = pred.squeeze(2)
        if target.ndim == 5: target = target.squeeze(2)
        if prev_frame is not None and prev_frame.ndim == 5: prev_frame = prev_frame.squeeze(2)
        if mask is not None and mask.ndim == 5: mask = mask.squeeze(2)

        # 1. 准备时间权重 (仅增强 MSE)
        extra_weights = None
        if self.use_temporal_weight:
            T = pred.shape[1]
            # 线性增加权重 (1.0 -> 2.0)，越远的时刻预测越难，但业务上同样重要
            extra_weights = torch.linspace(1.0, 2.0, steps=T, device=pred.device).view(1, T, 1, 1)

        # 2. 计算各项原始 Loss (Raw Values)
        l_mse = self.loss_mse(pred, target, mask, extra_weights)
        l_csi = self.loss_csi(pred, target, mask)
        l_spec = self.loss_spectral(pred, target, mask)
        
        # [修改] 传入 flow 和 prev_frame，同时获取 Cons Loss 和 Warp Loss
        l_cons, l_warp = self.loss_physics(pred, target, flow, prev_frame, mask)
        
        # 将所有 Loss 堆叠: [5]
        losses = torch.stack([l_mse, l_csi, l_spec, l_cons, l_warp])
        
        # 3. 自动加权计算 (Automatic Weighting)
        # s: 可学习的不确定性参数
        # [增强] 增加 Clamp 防止数值爆炸 (s过大导致exp(-s)过小尚可，但s过小导致exp(-s)爆炸很危险)
        s = self.params.clamp(min=-10.0, max=10.0).float()
        losses = losses.float()
        
        # precision: 相当于权重 (1 / sigma^2)
        precision = torch.exp(-s)
        
        # 贝叶斯多任务损失公式: L = 0.5 * (precision * raw_loss + log_variance)
        weighted_losses = 0.5 * (precision * losses + s)
        
        total_loss = weighted_losses.sum()
        
        # 4. 构建返回字典 (用于监控)
        loss_dict = {
            'total': total_loss,
            # --- 原始 Loss (用于观察物理指标绝对值) ---
            'mse_raw': l_mse.detach(),
            'csi_raw': l_csi.detach(),
            'spec_raw': l_spec.detach(),
            'cons_raw': l_cons.detach(),
            'warp_raw': l_warp.detach(), # [新增] 监控平流误差
            # --- 学习到的实际权重 (0.5 * exp(-s)) ---
            # 观察这些值的变化可以看出模型当前关注哪些任务
            'w_mse': 0.5 * precision[0].detach(),
            'w_csi': 0.5 * precision[1].detach(),
            'w_spec': 0.5 * precision[2].detach(),
            'w_cons': 0.5 * precision[3].detach(),
            'w_warp': 0.5 * precision[4].detach(), # [新增] Warp 权重
            # --- 不确定性参数 s ---
            's_mse': s[0].detach(),
            's_csi': s[1].detach()
        }
        
        return total_loss, loss_dict