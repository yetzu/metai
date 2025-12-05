# metai/model/met_mamba/loss.py

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Tuple

# ==============================================================================
# 物理常量与工具
# ==============================================================================
PHYSICAL_MAX = 30.0  
LOG_NORM_FACTOR = math.log(PHYSICAL_MAX + 1)

def mm_to_lognorm(mm_val: float) -> float:
    """将物理降水值(mm)转换为对数归一化值(0-1)。"""
    return math.log(mm_val + 1) / LOG_NORM_FACTOR

# ==============================================================================
# 1. 强度加权 L1 损失 (Weighted L1 Loss)
# ==============================================================================
class WeightedL1Loss(nn.Module):
    """
    [回归损失] 强度加权 L1 Loss。
    
    目的：
        对抗"平均值模糊"现象。通过给予强降水像素更高的权重，迫使模型拟合极值。
        
    公式：
        Weight = 1.0 + 10.0 * (Target ** 2)
        Loss = Mean( |Pred - Target| * Weight )
    """
    def __init__(self, base_weight=1.0, scale_weight=10.0):
        super().__init__()
        self.base_weight = base_weight
        self.scale_weight = scale_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred: 预测值 [B, H, W], 范围 [0, 1]
            target: 实况值 [B, H, W], 范围 [0, 1]
            mask: 有效区域掩码 [B, H, W]
        """
        diff = torch.abs(pred - target)
        
        # 动态权重：Target 越大，权重呈指数级增长
        weights = self.base_weight + self.scale_weight * (target ** 2)
        loss_map = diff * weights
        
        if mask is not None:
            loss_map = loss_map * mask
            return loss_map.sum() / (mask.sum() + 1e-8)
        
        return loss_map.mean()

# ==============================================================================
# 2. 相关性损失 (Correlation Loss)
# ==============================================================================
class CorrLoss(nn.Module):
    """
    [结构损失] 1 - Pearson 相关系数。
    
    目的：
        优化评分公式中的 Score_k ~ sqrt(exp(R-1)) 项。
        防止模型输出全0或常数场（此时方差为0，会导致 R 无法计算或极低）。
        
    公式：
        Loss = 1 - R (R 范围 -1 到 1，目标是 1)
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            pred = pred * mask
            target = target * mask
        
        # 展平 spatial 维度: [B, H, W] -> [B, N]
        pred_flat = pred.view(pred.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)

        # 中心化 (Substract Mean)
        pred_mean = pred_flat.mean(dim=1, keepdim=True)
        target_mean = target_flat.mean(dim=1, keepdim=True)
        pred_c = pred_flat - pred_mean
        target_c = target_flat - target_mean

        # 计算相关系数 R
        num = (pred_c * target_c).sum(dim=1)
        den = torch.sqrt((pred_c ** 2).sum(dim=1) * (target_c ** 2).sum(dim=1)) + self.eps
        r = num / den
        
        return 1.0 - r.mean()

# ==============================================================================
# 3. 多阈值 CSI 损失 (CSI Loss)
# ==============================================================================
class CSILoss(nn.Module):
    """
    [指标损失] 多阈值加权软 CSI。
    
    目的：
        直接优化评分表2中的 Critical Success Index。
        严格按照官方规定的 5 个分级阈值和权重进行加权。
        
    阈值设置 (mm):
        [0.1, 1.0, 2.0, 5.0, 8.0]
    对应权重:
        [0.1, 0.1, 0.2, 0.25, 0.35]
    """
    def __init__(self):
        super().__init__()
        # 计算 Log-Space 下的阈值
        log_factor = math.log(PHYSICAL_MAX + 1)
        th_mm = [0.1, 1.0, 2.0, 5.0, 8.0]
        th_norm = [math.log(x + 1) / log_factor for x in th_mm]
        
        # 注册为 Buffer，随模型保存但不更新梯度
        self.register_buffer('thresholds', torch.tensor(th_norm))
        self.register_buffer('weights', torch.tensor([0.1, 0.1, 0.2, 0.25, 0.35]))
        
        self.temperature = 30.0 # Sigmoid 温度系数，越大越接近阶跃函数

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = 0.0
        # 遍历每个阈值计算 Soft CSI
        for i, thresh in enumerate(self.thresholds):
            w = self.weights[i]
            
            # Soft Classification: Sigmoid 近似
            pred_score = torch.sigmoid((pred - thresh) * self.temperature)
            target_score = (target > thresh).float() # Target 保持硬阈值
            
            if mask is not None:
                pred_score = pred_score * mask
                target_score = target_score * mask
            
            # Soft Confusion Matrix
            intersection = (pred_score * target_score).sum(dim=(-2, -1))
            union = pred_score.sum(dim=(-2, -1)) + target_score.sum(dim=(-2, -1)) - intersection
            
            # Loss = Weighted sum of (1 - CSI)
            csi = (intersection + 1e-6) / (union + 1e-6)
            loss += (1.0 - csi.mean()) * w
            
        return loss

# ==============================================================================
# 4. 混合总损失 (HybridLoss)
# ==============================================================================
class HybridLoss(nn.Module):
    """
    [总损失] 竞赛规则对齐损失函数 (HybridLoss)。
    
    特性：
    1. **时效加权**：严格按照表1对 20 个预测帧进行加权 (60min权重最大)。
    2. **分级加权**：通过 CSILoss 对强降水给予更高权重。
    3. **多维优化**：同时优化数值误差(L1)、空间结构(Corr)和分类指标(CSI)。
    
    参数建议：
        lambda_mae=10.0 (平衡 L1 较小的数值量级)
        lambda_csi=1.0
        lambda_corr=1.0
    """
    def __init__(self, lambda_mae=10.0, lambda_csi=1.0, lambda_corr=1.0):
        super().__init__()
        self.lambda_mae = lambda_mae 
        self.lambda_csi = lambda_csi
        self.lambda_corr = lambda_corr
        
        # 初始化子损失
        self.loss_mae = WeightedL1Loss()
        self.loss_csi = CSILoss()
        self.loss_corr = CorrLoss()
        
        # 表1：预报时效权重 (6min - 120min)
        # 权重分布：两头低(0.0075)，中间高(0.1)，第10帧(60min)最为重要
        time_weights = [
            0.0075, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
            0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.0075, 0.005
        ]
        self.register_buffer('time_weights', torch.tensor(time_weights))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                mask: Optional[torch.Tensor] = None, 
                current_epoch: int = 0,
                **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算总损失。
        
        Args:
            pred: [B, T, C, H, W]
            target: [B, T, C, H, W]
            
        Returns:
            total_loss: 标量 Tensor
            loss_dict: 用于日志记录的分项 Loss
        """
        # 1. 数值钳位 (防止 Log Space 溢出及数值不稳定)
        pred = torch.clamp(pred, 0.0, 1.0)
        
        B, T, C, H, W = pred.shape
        
        # 2. 准备时间权重
        if T == len(self.time_weights):
            tw = self.time_weights
        else:
            # 维度不匹配时的兜底策略 (均匀权重)
            tw = torch.ones(T, device=pred.device) / T
            
        total_loss = 0.0
        
        # 监控变量
        log_mae = 0.0
        log_csi = 0.0
        log_corr = 0.0
        
        # 3. 逐帧计算 (Time-Distributed Calculation)
        for t in range(T):
            # 提取单帧 [B, H, W] (假设 C=1)
            p_t = pred[:, t, 0]
            t_t = target[:, t, 0]
            m_t = mask[:, t, 0] if mask is not None else None
            
            # 放大时间权重以保持梯度量级 (*20.0 使均值约为1)
            w_t = tw[t] * 20.0 
            
            # 计算各子 Loss
            l_mae = self.loss_mae(p_t, t_t, m_t)
            l_csi = self.loss_csi(p_t, t_t, m_t)
            l_corr = self.loss_corr(p_t, t_t, m_t)
            
            # 加权聚合
            frame_loss = self.lambda_mae * l_mae + \
                         self.lambda_csi * l_csi + \
                         self.lambda_corr * l_corr
            
            # 累加总 Loss
            total_loss += frame_loss * w_t
            
            # 记录日志
            with torch.no_grad():
                log_mae += l_mae * w_t
                log_csi += l_csi * w_t
                log_corr += l_corr * w_t

        loss_dict = {
            'total_loss': total_loss,
            'l_mae': log_mae.detach(),
            'l_csi': log_csi.detach(),
            'l_corr': log_corr.detach()
        }
        
        return total_loss, loss_dict