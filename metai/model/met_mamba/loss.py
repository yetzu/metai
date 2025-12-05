# metai/model/met_mamba/loss.py

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Tuple

# ==============================================================================
# 物理常量与工具
# ==============================================================================
PHYSICAL_MAX = 30.0  

# ==============================================================================
# 1. 强度加权 L1 损失 (Weighted L1 Loss)
# ==============================================================================
class WeightedL1Loss(nn.Module):
    """
    [回归损失] 强度加权 L1 Loss。
    公式：Weight = 1.0 + 10.0 * (Target ** 2)
    """
    def __init__(self, base_weight=1.0, scale_weight=10.0):
        super().__init__()
        self.base_weight = base_weight
        self.scale_weight = scale_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        diff = torch.abs(pred - target)
        # 动态权重：Target 越大，权重呈指数级增长，迫使模型关注强降水
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
    防止模型输出全0或常数场。
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            pred = pred * mask
            target = target * mask
        
        # [B, H, W] -> [B, N]
        pred_flat = pred.view(pred.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)

        # Centering
        pred_c = pred_flat - pred_flat.mean(dim=1, keepdim=True)
        target_c = target_flat - target_flat.mean(dim=1, keepdim=True)

        # Pearson R
        num = (pred_c * target_c).sum(dim=1)
        den = torch.sqrt((pred_c ** 2).sum(dim=1) * (target_c ** 2).sum(dim=1)) + self.eps
        r = num / den
        
        return 1.0 - r.mean()

# ==============================================================================
# 3. 多阈值 CSI 损失 (CSI Loss)
# ==============================================================================
class CSILoss(nn.Module):
    """
    [指标损失] 多阈值加权软 CSI (Soft-CSI)。
    直接优化竞赛指标。
    """
    def __init__(self):
        super().__init__()
        # 计算 Log-Space 下的阈值 (0.1mm ~ 8.0mm)
        log_factor = math.log(PHYSICAL_MAX + 1)
        th_mm = [0.1, 1.0, 2.0, 5.0, 8.0]
        th_norm = [math.log(x + 1) / log_factor for x in th_mm]
        
        self.register_buffer('thresholds', torch.tensor(th_norm))
        self.register_buffer('weights', torch.tensor([0.1, 0.1, 0.2, 0.25, 0.35]))
        self.temperature = 30.0 # Sigmoid 温度系数

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = 0.0
        for i, thresh in enumerate(self.thresholds):
            w = self.weights[i]
            # Sigmoid 近似阶跃函数
            pred_score = torch.sigmoid((pred - thresh) * self.temperature)
            target_score = (target > thresh).float() 
            
            if mask is not None:
                pred_score = pred_score * mask
                target_score = target_score * mask
            
            # Soft Confusion Matrix
            intersection = (pred_score * target_score).sum(dim=(-2, -1))
            union = pred_score.sum(dim=(-2, -1)) + target_score.sum(dim=(-2, -1)) - intersection
            
            csi = (intersection + 1e-6) / (union + 1e-6)
            loss += (1.0 - csi.mean()) * w
            
        return loss

# ==============================================================================
# 4. 混合总损失 (HybridLoss)
# ==============================================================================
class HybridLoss(nn.Module):
    """
    [总损失] 
    """
    def __init__(self, weight_mae=10.0, weight_csi=1.0, weight_corr=1.0):
        super().__init__()
        self.weight_mae = weight_mae 
        self.weight_csi = weight_csi
        self.weight_corr = weight_corr
        
        self.loss_mae = WeightedL1Loss()
        self.loss_csi = CSILoss()
        self.loss_corr = CorrLoss()
        
        # 预报时效权重 (20帧)
        time_weights = [
            0.0075, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
            0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.0075, 0.005
        ]
        self.register_buffer('time_weights', torch.tensor(time_weights))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                mask: Optional[torch.Tensor] = None, 
                **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        # 1. 数值钳位 (稳定性)
        pred = torch.clamp(pred, 0.0, 1.0)
        
        B, T, C, H, W = pred.shape
        
        # 2. 准备时间权重
        if T == len(self.time_weights):
            tw = self.time_weights
        else:
            tw = torch.ones(T, device=pred.device) / T
            
        total_loss = 0.0
        log_mae, log_csi, log_corr = 0.0, 0.0, 0.0
        
        # 3. 逐帧计算
        for t in range(T):
            p_t = pred[:, t, 0]
            t_t = target[:, t, 0]
            m_t = mask[:, t, 0] if mask is not None else None
            
            w_t = tw[t] * 20.0 # 归一化系数，使权重均值约为1
            
            l_mae = self.loss_mae(p_t, t_t, m_t)
            l_csi = self.loss_csi(p_t, t_t, m_t)
            l_corr = self.loss_corr(p_t, t_t, m_t)
            
            frame_loss = self.weight_mae * l_mae + \
                         self.weight_csi * l_csi + \
                         self.weight_corr * l_corr
            
            total_loss += frame_loss * w_t
            
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