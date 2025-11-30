import torch
import torch.nn as nn
import math
from typing import Optional, List, Dict, Tuple

# ==========================
# 物理常量与工具
# ==========================
MM_MAX = 30.0
LOG_NORM_FACTOR = math.log(MM_MAX + 1)

def mm_to_lognorm(mm_val: float) -> float:
    """将物理降水值 (mm) 转换为 Log 归一化值 (0-1)"""
    return math.log(mm_val + 1) / LOG_NORM_FACTOR

# ==========================
# 子损失模块 (Sub-Losses)
# ==========================

class FocalLoss(nn.Module):
    """
    Focal L1 Loss: 结合静态强度加权与动态误差聚焦。
    解决强度预测不准、模糊及极值惩罚不足的问题。
    """
    def __init__(self, 
                 weights_val: Tuple[float, ...] = (0.1, 1.0, 1.2, 2.5, 3.5, 5.0),
                 alpha: float = 2.0,
                 gamma: float = 1.0,
                 false_alarm_penalty: float = 5.0,
                 loss_scale: float = 10.0):
        super().__init__()
        
        # 强度分级阈值 (mm -> log)
        thresholds_mm = [0.1, 1.0, 2.0, 5.0, 8.0]
        self.register_buffer('thresholds', torch.tensor([mm_to_lognorm(t) for t in thresholds_mm]))
        self.register_buffer('static_weights', torch.tensor([w * loss_scale for w in weights_val]))
        
        self.alpha = alpha
        self.gamma = gamma
        self.false_alarm_penalty = false_alarm_penalty
        self.rain_start = mm_to_lognorm(0.1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        l1_diff = torch.abs(pred - target)
        
        # 1. 静态分级权重 (Static Intensity Weighting)
        indices = torch.bucketize(target, self.thresholds)
        w_static = self.static_weights[indices]
        
        # 2. 虚警惩罚 (False Alarm Penalty)
        if self.false_alarm_penalty > 0:
            is_false_alarm = (target < self.rain_start) & (pred > self.rain_start)
            if is_false_alarm.any():
                w_static = w_static.clone()
                w_static[is_false_alarm] *= self.false_alarm_penalty

        # 3. 动态误差聚焦 (Focal Modulation)
        w_dynamic = (1.0 + self.alpha * l1_diff).pow(self.gamma)
        
        loss = l1_diff * w_static * w_dynamic
        
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-6)
        
        return loss.mean()


class CorrLoss(nn.Module):
    """
    Smoothed Pearson Correlation Loss.
    引入方差平滑项 (eps)，解决平坦区域梯度不稳定的问题。
    """
    def __init__(self, eps: float = 1e-4, ignore_zeros: bool = True):
        super().__init__()
        self.eps = eps
        self.ignore_zeros = ignore_zeros
        self.threshold = mm_to_lognorm(0.1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(pred, dtype=torch.bool)
        else:
            mask = mask.bool()
            
        if self.ignore_zeros:
            valid_rain = (pred >= self.threshold) | (target >= self.threshold)
            mask = mask & valid_rain

        p = pred[mask]
        t = target[mask]

        if p.numel() < 5: return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # 中心化
        p_sub = p - p.mean()
        t_sub = t - t.mean()

        # 计算相关性 (含平滑项)
        cov = (p_sub * t_sub).sum()
        var_p = (p_sub.pow(2)).sum()
        var_t = (t_sub.pow(2)).sum()
        
        denom = torch.sqrt(var_p + self.eps) * torch.sqrt(var_t + self.eps)
        
        r = cov / (denom + 1e-8)
        return 1.0 - torch.clamp(r, -1.0, 1.0)


class GradLoss(nn.Module):
    """
    Unified Spatio-Temporal Gradient Loss.
    Unified calculation of spatial gradient (texture) and temporal gradient (motion) differences.
    """
    def __init__(self, spatial_weight: float = 1.0, temporal_weight: float = 1.0):
        super().__init__()
        self.w_s = spatial_weight
        self.w_t = temporal_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = torch.tensor(0.0, device=pred.device)
        
        # 1. Spatial Gradient (Separate H and W calculation)
        if self.w_s > 0:
            # H/W differences
            p_dx = torch.abs(pred[..., :, 1:] - pred[..., :, :-1])
            p_dy = torch.abs(pred[..., 1:, :] - pred[..., :-1, :])
            t_dx = torch.abs(target[..., :, 1:] - target[..., :, :-1])
            t_dy = torch.abs(target[..., 1:, :] - target[..., :-1, :])
            
            diff_dx = torch.abs(p_dx - t_dx)
            diff_dy = torch.abs(p_dy - t_dy)
            
            if mask is not None:
                # Use multiplication (*) instead of bitwise AND (&) for Float tensors
                m_dx = mask[..., :, 1:] * mask[..., :, :-1]
                loss_dx = (diff_dx * m_dx).sum() / (m_dx.sum() + 1e-6)
                
                m_dy = mask[..., 1:, :] * mask[..., :-1, :]
                loss_dy = (diff_dy * m_dy).sum() / (m_dy.sum() + 1e-6)
                
                loss += (loss_dx + loss_dy) * self.w_s
            else:
                loss += (diff_dx.mean() + diff_dy.mean()) * self.w_s

        # 2. Temporal Gradient
        if self.w_t > 0 and pred.shape[1] > 1:
            p_dt = pred[:, 1:] - pred[:, :-1]
            t_dt = target[:, 1:] - target[:, :-1]
            
            gdl_t = torch.abs(p_dt - t_dt)
            
            if mask is not None:
                # Use multiplication (*) here as well
                m_t = mask[:, 1:] * mask[:, :-1]
                gdl_t = gdl_t * m_t
                loss += (gdl_t.sum() / (m_t.sum() + 1e-6)) * self.w_t
            else:
                loss += gdl_t.mean() * self.w_t
                
        return loss


class DiceLoss(nn.Module):
    """
    Soft Dice Loss (Optimized for TS Score).
    """
    def __init__(self, weights: List[float] = [0.1, 0.1, 0.2, 0.25, 0.35]):
        super().__init__()
        self.thresholds = [mm_to_lognorm(x) for x in [0.1, 1.0, 2.0, 5.0, 8.0]]
        self.weights = weights
        self.smooth = 1e-5
        self.temperature = 50.0 

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = torch.tensor(0.0, device=pred.device)
        
        for i, thresh in enumerate(self.thresholds):
            # Sigmoid 近似阶跃函数
            p_mask = torch.sigmoid((pred - thresh) * self.temperature)
            t_mask = (target >= thresh).float()
            
            if mask is not None:
                p_mask = p_mask * mask
                t_mask = t_mask * mask

            intersect = (p_mask * t_mask).sum()
            union = p_mask.sum() + t_mask.sum()
            
            dice = (2.0 * intersect + self.smooth) / (union + self.smooth)
            loss += (1.0 - dice) * self.weights[i]
            
        return loss / sum(self.weights)


# ==========================
# 主损失函数 (Hybrid)
# ==========================

class HybridLoss(nn.Module):
    """
    Hybrid Loss: 全能型混合损失调度器。
    集成 Focal, Grad, Corr, Dice 四大分量。
    """
    def __init__(self, 
                 weight_focal: float = 1.0, 
                 weight_grad: float = 10.0,
                 weight_corr: float = 0.5, 
                 weight_dice: float = 1.0,
                 # 参数配置
                 intensity_weights: Tuple[float, ...] = (0.1, 1.0, 1.2, 2.5, 3.5, 5.0),
                 focal_alpha: float = 2.0,
                 focal_gamma: float = 1.0,
                 false_alarm_penalty: float = 5.0,
                 corr_smooth_eps: float = 1e-4,
                 **kwargs):
        super().__init__()
        
        self.weights = {
            'focal': weight_focal, 
            'grad': weight_grad,
            'corr': weight_corr,
            'dice': weight_dice
        }
        
        # 1. Focal Loss (强度与模糊)
        self.loss_focal = FocalLoss(
            weights_val=intensity_weights,
            alpha=focal_alpha,
            gamma=focal_gamma,
            false_alarm_penalty=false_alarm_penalty
        )
        # 2. Grad Loss (纹理与运动)
        self.loss_grad = GradLoss(spatial_weight=1.0, temporal_weight=0.5)
        # 3. Corr Loss (分布一致性)
        self.loss_corr = CorrLoss(eps=corr_smooth_eps)
        # 4. Dice Loss (TS 评分)
        self.loss_dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 计算分项
        l_focal = self.loss_focal(pred, target, mask)
        l_grad = self.loss_grad(pred, target, mask)
        l_corr = self.loss_corr(pred, target, mask)
        l_dice = self.loss_dice(pred, target, mask)
        
        # 加权求和
        total = (self.weights['focal'] * l_focal + 
                 self.weights['grad'] * l_grad + 
                 self.weights['corr'] * l_corr + 
                 self.weights['dice'] * l_dice)
        
        loss_dict = {
            'total': total,
            'focal': l_focal,
            'grad': l_grad,
            'corr': l_corr,
            'dice': l_dice
        }
        
        return total, loss_dict