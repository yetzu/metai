# metai/model/met_mamba/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Dict, Tuple

# 尝试导入 torchmetrics
try:
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("Warning: torchmetrics not found. MS-SSIM loss will be skipped.")

# ==========================
# 物理常量与工具
# ==========================
MM_MAX = 30.0
LOG_NORM_FACTOR = math.log(MM_MAX + 1)

def mm_to_lognorm(mm_val: float) -> float:
    """将物理降水值 (mm) 转换为 Log 归一化值 (0-1)"""
    return math.log(mm_val + 1) / LOG_NORM_FACTOR

# ==========================
# 核心 Loss 组件
# ==========================

class WeightedScoreSoftCSILoss(nn.Module):
    """
    [竞赛核心] 软 CSI/TS 评分损失函数
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        thresholds_mm = [0.1, 1.0, 2.0, 5.0, 8.0]
        weights_intensity = [0.1, 0.1, 0.2, 0.25, 0.35]
        
        self.register_buffer('thresholds', torch.tensor([mm_to_lognorm(t) for t in thresholds_mm]))
        self.register_buffer('intensity_weights', torch.tensor(weights_intensity))
        
        time_weights_raw = [
            0.0075, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
            0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.0075, 0.005 
        ]
        self.register_buffer('time_weights', torch.tensor(time_weights_raw).view(1, -1, 1, 1))
        
        self.smooth = smooth
        self.temperature = 50.0 

    def forward(self, pred, target, mask=None):
        # [修复] 统一转为 4D [B, T, H, W]
        if pred.dim() == 5: pred = pred.squeeze(2)
        if target.dim() == 5: target = target.squeeze(2)
        if mask is not None and mask.dim() == 5: mask = mask.squeeze(2)

        T = pred.shape[1]
        curr_time_w = self.time_weights[:, :T, :, :]
        curr_time_w = curr_time_w / (curr_time_w.mean() + 1e-8)
        
        total_loss = 0.0
        total_w_sum = 0.0

        for i, thresh in enumerate(self.thresholds):
            w = self.intensity_weights[i]
            
            pred_score = torch.sigmoid((pred - thresh) * self.temperature)
            target_score = (target > thresh).float()
            
            if mask is not None:
                pred_score = pred_score * mask
                target_score = target_score * mask
            
            intersection = (pred_score * target_score).sum(dim=(-2, -1))
            union = pred_score.sum(dim=(-2, -1)) + target_score.sum(dim=(-2, -1)) - intersection
            
            csi = (intersection + self.smooth) / (union + self.smooth)
            loss_map = 1.0 - csi 
            
            weighted_loss_t = (loss_map * curr_time_w.squeeze(-1).squeeze(-1)).mean()
            
            total_loss += weighted_loss_t * w
            total_w_sum += w
            
        return total_loss / (total_w_sum + 1e-8)


class WeightedEvolutionLoss(nn.Module):
    """
    [时空一致性] 物理感知演变损失
    """
    def __init__(self, weight_scale=5.0):
        super().__init__()
        self.weight_scale = weight_scale

    def forward(self, pred, target, mask=None):
        # [修复] 关键修复：统一转为 4D [B, T, H, W]，避免 5D 与 4D 广播错误
        if pred.dim() == 5: pred = pred.squeeze(2)
        if target.dim() == 5: target = target.squeeze(2)
        if mask is not None and mask.dim() == 5: mask = mask.squeeze(2)

        if pred.shape[1] < 2: return torch.tensor(0.0, device=pred.device)

        # 时间差分: [B, T-1, H, W]
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        
        diff_error = torch.abs(pred_diff - target_diff)
        
        weight_map = 1.0 + self.weight_scale * target[:, 1:]
        
        if mask is not None:
            mask_t = mask[:, 1:]
            loss = (diff_error * weight_map * mask_t).sum() / (mask_t.sum() + 1e-8)
        else:
            loss = (diff_error * weight_map).mean()
            
        return loss


class FocalLoss(nn.Module):
    """
    [基础回归] Focal L1 Loss
    """
    def __init__(self, 
                 weights_val: Tuple[float, ...] = (0.1, 1.0, 2.0, 5.0, 10.0, 20.0),
                 alpha: float = 2.0,
                 gamma: float = 1.0,
                 false_alarm_penalty: float = 5.0,
                 loss_scale: float = 10.0):
        super().__init__()
        
        thresholds_mm = [0.1, 1.0, 2.0, 5.0, 8.0]
        self.register_buffer('thresholds', torch.tensor([mm_to_lognorm(t) for t in thresholds_mm]))
        self.register_buffer('static_weights', torch.tensor([w * loss_scale for w in weights_val]))
        
        self.alpha = alpha
        self.gamma = gamma
        self.false_alarm_penalty = false_alarm_penalty
        self.rain_start = mm_to_lognorm(0.1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Focal Loss 可以兼容 5D 或 4D，无需强制 squeeze，保持原样即可
        l1_diff = torch.abs(pred - target)
        
        indices = torch.bucketize(target, self.thresholds)
        w_static = self.static_weights[indices]
        
        if self.false_alarm_penalty > 0:
            is_false_alarm = (target < self.rain_start) & (pred > self.rain_start)
            if is_false_alarm.any():
                w_static = w_static.clone()
                w_static[is_false_alarm] *= self.false_alarm_penalty

        w_dynamic = (1.0 + self.alpha * l1_diff).pow(self.gamma)
        loss = l1_diff * w_static * w_dynamic
        
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-6)
        return loss.mean()


class CorrLoss(nn.Module):
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

        if p.numel() < 5: 
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        p_sub = p - p.mean()
        t_sub = t - t.mean()

        cov = (p_sub * t_sub).sum()
        var_p = (p_sub.pow(2)).sum()
        var_t = (t_sub.pow(2)).sum()
        
        denom = torch.sqrt(var_p + self.eps) * torch.sqrt(var_t + self.eps)
        r = cov / (denom + 1e-8)
        return 1.0 - torch.clamp(r, -1.0, 1.0)


# ==========================
# 混合 Loss (Hybrid)
# ==========================

class HybridLoss(nn.Module):
    def __init__(self, 
                 weight_focal: float = 1.0, 
                 weight_msssim: float = 1.0, 
                 weight_csi: float = 1.0, 
                 weight_evo: float = 0.5, 
                 weight_corr: float = 0.5, 
                 **kwargs):
        super().__init__()
        
        self.weights = {
            'focal': weight_focal, 
            'msssim': weight_msssim,
            'csi': weight_csi,
            'evo': weight_evo,
            'corr': weight_corr
        }
        
        self.loss_focal = FocalLoss(false_alarm_penalty=5.0)
        
        if TORCHMETRICS_AVAILABLE:
            self.loss_msssim = MultiScaleStructuralSimilarityIndexMeasure(
                data_range=1.0, kernel_size=11, reduction='elementwise_mean'
            )
        else:
            self.loss_msssim = None
        
        self.loss_csi = WeightedScoreSoftCSILoss()
        self.loss_evo = WeightedEvolutionLoss(weight_scale=5.0)
        self.loss_corr = CorrLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 计算 Loss 分量
        l_focal = self.loss_focal(pred, target, mask)
        l_csi = self.loss_csi(pred, target, mask)
        l_evo = self.loss_evo(pred, target, mask)
        l_corr = self.loss_corr(pred, target, mask)
        
        l_msssim = torch.tensor(0.0, device=pred.device)
        if self.loss_msssim is not None:
            p_ssim = pred
            t_ssim = target
            
            # SSIM 需要 [N, C, H, W]
            if p_ssim.ndim == 4: # [B, T, H, W] -> [B, T, 1, H, W]
                p_ssim = p_ssim.unsqueeze(2)
                t_ssim = t_ssim.unsqueeze(2)
            if p_ssim.ndim == 5: # [B, T, 1, H, W] -> [B*T, 1, H, W]
                b, t, c, h, w = p_ssim.shape
                p_ssim = p_ssim.reshape(-1, c, h, w)
                t_ssim = t_ssim.reshape(-1, c, h, w)
            
            ssim_score = self.loss_msssim(p_ssim, t_ssim)
            l_msssim = 1.0 - ssim_score

        # 加权求和
        total = (self.weights['focal'] * l_focal + 
                 self.weights['msssim'] * l_msssim + 
                 self.weights['csi'] * l_csi + 
                 self.weights['evo'] * l_evo + 
                 self.weights['corr'] * l_corr)
        
        loss_dict = {
            'total': total,
            'focal': l_focal,
            'msssim': l_msssim,
            'csi': l_csi,
            'evo': l_evo,
            'corr': l_corr
        }
        
        return total, loss_dict