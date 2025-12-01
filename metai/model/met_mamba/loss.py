# metai/model/met_mamba/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Dict, Tuple

# 引入 TorchMetrics 以获得更稳定的 MS-SSIM 计算
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

# ==========================
# 物理常量与工具
# ==========================
MM_MAX = 30.0
LOG_NORM_FACTOR = math.log(MM_MAX + 1)

def mm_to_lognorm(mm_val: float) -> float:
    """将物理降水值 (mm) 转换为 Log 归一化值 (0-1)"""
    return math.log(mm_val + 1) / LOG_NORM_FACTOR

# ==========================
# 基础 Loss 实现
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
        
        # 1. 静态分级权重
        indices = torch.bucketize(target, self.thresholds)
        w_static = self.static_weights[indices]
        
        # 2. 虚警惩罚
        if self.false_alarm_penalty > 0:
            is_false_alarm = (target < self.rain_start) & (pred > self.rain_start)
            if is_false_alarm.any():
                w_static = w_static.clone()
                w_static[is_false_alarm] *= self.false_alarm_penalty

        # 3. 动态误差聚焦
        w_dynamic = (1.0 + self.alpha * l1_diff).pow(self.gamma)
        
        loss = l1_diff * w_static * w_dynamic
        
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-6)
        
        return loss.mean()

class CorrLoss(nn.Module):
    """
    平滑皮尔逊相关系数损失 (Smoothed Pearson Correlation Loss)。
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

        if p.numel() < 5: 
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

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

class DiceLoss(nn.Module):
    """
    软 Dice 损失 (Soft Dice Loss)。
    针对多个降水阈值优化 TS 评分。
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
            # 使用 Sigmoid 近似阶跃函数，使其可微
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
    混合损失函数 (Hybrid Loss)。
    集成 Focal, MS-SSIM (via torchmetrics), Corr, Dice 四大分量。
    """
    def __init__(self, 
                 weight_focal: float = 1.0, 
                 weight_msssim: float = 1.0, 
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
            'msssim': weight_msssim,
            'corr': weight_corr,
            'dice': weight_dice
        }
        
        # 1. Focal Loss
        self.loss_focal = FocalLoss(
            weights_val=intensity_weights, alpha=focal_alpha, gamma=focal_gamma,
            false_alarm_penalty=false_alarm_penalty
        )
        
        # 2. MS-SSIM Loss (替换为 torchmetrics)
        # data_range=1.0: 对应归一化后的数据范围
        # kernel_size=11: 默认窗口大小
        self.loss_msssim = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=1.0, 
            kernel_size=11,
            reduction='elementwise_mean'
        )
        
        # 3. Corr Loss
        self.loss_corr = CorrLoss(eps=corr_smooth_eps)
        
        # 4. Dice Loss
        self.loss_dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 计算各个分项 Loss
        l_focal = self.loss_focal(pred, target, mask)
        l_corr = self.loss_corr(pred, target, mask)
        l_dice = self.loss_dice(pred, target, mask)
        
        # === MS-SSIM 计算 ===
        # 需要将 (B, T, C, H, W) 或 (B, T, H, W) 展平为 (N, C, H, W)
        p_ssim = pred
        t_ssim = target
        
        # 1. 确保有 Channel 维度: (B, T, H, W) -> (B, T, 1, H, W)
        if p_ssim.ndim == 4:
            p_ssim = p_ssim.unsqueeze(2)
            t_ssim = t_ssim.unsqueeze(2)
            
        # 2. 展平时间维度: (B, T, C, H, W) -> (B*T, C, H, W)
        if p_ssim.ndim == 5:
            b, t, c, h, w = p_ssim.shape
            p_ssim = p_ssim.reshape(-1, c, h, w)
            t_ssim = t_ssim.reshape(-1, c, h, w)
        
        # 3. 计算 Metric 并转为 Loss (Loss = 1 - SSIM)
        # torchmetrics 内部处理了数值稳定性
        ssim_score = self.loss_msssim(p_ssim, t_ssim)
        l_msssim = 1.0 - ssim_score

        # 加权求和
        total = (self.weights['focal'] * l_focal + 
                 self.weights['msssim'] * l_msssim + 
                 self.weights['corr'] * l_corr + 
                 self.weights['dice'] * l_dice)
        
        loss_dict = {
            'total': total,
            'focal': l_focal,
            'msssim': l_msssim,
            'corr': l_corr,
            'dice': l_dice
        }
        
        return total, loss_dict