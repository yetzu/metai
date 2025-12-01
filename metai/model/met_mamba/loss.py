# metai/model/met_mamba/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Dict, Tuple

# 尝试导入 torchmetrics 以获得稳定的 MS-SSIM 计算
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
    严格对齐官方评分表的阈值、强度权重和时效权重。
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        # 官方评分表配置
        thresholds_mm = [0.1, 1.0, 2.0, 5.0, 8.0]
        weights_intensity = [0.1, 0.1, 0.2, 0.25, 0.35]
        
        # 转换为 Log 归一化空间
        self.register_buffer('thresholds', torch.tensor([mm_to_lognorm(t) for t in thresholds_mm]))
        self.register_buffer('intensity_weights', torch.tensor(weights_intensity))
        
        # 时效权重 (6min -> 120min)，第10帧(60min)权重最高
        time_weights_raw = [
            0.0075, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
            0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.0075, 0.005 
        ]
        # [1, T, 1, 1] 方便广播
        self.register_buffer('time_weights', torch.tensor(time_weights_raw).view(1, -1, 1, 1))
        
        self.smooth = smooth
        self.temperature = 50.0 # 控制 Sigmoid 陡峭程度，模拟阶跃函数

    def forward(self, pred, target, mask=None):
        T = pred.shape[1]
        # 动态截取当前 T 的权重
        curr_time_w = self.time_weights[:, :T, :, :]
        # 归一化权重，使其均值为1，避免 Loss 幅度随 T 变化剧烈
        curr_time_w = curr_time_w / (curr_time_w.mean() + 1e-8)
        
        total_loss = 0.0
        total_w_sum = 0.0

        for i, thresh in enumerate(self.thresholds):
            w = self.intensity_weights[i]
            
            # 软二值化: (pred - thresh) * temp
            # 当 pred > thresh 时 sigmoid -> 1
            pred_score = torch.sigmoid((pred - thresh) * self.temperature)
            target_score = (target > thresh).float()
            
            if mask is not None:
                pred_score = pred_score * mask
                target_score = target_score * mask
            
            # 计算 Soft CSI (Intersection over Union)
            intersection = (pred_score * target_score).sum(dim=(-2, -1))
            union = pred_score.sum(dim=(-2, -1)) + target_score.sum(dim=(-2, -1)) - intersection
            
            csi = (intersection + self.smooth) / (union + self.smooth)
            loss_map = 1.0 - csi # [B, T]
            
            # 应用时效权重 (在 T 维度平均)
            weighted_loss_t = (loss_map * curr_time_w.squeeze(-1).squeeze(-1)).mean()
            
            total_loss += weighted_loss_t * w
            total_w_sum += w
            
        return total_loss / (total_w_sum + 1e-8)


class WeightedEvolutionLoss(nn.Module):
    """
    [时空一致性] 物理感知演变损失
    约束 dI/dt (预测的变化量应等于真实的变化量)，并对强回波区加权。
    自动兼容 Curriculum Learning (Mask 机制)。
    """
    def __init__(self, weight_scale=5.0):
        super().__init__()
        self.weight_scale = weight_scale

    def forward(self, pred, target, mask=None):
        if pred.shape[1] < 2: return torch.tensor(0.0, device=pred.device)

        # 时间差分: P(t+1) - P(t)
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        
        diff_error = torch.abs(pred_diff - target_diff)
        
        # 动态加权: 强回波区的变化更重要
        weight_map = 1.0 + self.weight_scale * target[:, 1:]
        
        if mask is not None:
            # 维度对齐: (B, T, H, W) -> squeeze -> (B, T, H, W)
            if mask.dim() == 5: mask = mask.squeeze(2)
            
            # [关键] 取 mask[:, 1:] 对应 t+1 时刻的有效性
            # 如果课程学习屏蔽了 t+1，这里的 diff_error 也会被屏蔽
            mask_t = mask[:, 1:]
            loss = (diff_error * weight_map * mask_t).sum() / (mask_t.sum() + 1e-8)
        else:
            loss = (diff_error * weight_map).mean()
            
        return loss


class FocalLoss(nn.Module):
    """
    [基础回归] Focal L1 Loss
    结合静态强度加权与动态误差聚焦。
    """
    def __init__(self, 
                 # [调整] 参考 SimVP 策略，大幅调大高强度区(>8.0mm)的权重，设为 20.0
                 weights_val: Tuple[float, ...] = (0.1, 1.0, 2.0, 5.0, 10.0, 20.0),
                 alpha: float = 2.0,
                 gamma: float = 1.0,
                 false_alarm_penalty: float = 5.0, # 虚警惩罚
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
        l1_diff = torch.abs(pred - target)
        
        # 1. 静态分级权重 (Bucketize)
        indices = torch.bucketize(target, self.thresholds)
        w_static = self.static_weights[indices]
        
        # 2. 虚警惩罚 (False Alarm Penalty)
        if self.false_alarm_penalty > 0:
            is_false_alarm = (target < self.rain_start) & (pred > self.rain_start)
            if is_false_alarm.any():
                w_static = w_static.clone()
                w_static[is_false_alarm] *= self.false_alarm_penalty

        # 3. 动态聚焦 (Focal)
        w_dynamic = (1.0 + self.alpha * l1_diff).pow(self.gamma)
        
        loss = l1_diff * w_static * w_dynamic
        
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-6)
        return loss.mean()


class CorrLoss(nn.Module):
    """
    [分布一致性] 平滑皮尔逊相关系数 Loss
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
    """
    [SOTA] 物理感知混合损失函数
    集成 Focal(L1), MS-SSIM(TorchMetrics), Soft-CSI(SimVP), Evolution(SimVP), Corr
    """
    def __init__(self, 
                 weight_focal: float = 1.0, 
                 weight_msssim: float = 1.0, 
                 weight_csi: float = 1.0,    # 替换原 Dice
                 weight_evo: float = 0.5,    # 新增 Evolution
                 weight_corr: float = 0.5,   # 保留 Corr
                 **kwargs):
        super().__init__()
        
        self.weights = {
            'focal': weight_focal, 
            'msssim': weight_msssim,
            'csi': weight_csi,
            'evo': weight_evo,
            'corr': weight_corr
        }
        
        # 1. Focal Loss (增强版权重)
        self.loss_focal = FocalLoss(false_alarm_penalty=5.0)
        
        # 2. MS-SSIM Loss (TorchMetrics - Stable)
        if TORCHMETRICS_AVAILABLE:
            self.loss_msssim = MultiScaleStructuralSimilarityIndexMeasure(
                data_range=1.0, kernel_size=11, reduction='elementwise_mean'
            )
        else:
            self.loss_msssim = None
        
        # 3. Soft-CSI Loss (SimVP 移植, 竞赛优化)
        self.loss_csi = WeightedScoreSoftCSILoss()
        
        # 4. Evolution Loss (SimVP 移植, 时序一致性)
        self.loss_evo = WeightedEvolutionLoss(weight_scale=5.0)

        # 5. Corr Loss (分布一致性)
        self.loss_corr = CorrLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Focal Loss (支持 Mask)
        l_focal = self.loss_focal(pred, target, mask)
        
        # Soft-CSI Loss (支持 Mask, 竞赛核心)
        l_csi = self.loss_csi(pred, target, mask)
        
        # Evolution Loss (支持 Mask, 抑制闪烁)
        l_evo = self.loss_evo(pred, target, mask)
        
        # Corr Loss (支持 Mask)
        l_corr = self.loss_corr(pred, target, mask)
        
        # MS-SSIM Loss (维度适配 + 数值稳定)
        l_msssim = torch.tensor(0.0, device=pred.device)
        if self.loss_msssim is not None:
            p_ssim = pred
            t_ssim = target
            
            # 1. 维度适配: (B, T, H, W) -> (B, T, 1, H, W)
            if p_ssim.ndim == 4:
                p_ssim = p_ssim.unsqueeze(2)
                t_ssim = t_ssim.unsqueeze(2)
                
            # 2. 维度展平: (B, T, C, H, W) -> (B*T, C, H, W)
            # TorchMetrics 期望 Batch 维度的图像
            if p_ssim.ndim == 5:
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