# metai/model/met_mamba/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Dict, Tuple
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import lpips

# ==============================================================================
# 常量与工具
# ==============================================================================

# 物理参数：必须与 Dataset 的归一化逻辑严格一致
MM_MAX = 30.0  
LOG_NORM_FACTOR = math.log(MM_MAX + 1)

def mm_to_lognorm(mm_val: float) -> float:
    """将物理降水值 (mm) 转换为对数归一化值 (0-1)。"""
    return math.log(mm_val + 1) / LOG_NORM_FACTOR

# ==============================================================================
# 核心 Loss 组件
# ==============================================================================

class BalancedMSELoss(nn.Module):
    """
    [归一化平衡 MSE]
    解决 'MSE Dominance' 的关键：
    通过 weights.sum() 归一化，允许我们使用极大的权重 (如 50x) 来改变梯度方向，
    而不会导致 Loss 数值爆炸。
    """
    def __init__(self, use_l1=True):
        super().__init__()
        self.use_l1 = use_l1
        
        # 定义阈值 (mm)
        self.thresh_light = 0.1
        self.thresh_mod   = 2.0
        self.thresh_heavy = 5.0
        
        # 预计算 Log 空间阈值
        self.log_thresh_light = mm_to_lognorm(self.thresh_light)
        self.log_thresh_mod = mm_to_lognorm(self.thresh_mod)
        self.log_thresh_heavy = mm_to_lognorm(self.thresh_heavy)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, extra_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_l1:
            diff = torch.abs(pred - target)
        else:
            diff = (pred - target) ** 2

        # 1. 激进的权重分配 (Aggressive Weighting)
        weights = torch.ones_like(target)
        weights[target >= self.log_thresh_light] = 5.0   
        weights[target >= self.log_thresh_mod]   = 20.0  
        weights[target >= self.log_thresh_heavy] = 50.0  

        loss_map = diff * weights
        
        # 2. 叠加额外权重 (如时间权重)
        if extra_weights is not None:
            loss_map = loss_map * extra_weights
            weights = weights * extra_weights

        # 3. 应用有效区域 Mask
        if mask is not None:
            loss_map = loss_map * mask
            weights = weights * mask

        # 4. [关键] 归一化：除以权重的总和
        return loss_map.sum() / (weights.sum() + 1e-8)


class DiceLoss(nn.Module):
    """
    [降水 Dice Loss]
    解决 'Zero Collapse' 的关键：
    将回归问题转化为二分类 (有雨/无雨) 问题。
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.threshold = mm_to_lognorm(0.1)
        # Scale=150.0 确保背景(0)的 Sigmoid 输出接近 0，减少噪声
        self.scale = 150.0 

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 软二值化
        pred_mask = torch.sigmoid((pred - self.threshold) * self.scale)
        target_mask = (target > self.threshold).float()
        
        if mask is not None:
            pred_mask = pred_mask * mask
            target_mask = target_mask * mask
            
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class GradientDifferenceLoss(nn.Module):
    """
    [梯度差分损失]
    保持边缘锐度。
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 空间梯度
        p_dy = torch.abs(pred[..., 1:, :] - pred[..., :-1, :])
        p_dx = torch.abs(pred[..., :, 1:] - pred[..., :, :-1])
        t_dy = torch.abs(target[..., 1:, :] - target[..., :-1, :])
        t_dx = torch.abs(target[..., :, 1:] - target[..., :, :-1])

        diff_dy = torch.abs(p_dy - t_dy) ** self.alpha
        diff_dx = torch.abs(p_dx - t_dx) ** self.alpha

        if mask is not None:
            m_dy = mask[..., 1:, :] * mask[..., :-1, :]
            m_dx = mask[..., :, 1:] * mask[..., :, :-1]
            loss_spatial = (diff_dy * m_dy).sum() / (m_dy.sum() + 1e-8) + \
                           (diff_dx * m_dx).sum() / (m_dx.sum() + 1e-8)
        else:
            loss_spatial = diff_dy.mean() + diff_dx.mean()

        return loss_spatial


class SpectralLoss(nn.Module):
    """
    [频域损失] (FACL)
    通过 FFT 约束频域幅度和相位，强制模型恢复高频纹理信息。
    """
    def __init__(self, loss_type='l2'):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            pred = pred * mask
            target = target * mask

        # 2D FFT (沿空间维度 H, W)
        fft_pred = torch.fft.rfft2(pred, dim=(-2, -1), norm='ortho')
        fft_target = torch.fft.rfft2(target, dim=(-2, -1), norm='ortho')

        # 1. 幅度损失 (清晰度)
        amp_pred = torch.abs(fft_pred)
        amp_target = torch.abs(fft_target)
        if self.loss_type == 'l2':
            loss_amp = F.mse_loss(amp_pred, amp_target)
        else:
            loss_amp = F.l1_loss(amp_pred, amp_target)

        # 2. 相关性损失 (结构位置)
        dot = torch.real(fft_pred * torch.conj(fft_target))
        norm = amp_pred * amp_target + 1e-8
        correlation = dot / norm
        loss_corr = 1.0 - torch.mean(correlation)

        return loss_amp + 0.5 * loss_corr


class CSILoss(nn.Module):
    """
    [软 CSI 评分损失]
    直接优化竞赛指标。
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        thresholds_mm = [0.1, 1.0, 2.0, 5.0, 8.0]
        weights_intensity = [1.0, 1.0, 2.0, 5.0, 10.0] 
        
        self.register_buffer('thresholds', torch.tensor([mm_to_lognorm(t) for t in thresholds_mm]))
        self.register_buffer('intensity_weights', torch.tensor(weights_intensity))
        self.smooth = smooth
        self.temperature = 50.0

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if pred.dim() == 5: pred = pred.squeeze(2)
        if target.dim() == 5: target = target.squeeze(2)
        if mask is not None and mask.dim() == 5: mask = mask.squeeze(2)

        total_loss = 0.0
        total_w = 0.0

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
            total_loss += (1.0 - csi).mean() * w
            total_w += w
            
        return total_loss / (total_w + 1e-8)


class HybridLoss(nn.Module):
    """
    [混合损失函数 - 终极版]
    全程启用所有 Loss，仅保留 Trainer 层面的序列长度课程学习。
    """
    def __init__(self, 
                 weight_bal_mse: float = 1.0, 
                 weight_dice: float = 1.0,    
                 weight_csi: float = 1.0, 
                 weight_gdl: float = 1.0,      
                 weight_facl: float = 0.0,    
                 weight_msssim: float = 0.0,   
                 weight_lpips: float = 0.0,
                 use_curriculum: bool = False, 
                 use_temporal_weight: bool = True,
                 **kwargs):
        super().__init__()
        
        self.weights = {
            'bal_mse': weight_bal_mse, 
            'dice': weight_dice,
            'csi': weight_csi, 
            'gdl': weight_gdl, 
            'facl': weight_facl,
            'msssim': weight_msssim,
            'lpips': weight_lpips
        }
        
        self.use_temporal_weight = use_temporal_weight
        
        # 初始化子 Loss
        self.loss_bal_mse = BalancedMSELoss(use_l1=True)
        self.loss_dice = DiceLoss() 
        self.loss_csi = CSILoss()
        self.loss_gdl = GradientDifferenceLoss()
        
        # 可选 Loss (现在 SpectralLoss 已定义，可以直接使用)
        self.loss_facl = None
        if weight_facl > 0: 
            self.loss_facl = SpectralLoss()  # [Fixed] 直接实例化，无需 import
            
        self.loss_msssim = None
        if weight_msssim > 0:
             self.loss_msssim = MultiScaleStructuralSimilarityIndexMeasure(
                data_range=1.0, kernel_size=11, reduction='elementwise_mean'
            )
        
        self.loss_lpips = None
        if weight_lpips > 0:
            self.loss_lpips = lpips.LPIPS(net='vgg').eval()
            self.loss_lpips.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                current_epoch: int = 100, total_epochs: int = 100) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        # 维度兼容性处理
        if pred.ndim == 5: pred = pred.squeeze(2)
        if target.ndim == 5: target = target.squeeze(2)
        if mask is not None and mask.ndim == 5: mask = mask.squeeze(2)
        
        # 1. 时间权重计算
        extra_weights = None
        if self.use_temporal_weight:
            T = pred.shape[1]
            t_w = torch.linspace(1.0, 2.0, steps=T, device=pred.device).view(1, T, 1, 1)
            extra_weights = t_w

        # 2. 计算基础 Loss
        l_mse = self.loss_bal_mse(pred, target, mask, extra_weights)
        l_dice = self.loss_dice(pred, target, mask)
        l_csi = self.loss_csi(pred, target, mask)
        l_gdl = self.loss_gdl(pred, target, mask)
        
        # 3. 计算可选 Loss
        l_facl = torch.tensor(0.0, device=pred.device)
        l_msssim = torch.tensor(0.0, device=pred.device)
        l_lpips = torch.tensor(0.0, device=pred.device)

        if self.loss_facl is not None:
            l_facl = self.loss_facl(pred, target, mask)
            
        if self.loss_msssim is not None:
             B, T, H, W = pred.shape
             l_msssim = 1.0 - self.loss_msssim(pred.view(-1, 1, H, W), target.view(-1, 1, H, W))

        if self.loss_lpips is not None:
             p_rgb = pred[:, -1:].repeat(1, 3, 1, 1) * 2 - 1 
             t_rgb = target[:, -1:].repeat(1, 3, 1, 1) * 2 - 1
             l_lpips = self.loss_lpips(p_rgb, t_rgb).mean()

        # 4. 加权求和
        total = (self.weights['bal_mse'] * l_mse + 
                 self.weights['dice'] * l_dice + 
                 self.weights['csi'] * l_csi + 
                 self.weights['gdl'] * l_gdl + 
                 self.weights['facl'] * l_facl + 
                 self.weights['msssim'] * l_msssim + 
                 self.weights['lpips'] * l_lpips)
        
        loss_dict = {
            'total': total, 
            'bal_mse': l_mse, 
            'dice': l_dice,
            'csi': l_csi, 
            'gdl': l_gdl,
            'facl': l_facl,
            'msssim': l_msssim
        }
        
        return total, loss_dict