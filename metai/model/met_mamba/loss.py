# metai/model/met_mamba/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
from typing import Optional, List, Dict, Tuple

from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import lpips

# ==============================================================================
# 常量与工具
# ==============================================================================

# 降水物理上限 (30mm/6min)
MM_MAX = 30.0  
LOG_NORM_FACTOR = math.log(MM_MAX + 1)

def mm_to_lognorm(mm_val: float) -> float:
    """将物理降水值 (mm) 转换为对数归一化值 (0-1)。"""
    return math.log(mm_val + 1) / LOG_NORM_FACTOR

def lognorm_to_mm(norm_val: torch.Tensor) -> torch.Tensor:
    """将对数归一化值 (0-1) 反转回物理值 (mm)。"""
    return torch.exp(norm_val * LOG_NORM_FACTOR) - 1

# ==============================================================================
# 基础 Loss 组件
# ==============================================================================

class SpectralLoss(nn.Module):
    """
    [频域] 傅里叶幅度与相关性损失 (FACL)
    说明: 通过 FFT 约束频域幅度和相位，强制模型恢复高频纹理信息。
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


class GradientDifferenceLoss(nn.Module):
    """
    [物理] 梯度差分损失 (GDL) - 强度加权版
    说明: 约束空间梯度 (边缘锐度) 和时间梯度 (运动连续性)。
    [修改] 引入强度加权：对强降水区域的梯度误差给予更高惩罚，避免强回波模糊。
    """
    def __init__(self, alpha=1.0, temporal_weight=1.0, weight_scale=5.0):
        super().__init__()
        self.alpha = alpha
        self.temporal_weight = temporal_weight
        self.weight_scale = weight_scale # 强度加权系数 (控制对强降水的关注程度)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # --- 1. 空间梯度 (Spatial Gradients) ---
        # 计算相邻像素差分 |I(x+1) - I(x)|
        p_dy = torch.abs(pred[..., 1:, :] - pred[..., :-1, :])
        p_dx = torch.abs(pred[..., :, 1:] - pred[..., :, :-1])
        t_dy = torch.abs(target[..., 1:, :] - target[..., :-1, :])
        t_dx = torch.abs(target[..., :, 1:] - target[..., :, :-1])

        diff_dy = torch.abs(p_dy - t_dy) ** self.alpha
        diff_dx = torch.abs(p_dx - t_dx) ** self.alpha

        # [新增] 动态加权：基于 Target 强度对梯度误差加权
        # 逻辑：强降水边缘的模糊(梯度丢失)需要被更重地惩罚
        w_dy = 1.0 + self.weight_scale * target[..., 1:, :]
        w_dx = 1.0 + self.weight_scale * target[..., :, 1:]
        
        diff_dy = diff_dy * w_dy
        diff_dx = diff_dx * w_dx

        # 应用 Mask (空间)
        if mask is not None:
            # 梯度图尺寸比原图少1，Mask 需对应切片
            m_dy = mask[..., 1:, :] * mask[..., :-1, :]
            m_dx = mask[..., :, 1:] * mask[..., :, :-1]
            
            loss_spatial = (diff_dy * m_dy).sum() / (m_dy.sum() + 1e-8) + \
                           (diff_dx * m_dx).sum() / (m_dx.sum() + 1e-8)
        else:
            loss_spatial = diff_dy.mean() + diff_dx.mean()

        # --- 2. 时间梯度 (Temporal Gradients) ---
        loss_temp = torch.tensor(0.0, device=pred.device)
        if pred.shape[1] > 1:
            p_dt = torch.abs(pred[:, 1:] - pred[:, :-1])
            t_dt = torch.abs(target[:, 1:] - target[:, :-1])
            diff_dt = torch.abs(p_dt - t_dt) ** self.alpha
            
            # [新增] 时间梯度加权：重点关注强回波的移动/生消
            w_dt = 1.0 + self.weight_scale * target[:, 1:]
            diff_dt = diff_dt * w_dt
            
            # 应用 Mask (时间)
            if mask is not None:
                m_dt = mask[:, 1:] * mask[:, :-1]
                loss_temp = (diff_dt * m_dt).sum() / (m_dt.sum() + 1e-8)
            else:
                loss_temp = diff_dt.mean()

        return loss_spatial + self.temporal_weight * loss_temp


class BalancedMSELoss(nn.Module):
    """
    [回归] 平衡均方误差
    说明: 根据降水强度分段加权，解决长尾分布问题。
    阈值: 0.1(5x), 2.0(20x), 5.0(50x)
    """
    def __init__(self, use_l1=True):
        super().__init__()
        self.use_l1 = use_l1
        
        # 分段阈值 (对应物理值)
        self.thresh_light = 0.1
        self.thresh_mod   = 2.0
        self.thresh_heavy = 5.0
        
        self.log_thresh_light = mm_to_lognorm(self.thresh_light)
        self.log_thresh_mod = mm_to_lognorm(self.thresh_mod)
        self.log_thresh_heavy = mm_to_lognorm(self.thresh_heavy)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, extra_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_l1:
            diff = torch.abs(pred - target)
        else:
            diff = (pred - target) ** 2

        # 动态权重生成
        weights = torch.ones_like(target)
        weights[target >= self.log_thresh_light] = 1.0
        weights[target >= self.log_thresh_mod] = 2.0
        weights[target >= self.log_thresh_heavy] = 5.0  

        loss_map = diff * weights
        
        # 叠加额外权重 (如时间权重)
        if extra_weights is not None:
            loss_map = loss_map * extra_weights

        # 应用 Mask 归一化
        if mask is not None:
            return (loss_map * mask).sum() / (mask.sum() + 1e-8)
        
        return loss_map.mean()


class SoftCSILoss(nn.Module):
    """
    [指标] 软 CSI 评分损失
    说明: 使用 Sigmoid 平滑近似不可微的 CSI 指标。
    阈值: [0.1, 1.0, 2.0, 5.0, 8.0]
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        
        thresholds_mm = [0.1, 1.0, 2.0, 5.0, 8.0]
        weights_intensity = [1.0, 1.0, 2.0, 5.0, 10.0]
        
        self.register_buffer('thresholds', torch.tensor([mm_to_lognorm(t) for t in thresholds_mm]))
        self.register_buffer('intensity_weights', torch.tensor(weights_intensity))
        self.smooth = smooth
        self.temperature = 20.0 

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if pred.dim() == 5: pred = pred.squeeze(2)
        if target.dim() == 5: target = target.squeeze(2)
        if mask is not None and mask.dim() == 5: mask = mask.squeeze(2)

        total_loss = 0.0
        total_w = 0.0

        for i, thresh in enumerate(self.thresholds):
            w = self.intensity_weights[i]
            
            # Sigmoid 近似
            pred_score = torch.sigmoid((pred - thresh) * self.temperature)
            target_score = (target > thresh).float()
            
            # 应用 Mask
            if mask is not None:
                pred_score = pred_score * mask
                target_score = target_score * mask
            
            # 计算软交并集
            intersection = (pred_score * target_score).sum(dim=(-2, -1))
            union = pred_score.sum(dim=(-2, -1)) + target_score.sum(dim=(-2, -1)) - intersection
            
            csi = (intersection + self.smooth) / (union + self.smooth)
            total_loss += (1.0 - csi).mean() * w
            total_w += w
            
        return total_loss / (total_w + 1e-8)

# ==============================================================================
# 混合损失
# ==============================================================================

class HybridLoss(nn.Module):
    """
    混合损失函数
    集成: BalancedMSE + SoftCSI + GDL + Spectral + SSIM + LPIPS
    特性: 支持课程学习 (纹理权重渐进) 和时间维度加权
    """
    def __init__(self, 
                 weight_bal_mse: float = 1.0, 
                 weight_facl: float = 0.01,    # [优化] 默认值降低，防止初始冲击
                 weight_gdl: float = 0.1,      # [优化] 默认值降低
                 weight_csi: float = 0.5, 
                 weight_msssim: float = 0.1,   # [优化] 默认值降低
                 weight_lpips: float = 0.0,    # 默认不开启 LPIPS (计算量大且易不稳定)
                 use_curriculum: bool = True,
                 use_temporal_weight: bool = True,
                 **kwargs):
        super().__init__()
        
        self.weights = {
            'bal_mse': weight_bal_mse, 
            'facl': weight_facl, 
            'gdl': weight_gdl, 
            'csi': weight_csi, 
            'msssim': weight_msssim,
            'lpips': weight_lpips
        }
        
        self.use_curriculum = use_curriculum
        self.use_temporal_weight = use_temporal_weight
        
        # 初始化子模块
        self.loss_bal_mse = BalancedMSELoss(use_l1=True)
        self.loss_facl = SpectralLoss()
        # [修改] 使用强度加权版的 GDL
        self.loss_gdl = GradientDifferenceLoss(temporal_weight=2.0, weight_scale=5.0)
        self.loss_csi = SoftCSILoss()
        
        self.loss_msssim = None
        if weight_msssim > 0:
             self.loss_msssim = MultiScaleStructuralSimilarityIndexMeasure(
                data_range=1.0, kernel_size=11, reduction='elementwise_mean'
            )
            
        self.loss_lpips = None
        if weight_lpips > 0.0:
            print("Initializing LPIPS (VGG)...")
            self.loss_lpips = lpips.LPIPS(net='vgg').eval()
            self.loss_lpips.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                current_epoch: int = 100, total_epochs: int = 100) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        # 维度规整 [B, T, H, W]
        if pred.ndim == 5: pred = pred.squeeze(2)
        if target.ndim == 5: target = target.squeeze(2)
        if mask is not None and mask.ndim == 5: mask = mask.squeeze(2)
        
        # 1. 课程学习权重计算
        # [关键修改] "错峰出行"策略：推迟纹理损失介入时间
        # 0% ~ 30%: 仅 MSE + CSI (适应序列长度增长)
        # 30% ~ 80%: 线性增加纹理权重
        texture_weight_factor = 1.0
        if self.use_curriculum and total_epochs > 0:
            safe_current = min(current_epoch, total_epochs)
            progress = safe_current / float(total_epochs)
            
            if progress < 0.3:
                texture_weight_factor = 0.0 
            elif progress < 0.8:
                texture_weight_factor = (progress - 0.3) / 0.5
            else:
                texture_weight_factor = 1.0

        # 2. 时间维度加权
        extra_weights = None
        if self.use_temporal_weight:
            T = pred.shape[1]
            t_w = torch.linspace(1.0, 1.5, steps=T, device=pred.device).view(1, T, 1, 1)
            extra_weights = t_w

        # 3. 计算基础损失 (始终存在)
        l_mse = self.loss_bal_mse(pred, target, mask, extra_weights=extra_weights)
        l_csi = self.loss_csi(pred, target, mask)
        
        l_facl = torch.tensor(0.0, device=pred.device)
        l_gdl = torch.tensor(0.0, device=pred.device)
        l_msssim = torch.tensor(0.0, device=pred.device)
        l_lpips = torch.tensor(0.0, device=pred.device)

        # 4. 计算纹理/感知损失 (根据课程进度介入)
        if texture_weight_factor > 0:
            l_facl = self.loss_facl(pred, target, mask) 
            l_gdl = self.loss_gdl(pred, target, mask)
            
            # SSIM 和 LPIPS 使用遮罩后的输入
            if mask is not None:
                pred_masked = pred * mask
                target_masked = target * mask
            else:
                pred_masked = pred
                target_masked = target

            if self.loss_msssim is not None:
                # Reshape for SSIM [B*T, 1, H, W]
                B, T, H, W = pred.shape
                p_ssim = pred_masked.reshape(-1, 1, H, W)
                t_ssim = target_masked.reshape(-1, 1, H, W)
                l_msssim = 1.0 - self.loss_msssim(p_ssim, t_ssim)
            
            if self.loss_lpips is not None:
                # LPIPS 仅计算最后一帧 (通常是最难的)
                p_rgb = pred_masked[:, -1:].repeat(1, 3, 1, 1) * 2 - 1 
                t_rgb = target_masked[:, -1:].repeat(1, 3, 1, 1) * 2 - 1
                l_lpips = self.loss_lpips(p_rgb, t_rgb).mean()

        # 5. 加权求和
        w_facl = self.weights['facl'] * texture_weight_factor
        w_gdl = self.weights['gdl'] * texture_weight_factor
        w_msssim = self.weights['msssim'] * texture_weight_factor
        w_lpips = self.weights['lpips'] * texture_weight_factor
        
        total = (self.weights['bal_mse'] * l_mse + 
                 self.weights['csi'] * l_csi + 
                 w_facl * l_facl + 
                 w_gdl * l_gdl + 
                 w_msssim * l_msssim + 
                 w_lpips * l_lpips)
        
        loss_dict = {
            'total': total, 'bal_mse': l_mse, 'csi': l_csi, 
            'facl': l_facl, 'gdl': l_gdl, 'msssim': l_msssim, 'lpips': l_lpips
        }
        
        return total, loss_dict