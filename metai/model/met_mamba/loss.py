# metai/model/met_mamba/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple, Union

# ==============================================================================
# 工具函数 & 物理常量
# ==============================================================================

# 物理参数：必须与 Dataset 的归一化逻辑严格一致
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
    引入“阶梯式加权”，针对强降水给予更高权重。
    """
    def __init__(self):
        super().__init__()
        # 定义关键业务阈值 (mm -> Log Space)
        self.thresh_light = mm_to_lognorm(0.1)  # 小雨
        self.thresh_mod   = mm_to_lognorm(2.0)  # 中雨
        self.thresh_heavy = mm_to_lognorm(5.0)  # 大/暴雨

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, extra_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 使用 L1 Loss 替代 MSE，对异常值更鲁棒
        diff = torch.abs(pred - target)
        
        # --- 阶梯式权重分配 ---
        weights = torch.ones_like(target)
        weights[target >= self.thresh_light] = 5.0   
        weights[target >= self.thresh_mod]   = 20.0  
        weights[target >= self.thresh_heavy] = 50.0  

        loss_map = diff * weights
        
        # 叠加额外权重
        if extra_weights is not None:
            loss_map = loss_map * extra_weights
            weights = weights * extra_weights

        # 应用有效区域 Mask
        if mask is not None:
            loss_map = loss_map * mask
            weights = weights * mask

        return loss_map.sum() / (weights.sum() + 1e-8)


class CSILoss(nn.Module):
    """
    [拓扑结构 Loss - 软化 CSI 指标]
    直接优化 Critical Success Index (CSI)。
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
        self.temperature = 50.0 

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        total_loss = 0.0
        total_w = 0.0
        
        for i, thresh in enumerate(self.thresholds):
            w = self.intensity_weights[i]
            # 软 CSI 计算
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


class SpectralLoss(nn.Module):
    """
    [频域纹理 Loss - FFT]
    约束幅度谱，保持高频纹理细节。
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

        # 幅度谱 Loss (L1)
        amp_pred = torch.abs(fft_pred)
        amp_target = torch.abs(fft_target)
        loss_amp = F.l1_loss(amp_pred, amp_target)
        
        return loss_amp


class PhysicsConstraintsLoss(nn.Module):
    """
    [物理约束 Loss]
    1. 非对称局部质量守恒
    2. 显式平流一致性 (Warp Loss)
    """
    def __init__(self, pool_size=4, under_penalty=2.0):
        super().__init__()
        self.pool_size = pool_size
        self.under_penalty = under_penalty 

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
            
        b, t, h, w = pred.shape
        p_reshaped = pred_masked.view(b * t, 1, h, w)
        t_reshaped = target_masked.view(b * t, 1, h, w)
        
        p_local = F.avg_pool2d(p_reshaped, kernel_size=self.pool_size, stride=self.pool_size)
        t_local = F.avg_pool2d(t_reshaped, kernel_size=self.pool_size, stride=self.pool_size)
        
        # 漏报惩罚
        diff = p_local - t_local
        weight_map = torch.where(diff < 0, self.under_penalty, 1.0)
        
        loss_cons = (torch.abs(diff) * weight_map).mean()
        
        # --- 2. 显式 Warp Loss ---
        loss_warp = torch.tensor(0.0, device=pred.device)
        
        if flow is not None and prev_frame is not None:
            src_imgs = torch.cat([prev_frame, target[:, :-1]], dim=1) 
            tgt_imgs = target
            
            B, T, C, H, W = tgt_imgs.shape
            
            # 上采样流场
            flow_up = F.interpolate(
                flow.view(B*T, 2, flow.shape[-2], flow.shape[-1]), 
                size=(H, W), mode='bilinear', align_corners=False
            ) 
            
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=flow.device),
                torch.linspace(-1, 1, W, device=flow.device),
                indexing='ij'
            )
            base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B*T, -1, -1, -1)
            
            # 叠加流场
            sampling_grid = base_grid + flow_up.permute(0, 2, 3, 1)
            
            # 执行 Warping
            src_flat = src_imgs.reshape(B*T, C, H, W)
            # 使用 reflection padding 缓解边界消失问题
            warped_flat = F.grid_sample(src_flat, sampling_grid, mode='bilinear', padding_mode='reflection', align_corners=False)
            warped = warped_flat.view(B, T, C, H, W)
            
            if mask is not None:
                loss_warp = F.l1_loss(warped * mask, tgt_imgs * mask)
            else:
                loss_warp = F.l1_loss(warped, tgt_imgs)
        
        return loss_cons, loss_warp

# ==============================================================================
# [新增] 对抗损失 (Adversarial Loss for GAN)
# ==============================================================================

class GANLoss(nn.Module):
    """
    [对抗损失] 支持 Hinge Loss (推荐用于 Spectral Norm) 和 Vanilla GAN Loss。
    """
    def __init__(self, mode='hinge'):
        super().__init__()
        self.mode = mode
        
    def get_disc_loss(self, logits_real, logits_fake):
        """
        计算判别器 Loss: 尽可能区分真假
        """
        if self.mode == 'hinge':
            loss_real = F.relu(1.0 - logits_real).mean()
            loss_fake = F.relu(1.0 + logits_fake).mean()
            return (loss_real + loss_fake) / 2.0
        elif self.mode == 'vanilla':
            loss_real = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
            loss_fake = F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
            return (loss_real + loss_fake) / 2.0
        else:
            raise ValueError(f"Unknown GAN mode: {self.mode}")

    def get_gen_loss(self, logits_fake):
        """
        计算生成器 Loss: 尽可能骗过判别器
        """
        if self.mode == 'hinge':
            return -logits_fake.mean()
        elif self.mode == 'vanilla':
            return F.binary_cross_entropy_with_logits(logits_fake, torch.ones_like(logits_fake))
        else:
            raise ValueError(f"Unknown GAN mode: {self.mode}")

# ==============================================================================
# 智能混合 Loss (Automatic Weighted Hybrid Loss)
# ==============================================================================

class HybridLoss(nn.Module):
    """
    [智能混合损失 - 内容重建部分]
    基于 Kendall's Multi-Task Learning 策略，增加正则化防止权重坍塌。
    
    注意：此类仅计算 Content Loss，不包含 Adversarial Loss。
    """
    def __init__(self, 
                 use_temporal_weight: bool = True,
                 **kwargs): 
        super().__init__()
        self.use_temporal_weight = use_temporal_weight
        
        # 1. 初始化各子 Loss
        self.loss_mse = BalancedMSELoss()
        self.loss_csi = CSILoss()
        self.loss_spectral = SpectralLoss()
        self.loss_physics = PhysicsConstraintsLoss(under_penalty=2.0)
        
        # 2. 定义可学习参数 (Learnable Weights)
        # s 代表 log(variance)。形状为 5: [MSE, CSI, Spectral, Conservation, Warp]
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

        # 1. 准备时间权重
        extra_weights = None
        if self.use_temporal_weight:
            T = pred.shape[1]
            extra_weights = torch.linspace(1.0, 2.0, steps=T, device=pred.device).view(1, T, 1, 1)

        # 2. 计算各项原始 Loss
        l_mse = self.loss_mse(pred, target, mask, extra_weights)
        l_csi = self.loss_csi(pred, target, mask)
        l_spec = self.loss_spectral(pred, target, mask)
        l_cons, l_warp = self.loss_physics(pred, target, flow, prev_frame, mask)
        
        losses = torch.stack([l_mse, l_csi, l_spec, l_cons, l_warp])
        
        # 3. 自动加权计算 (稳定性修正)
        s = self.params.clamp(min=-10.0, max=3.0).float()
        losses = losses.float()
        
        # precision: 相当于权重 (1 / sigma^2)
        precision = torch.exp(-s)
        
        # 贝叶斯多任务损失公式
        weighted_losses = 0.5 * (precision * losses + s)
        
        # 正则化项: 迫使 s 接近 0
        reg_loss = 0.1 * torch.sum(s ** 2)
        
        total_loss = weighted_losses.sum() + reg_loss
        
        # 4. 构建返回字典
        loss_dict = {
            'content_total': total_loss,
            # --- 原始 Loss ---
            'mse_raw': l_mse.detach(),
            'csi_raw': l_csi.detach(),
            'spec_raw': l_spec.detach(),
            'cons_raw': l_cons.detach(),
            'warp_raw': l_warp.detach(),
            # --- 实际权重 ---
            'w_mse': 0.5 * precision[0].detach(),
            'w_csi': 0.5 * precision[1].detach(),
            'w_spec': 0.5 * precision[2].detach(),
            'w_cons': 0.5 * precision[3].detach(),
            'w_warp': 0.5 * precision[4].detach(),
            # --- 参数监控 ---
            's_mse': s[0].detach(),
            'reg_s': reg_loss.detach()
        }
        
        return total_loss, loss_dict