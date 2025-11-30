# metai/model/met_mamba/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# MS-SSIM 实现 (纯 PyTorch)
# ==========================

def gaussian_kernel(window_size, sigma):
    """生成一维高斯核"""
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """生成二维高斯窗口"""
    _1D_window = gaussian_kernel(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

class MSSSIMLoss(nn.Module):
    """
    多尺度结构相似性损失 (MS-SSIM Loss)。
    相比普通的 L1/L2 损失，能更好地保持图像的结构和纹理，减少模糊感。
    """
    def __init__(self, window_size=11, size_average=True, channel=1):
        super(MSSSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = create_window(window_size, self.channel)
        # MS-SSIM 权重参数 (根据原论文设置)
        self.weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

    def _ssim(self, img1, img2, window, metric='ssim'):
        padding = self.window_size // 2
        mu1 = F.conv2d(img1, window, padding=padding, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=padding, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if metric == 'cs':
            # 对比度敏感度 (Structure)
            return ((2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)).mean(dim=(1, 2, 3))
        
        return ssim_map.mean(dim=(1, 2, 3))

    def forward(self, pred, target, mask=None):
        """
        计算 MS-SSIM 损失。
        注意: MS-SSIM 通常对全图计算，暂不支持 mask (若有 mask 建议先对图像进行 masking 处理)。
        """
        # 展平 Batch 和 Time 维度: [B, T, C, H, W] -> [B*T, C, H, W]
        if pred.ndim == 5:
            b, t, c, h, w = pred.shape
            img1 = pred.view(-1, c, h, w)
            img2 = target.view(-1, c, h, w)
        else:
            img1 = pred
            img2 = target
        
        # 动态适配 Device
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
            self.weights = self.weights.to(img1.device)

        msssim = []
        mcs = []
        
        for i in range(len(self.weights)):
            # 最后一个尺度计算完整的 SSIM，其他尺度计算 CS (Contrast Sensitivity)
            if i == len(self.weights) - 1:
                ssim_val = self._ssim(img1, img2, self.window, metric='ssim')
                msssim.append(ssim_val)
            else:
                cs_val = self._ssim(img1, img2, self.window, metric='cs')
                mcs.append(cs_val)
                
                # 下采样 (Average Pooling)
                img1 = F.avg_pool2d(img1, (2, 2))
                img2 = F.avg_pool2d(img2, (2, 2))

        # 堆叠并计算加权 MS-SSIM
        msssim = torch.stack(msssim)
        mcs = torch.stack(mcs)
        
        # MS-SSIM = (Product of CS^weight for i=1..M-1) * (SSIM^weight for i=M)
        # Loss = 1 - MS-SSIM
        p1 = (mcs ** self.weights[:-1].view(-1, 1))
        p2 = (msssim ** self.weights[-1].view(-1, 1))
        output = torch.prod(p1, dim=0) * p2
        
        loss = 1.0 - output.mean()
        return loss

# ==========================
# 其他子损失模块
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
    集成 Focal, MS-SSIM, Corr, Dice 四大分量，全面优化预测质量。
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
        
        # 1. Focal Loss (强度准确性与模糊控制)
        self.loss_focal = FocalLoss(
            weights_val=intensity_weights,
            alpha=focal_alpha,
            gamma=focal_gamma,
            false_alarm_penalty=false_alarm_penalty
        )
        # 2. MS-SSIM Loss (结构与纹理保持)
        self.loss_msssim = MSSSIMLoss(channel=1) # 降水通常是单通道
        # 3. Corr Loss (空间分布一致性)
        self.loss_corr = CorrLoss(eps=corr_smooth_eps)
        # 4. Dice Loss (TS 评分优化)
        self.loss_dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        l_focal = self.loss_focal(pred, target, mask)
        # MS-SSIM 计算通常不接受 mask，因此对全图计算
        l_msssim = self.loss_msssim(pred, target) 
        l_corr = self.loss_corr(pred, target, mask)
        l_dice = self.loss_dice(pred, target, mask)
        
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