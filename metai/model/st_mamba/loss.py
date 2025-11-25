import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试导入 SSIM
try:
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False

# =========================================================================
# 1. 核心回归: Spatial Focal MSE (修正版：支持物理阈值归一化)
# =========================================================================
class SpatialFocalMseLoss(nn.Module):
    """
    空间感知的 Focal MSE 损失函数 (Mask + 归一化修正版)
    """
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, eps: float = 1e-6,
                 kernel_size: int = 3, threshold: float = 0.1, reduction: str = 'mean'):
        """
        Args:
            threshold: 物理阈值 (mm)，默认为 0.1mm (区分有雨/无雨)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction
        
        # [关键修正] 引入 MM_MAX 进行阈值归一化
        self.MM_MAX = 30.0 
        # 将物理阈值 (0.1mm) 转换为归一化阈值 (0.1/30.0)
        self.register_buffer('threshold_norm', torch.tensor(threshold / self.MM_MAX))
        
        # 空间权重卷积核
        kernel_val = 1.0 / (kernel_size * kernel_size)
        self.register_buffer('kernel', torch.ones(1, 1, kernel_size, kernel_size) * kernel_val)
        self.pad = nn.ZeroPad2d(kernel_size // 2)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        pred, target: [B, T, H, W] (已归一化 0-1)
        mask: [B, T, H, W]
        """
        if pred.dim() == 5: pred = pred.squeeze(2)
        if target.dim() == 5: target = target.squeeze(2)
        if mask is not None and mask.dim() == 5: mask = mask.squeeze(2)

        # --- 1. 类别权重 (使用归一化后的阈值) ---
        # [修正] 使用 self.threshold_norm 替代原始 threshold
        is_rain = (target > self.threshold_norm).float()
        
        if mask is not None:
            mask = mask.float()
            n_rain = (is_rain * mask).sum()
            n_no_rain = (mask - (is_rain * mask)).sum()
        else:
            n_rain = is_rain.sum()
            n_no_rain = (1.0 - is_rain).sum()
            
        n_rain = torch.clamp(n_rain, min=self.eps)
        n_no_rain = torch.clamp(n_no_rain, min=self.eps)
        
        class_weight = torch.where(
            is_rain > 0.5,
            n_no_rain / n_rain, 
            n_rain / n_no_rain 
        )
        weight_map = self.alpha * class_weight

        # --- 2. Focal 权重 ---
        error = torch.abs(pred - target)
        error_norm = error / (error.max().detach() + self.eps)
        prob = torch.sigmoid(-error_norm * 5.0 + 2.0)
        focal_w = torch.clamp((1 - prob) ** self.gamma, min=0.1)
        
        weight_map = weight_map * focal_w

        # --- 3. 空间权重 ---
        B, T, H, W = target.shape
        target_flat = target.view(B*T, 1, H, W)
        
        if mask is not None:
            target_flat = target_flat * mask.view(B*T, 1, H, W)
            
        spatial_w = F.conv2d(self.pad(target_flat), self.kernel)
        spatial_w = spatial_w.view(B, T, H, W)
        weight_map = weight_map * (0.5 + spatial_w)

        # --- 4. Loss 计算 ---
        loss = weight_map * (pred - target) ** 2
        
        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() + self.eps)
            return loss.sum()
        
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()

# =========================================================================
# 2. 物理约束: Weighted Evolution Loss
# =========================================================================
class WeightedEvolutionLoss(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, pred, target, mask=None):
        # dI/dt
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        
        diff_err = torch.abs(pred_diff - target_diff)
        
        # [逻辑检查] target 是归一化的 (0~1)
        # 1.0 + 5.0 * target 意味着强回波区域(target->1)的权重是 6.0
        # 无雨区域权重是 1.0
        # 这个比例 (6:1) 在归一化空间是合理的，无需引入 MM_MAX
        w_map = 1.0 + 5.0 * target[:, 1:]
        
        loss_map = diff_err * w_map
        
        if mask is not None:
            valid_mask = mask[:, 1:] * mask[:, :-1]
            loss = (loss_map * valid_mask).sum() / (valid_mask.sum() + 1e-6)
        else:
            loss = loss_map.mean()
            
        return loss * self.scale

# =========================================================================
# 3. 指标优化: Soft CSI (已包含 MM_MAX 处理)
# =========================================================================
class WeightedScoreSoftCSILoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.MM_MAX = 30.0 
        thresholds = [0.1, 1.0, 2.0, 5.0, 8.0] 
        weights    = [1.0, 1.0, 2.0, 5.0, 10.0]
        
        # [逻辑确认] 这里正确除以了 MM_MAX
        self.register_buffer('t_vals', torch.tensor(thresholds) / self.MM_MAX)
        self.register_buffer('w_vals', torch.tensor(weights))
        self.smooth = smooth

    def forward(self, pred, target, mask=None):
        total_loss = 0.0
        total_w = 0.0

        for i, t in enumerate(self.t_vals):
            w = self.w_vals[i]
            pred_bin = torch.sigmoid((pred - t) * 50)
            target_bin = (target > t).float()
            
            if mask is not None:
                pred_bin = pred_bin * mask
                target_bin = target_bin * mask
                
            intersection = (pred_bin * target_bin).sum(dim=(-2, -1))
            union = pred_bin.sum(dim=(-2, -1)) + target_bin.sum(dim=(-2, -1)) - intersection
            
            csi = (intersection + self.smooth) / (union + self.smooth)
            total_loss += (1.0 - csi).mean() * w
            total_w += w

        return total_loss / total_w

# =========================================================================
# 4. 混合损失 (参数透传)
# =========================================================================
class HybridLoss(nn.Module):
    def __init__(self, 
                 focal_weight=10.0,
                 evo_weight=0.0,
                 ssim_weight=0.0,
                 csi_weight=0.0,
                 # 新增阈值参数
                 focal_threshold=0.1): 
        super().__init__()
        self.weights = {
            'focal': focal_weight,
            'evo': evo_weight,
            'ssim': ssim_weight,
            'csi': csi_weight
        }
        
        # 传递物理阈值 0.1mm
        self.focal_loss = SpatialFocalMseLoss(kernel_size=3, threshold=focal_threshold)
        self.evo_loss = WeightedEvolutionLoss()
        self.csi_loss = WeightedScoreSoftCSILoss()
        
        if TORCHMETRICS_AVAILABLE:
            self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, reduction='none')
        else:
            self.ms_ssim = None

    def forward(self, logits, target, mask=None):
        if logits.dim() == 5: logits = logits.squeeze(2)
        if target.dim() == 5: target = target.squeeze(2)
        if mask is not None and mask.dim() == 5: mask = mask.squeeze(2)
        
        pred = torch.sigmoid(logits)
        
        loss_dict = {}
        total_loss = 0.0
        
        # A. Focal Loss (Core)
        if self.weights['focal'] > 0:
            f_loss = self.focal_loss(pred, target, mask)
            total_loss += self.weights['focal'] * f_loss
            loss_dict['focal'] = f_loss.item()
            
        # B. Evolution Loss
        if self.weights['evo'] > 0 and pred.shape[1] > 1:
            e_loss = self.evo_loss(pred, target, mask)
            total_loss += self.weights['evo'] * e_loss
            loss_dict['evo'] = e_loss.item()
            
        # C. SSIM Loss
        if self.ms_ssim is not None and self.weights['ssim'] > 0:
            b, t, h, w = pred.shape
            p_flat = pred.view(b*t, 1, h, w)
            t_flat = target.view(b*t, 1, h, w)
            
            if mask is not None:
                m_flat = mask.view(b*t, 1, h, w)
                p_flat = p_flat * m_flat
                t_flat = t_flat * m_flat
                
            ssim_val = self.ms_ssim(p_flat, t_flat).mean()
            s_loss = 1.0 - ssim_val
            
            total_loss += self.weights['ssim'] * s_loss
            loss_dict['ssim'] = s_loss.item()
            
        # D. CSI Loss
        if self.weights['csi'] > 0:
            c_loss = self.csi_loss(pred, target, mask)
            total_loss += self.weights['csi'] * c_loss
            loss_dict['csi'] = c_loss.item()
            
        loss_dict['total'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        return total_loss, loss_dict