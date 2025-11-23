import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

# 尝试导入 torchmetrics，如果不存在则提供回退方案
try:
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("Warning: torchmetrics not found. MS-SSIM will be skipped.")


class WeightedScoreSoftCSILoss(nn.Module):
    """
    基于竞赛评分规则的 Soft-CSI 损失函数，支持强度加权、时效加权和 Masking。
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.MM_MAX = 30.0 
        
        # --- 1. 对齐强度分级及权重 ---
        thresholds_raw = [0.1, 1.0, 2.0, 5.0, 8.0]
        weights_raw    = [0.1, 0.1, 0.2, 0.2, 0.3] 
        self.register_buffer('thresholds', torch.tensor(thresholds_raw) / self.MM_MAX)
        self.register_buffer('intensity_weights', torch.tensor(weights_raw))
        
        # --- 2. 对齐时效及权重 ---
        time_weights_raw = [
            0.0075, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
            0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.0075, 0.005 
        ]
        self.register_buffer('time_weights', torch.tensor(time_weights_raw).view(1, -1, 1, 1))
        
        self.smooth = smooth

    def forward(self, pred, target, mask=None):
        """
        pred: [B, T, H, W], 范围 [0, 1]
        target: [B, T, H, W], 范围 [0, 1]
        mask: [B, T, H, W] 或 [B, 1, H, W]
        """
        T = pred.shape[1]
        current_time_weights = self.time_weights[:, :T, :, :]
        # 归一化时间权重
        current_time_weights = current_time_weights / current_time_weights.mean()
        
        # 统一 Mask 维度 (如果 mask 存在)
        if mask is not None:
            if mask.dim() == 4 and mask.shape[1] == 1 and pred.shape[1] > 1:
                mask = mask.expand(-1, pred.shape[1], -1, -1)
            elif mask.dim() == 5:
                mask = mask.squeeze(2)

        total_weighted_loss = 0.0
        total_weight_sum = 0.0

        for i, t in enumerate(self.thresholds):
            w = self.intensity_weights[i]
            
            # 1. 软二值化
            pred_score = torch.sigmoid((pred - t) * 50)
            target_score = (target > t).float()
            
            # 关键修改：应用 Mask
            if mask is not None:
                pred_score = pred_score * mask
                target_score = target_score * mask
                
            # 2. 计算 Intersection (TP) 和 Union (TP + FN + FP)
            intersection = (pred_score * target_score).sum(dim=(-2, -1))
            total_pred = pred_score.sum(dim=(-2, -1))
            total_target = target_score.sum(dim=(-2, -1))
            union = total_pred + total_target - intersection
            
            # 3. 计算每个时间步、每个样本的 CSI
            csi = (intersection + self.smooth) / (union + self.smooth)
            loss_map = 1.0 - csi
            
            # 4. 应用时间权重
            weighted_loss_t = (loss_map * current_time_weights.squeeze(-1).squeeze(-1)).mean()
            
            # 5. 应用强度权重
            total_weighted_loss += weighted_loss_t * w
            total_weight_sum += w

        return total_weighted_loss / total_weight_sum


class LogSpectralDistanceLoss(nn.Module):
    """
    频域损失。用于抗模糊，强制模型在频域保持高频分量。
    注意：此损失为全局约束，不建议应用局部 Masking。
    """
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target, mask=None): # 签名接受 mask，但内部不使用
        # FFT 变换需要 float32
        pred_fp32 = pred.float()
        target_fp32 = target.float()
        
        # FFT 变换
        pred_fft = torch.fft.rfft2(pred_fp32, dim=(-2, -1), norm='ortho')
        target_fft = torch.fft.rfft2(target_fp32, dim=(-2, -1), norm='ortho')
        
        # 幅度谱
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # 对数距离 (L1 loss on log-magnitude)
        loss = F.l1_loss(torch.log(pred_mag + self.epsilon), torch.log(target_mag + self.epsilon))
        
        return loss


class WeightedEvolutionLoss(nn.Module):
    """
    物理感知的加权演变损失。对强回波区域的变化赋予更高权重。
    支持 Masking。
    """
    def __init__(self, weight_scale=5.0):
        super().__init__()
        self.weight_scale = weight_scale

    def forward(self, pred, target, mask=None):
        # 计算时间差分 (dI/dt)
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        
        # 计算误差
        diff_error = torch.abs(pred_diff - target_diff)
        
        # 动态加权：如果该位置是强回波，则赋予更高权重
        weight_map = 1.0 + self.weight_scale * target[:, 1:]
        
        # 关键修改：应用 Mask
        if mask is not None:
            if mask.dim() == 5:
                mask = mask.squeeze(2)
            
            # 取 T-1 帧的 Mask (代表 t+1 时刻的有效性)
            mask_t_plus_1 = mask[:, 1:] 
            
            diff_error = diff_error * mask_t_plus_1 
            weight_map = weight_map * mask_t_plus_1 
            
            # 计算有效区域的平均值
            count_valid = mask_t_plus_1.sum()
            if count_valid > 0:
                weighted_loss = (diff_error * weight_map).sum() / count_valid
            else:
                weighted_loss = 0.0 
        
        else:
            weighted_loss = (diff_error * weight_map).mean()

        return weighted_loss


class HybridLoss(nn.Module):
    """
    Mamba 物理感知混合损失函数 (支持 Masking)
    """
    def __init__(self, 
                 l1_weight=1.0, 
                 ssim_weight=0.5, 
                 csi_weight=1.0, 
                 spectral_weight=0.1, 
                 evo_weight=0.5):
        super().__init__()
        self.weights = {
            'l1': l1_weight,
            'ssim': ssim_weight,
            'csi': csi_weight,
            'spec': spectral_weight,
            'evo': evo_weight
        }
        
        self.l1 = nn.L1Loss(reduction='none') # 必须使用 reduction='none' 来支持 Mask
        
        if TORCHMETRICS_AVAILABLE and ssim_weight > 0:
            self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, reduction='none')
        else:
            self.ms_ssim = None
            
        self.soft_csi = WeightedScoreSoftCSILoss()
        self.spectral = LogSpectralDistanceLoss()
        self.evolution = WeightedEvolutionLoss()

    def forward(self, logits, target, mask=None):
        """
        logits: [B, T, C, H, W] - 模型的原始输出
        target: [B, T, C, H, W] - 归一化后的真实值 [0, 1]
        mask: [B, T, C, H, W] 或 [B, T, H, W] - 0/1 张量
        """
        # 1. 预处理
        if logits.dim() == 5: logits = logits.squeeze(2)
        if target.dim() == 5: target = target.squeeze(2)
        if mask is not None and mask.dim() == 5: mask = mask.squeeze(2) # 简化 mask 维度
        
        pred = torch.sigmoid(logits)
        
        loss_dict = {}
        total_loss = 0.0
        
        # 2. L1 Loss (基础) - 关键修改：应用 Mask 和归一化
        l1_loss_map = self.l1(pred, target) # [B, T, H, W]
        
        if mask is not None:
            # 仅对有效区域求平均
            masked_error = l1_loss_map * mask
            count_valid = mask.sum()
            l1_loss = masked_error.sum() / count_valid if count_valid > 0 else 0.0
        else:
            l1_loss = l1_loss_map.mean()
            
        total_loss += self.weights['l1'] * l1_loss
        loss_dict['l1'] = l1_loss.item() if isinstance(l1_loss, torch.Tensor) else l1_loss
        
        # 3. Soft-CSI Loss (指标优化) - 传递 Mask
        if self.weights['csi'] > 0:
            csi_loss = self.soft_csi(pred, target, mask)
            total_loss += self.weights['csi'] * csi_loss
            loss_dict['csi'] = csi_loss.item()
            
        # 4. Spectral Loss (抗模糊) - 传递 Mask (但内部不使用)
        if self.weights['spec'] > 0:
            spec_loss = self.spectral(pred, target, mask)
            total_loss += self.weights['spec'] * spec_loss
            loss_dict['spec'] = spec_loss.item()
            
        # 5. Evolution Loss (物理约束) - 传递 Mask
        if self.weights['evo'] > 0 and pred.shape[1] > 1:
            evo_loss = self.evolution(pred, target, mask)
            total_loss += self.weights['evo'] * evo_loss
            loss_dict['evo'] = evo_loss.item()
            
        # 6. MS-SSIM Loss (结构一致性) - 应用 Mask
        if self.ms_ssim is not None and self.weights['ssim'] > 0:
            # SSIM 需要 Channel 维度 [B*T, 1, H, W]
            pred_c = pred.view(-1, 1, pred.shape[-2], pred.shape[-1])
            target_c = target.view(-1, 1, target.shape[-2], target.shape[-1])
            
            if mask is not None:
                # 在输入 SSIM 前将 Mask 区域置零
                mask_c = mask.view(-1, 1, mask.shape[-2], mask.shape[-1])
                pred_c = pred_c * mask_c
                target_c = target_c * mask_c
            
            # MS-SSIM 值的范围是 [0, 1]，1 - SSIM 即为损失
            ssim_val = self.ms_ssim(pred_c, target_c).mean() # 对所有 B*T 求平均
            ssim_loss = 1.0 - ssim_val
            total_loss += self.weights['ssim'] * ssim_loss
            loss_dict['ssim'] = ssim_loss.item()
        
        # 添加加权后的损失值到字典中
        loss_dict['total'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        for key in ['l1', 'csi', 'spec', 'evo', 'ssim']:
            if key in loss_dict and self.weights[key] > 0:
                loss_dict[f'{key}_weighted'] = self.weights[key] * loss_dict[key]
            
        return total_loss, loss_dict