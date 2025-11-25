import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

try:
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False

class WeightedScoreSoftCSILoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.MM_MAX = 30.0 
        thresholds_raw = [0.1, 1.0, 2.0, 5.0, 8.0]
        weights_raw    = [0.1, 0.1, 0.2, 0.2, 0.3] 
        self.register_buffer('thresholds', torch.tensor(thresholds_raw) / self.MM_MAX)
        self.register_buffer('intensity_weights', torch.tensor(weights_raw))
        time_weights_raw = [
            0.0075, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
            0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.0075, 0.005 
        ]
        self.register_buffer('time_weights', torch.tensor(time_weights_raw).view(1, -1, 1, 1))
        self.smooth = smooth

    def forward(self, pred, target, mask=None):
        T = pred.shape[1]
        current_time_weights = self.time_weights[:, :T, :, :]
        current_time_weights = current_time_weights / current_time_weights.mean()
        
        if mask is not None:
            if mask.dim() == 4 and mask.shape[1] == 1 and pred.shape[1] > 1:
                mask = mask.expand(-1, pred.shape[1], -1, -1)
            elif mask.dim() == 5:
                mask = mask.squeeze(2)

        total_weighted_loss = 0.0
        total_weight_sum = 0.0

        for i, t in enumerate(self.thresholds):
            w = self.intensity_weights[i]
            pred_score = torch.sigmoid((pred - t) * 50)
            target_score = (target > t).float()
            
            if mask is not None:
                pred_score = pred_score * mask
                target_score = target_score * mask
                
            intersection = (pred_score * target_score).sum(dim=(-2, -1))
            total_pred = pred_score.sum(dim=(-2, -1))
            total_target = target_score.sum(dim=(-2, -1))
            union = total_pred + total_target - intersection
            
            csi = (intersection + self.smooth) / (union + self.smooth)
            loss_map = 1.0 - csi
            
            weighted_loss_t = (loss_map * current_time_weights.squeeze(-1).squeeze(-1)).mean()
            total_weighted_loss += weighted_loss_t * w
            total_weight_sum += w

        return total_weighted_loss / total_weight_sum

class LogSpectralDistanceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target, mask=None):
        pred_fp32 = pred.float()
        target_fp32 = target.float()
        pred_fft = torch.fft.rfft2(pred_fp32, dim=(-2, -1), norm='ortho')
        target_fft = torch.fft.rfft2(target_fp32, dim=(-2, -1), norm='ortho')
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        loss = F.l1_loss(torch.log(pred_mag + self.epsilon), torch.log(target_mag + self.epsilon))
        return loss

class WeightedEvolutionLoss(nn.Module):
    def __init__(self, weight_scale=5.0):
        super().__init__()
        self.weight_scale = weight_scale

    def forward(self, pred, target, mask=None):
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        diff_error = torch.abs(pred_diff - target_diff)
        weight_map = 1.0 + self.weight_scale * target[:, 1:]
        
        if mask is not None:
            if mask.dim() == 5: mask = mask.squeeze(2)
            mask_t_plus_1 = mask[:, 1:] 
            diff_error = diff_error * mask_t_plus_1 
            weight_map = weight_map * mask_t_plus_1 
            count_valid = mask_t_plus_1.sum()
            if count_valid > 0:
                weighted_loss = (diff_error * weight_map).sum() / count_valid
            else:
                weighted_loss = 0.0 
        else:
            weighted_loss = (diff_error * weight_map).mean()
        return weighted_loss

class HybridLoss(nn.Module):
    def __init__(self, l1_weight=1.0, ssim_weight=0.5, csi_weight=1.0, spectral_weight=0.1, evo_weight=0.5):
        super().__init__()
        self.weights = {
            'l1': l1_weight, 'ssim': ssim_weight, 'csi': csi_weight, 'spec': spectral_weight, 'evo': evo_weight
        }
        self.l1 = nn.L1Loss(reduction='none')
        if TORCHMETRICS_AVAILABLE and ssim_weight > 0:
            self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, reduction='none')
        else:
            self.ms_ssim = None
        self.soft_csi = WeightedScoreSoftCSILoss()
        self.spectral = LogSpectralDistanceLoss()
        self.evolution = WeightedEvolutionLoss()

    def forward(self, logits, target, mask=None):
        if logits.dim() == 5: logits = logits.squeeze(2)
        if target.dim() == 5: target = target.squeeze(2)
        if mask is not None and mask.dim() == 5: mask = mask.squeeze(2)
        
        pred = torch.sigmoid(logits)
        loss_dict = {}
        total_loss = 0.0
        
        l1_loss_map = self.l1(pred, target)
        if mask is not None:
            masked_error = l1_loss_map * mask
            count_valid = mask.sum()
            l1_loss = masked_error.sum() / count_valid if count_valid > 0 else 0.0
        else:
            l1_loss = l1_loss_map.mean()
            
        total_loss += self.weights['l1'] * l1_loss
        loss_dict['l1'] = l1_loss.item() if isinstance(l1_loss, torch.Tensor) else l1_loss
        
        if self.weights['csi'] > 0:
            csi_loss = self.soft_csi(pred, target, mask)
            total_loss += self.weights['csi'] * csi_loss
            loss_dict['csi'] = csi_loss.item()
            
        if self.weights['spec'] > 0:
            spec_loss = self.spectral(pred, target, mask)
            total_loss += self.weights['spec'] * spec_loss
            loss_dict['spec'] = spec_loss.item()
            
        if self.weights['evo'] > 0 and pred.shape[1] > 1:
            evo_loss = self.evolution(pred, target, mask)
            total_loss += self.weights['evo'] * evo_loss
            loss_dict['evo'] = evo_loss.item()
            
        if self.ms_ssim is not None and self.weights['ssim'] > 0:
            pred_c = pred.view(-1, 1, pred.shape[-2], pred.shape[-1])
            target_c = target.view(-1, 1, target.shape[-2], target.shape[-1])
            if mask is not None:
                mask_c = mask.view(-1, 1, mask.shape[-2], mask.shape[-1])
                pred_c = pred_c * mask_c
                target_c = target_c * mask_c
            ssim_val = self.ms_ssim(pred_c, target_c).mean()
            ssim_loss = 1.0 - ssim_val
            total_loss += self.weights['ssim'] * ssim_loss
            loss_dict['ssim'] = ssim_loss.item()
        
        loss_dict['total'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        return total_loss, loss_dictimport torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

try:
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False

class WeightedScoreSoftCSILoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.MM_MAX = 30.0 
        thresholds_raw = [0.1, 1.0, 2.0, 5.0, 8.0]
        weights_raw    = [0.1, 0.1, 0.2, 0.2, 0.3] 
        self.register_buffer('thresholds', torch.tensor(thresholds_raw) / self.MM_MAX)
        self.register_buffer('intensity_weights', torch.tensor(weights_raw))
        time_weights_raw = [
            0.0075, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
            0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.0075, 0.005 
        ]
        self.register_buffer('time_weights', torch.tensor(time_weights_raw).view(1, -1, 1, 1))
        self.smooth = smooth

    def forward(self, pred, target, mask=None):
        T = pred.shape[1]
        current_time_weights = self.time_weights[:, :T, :, :]
        current_time_weights = current_time_weights / current_time_weights.mean()
        
        if mask is not None:
            if mask.dim() == 4 and mask.shape[1] == 1 and pred.shape[1] > 1:
                mask = mask.expand(-1, pred.shape[1], -1, -1)
            elif mask.dim() == 5:
                mask = mask.squeeze(2)

        total_weighted_loss = 0.0
        total_weight_sum = 0.0

        for i, t in enumerate(self.thresholds):
            w = self.intensity_weights[i]
            pred_score = torch.sigmoid((pred - t) * 50)
            target_score = (target > t).float()
            
            if mask is not None:
                pred_score = pred_score * mask
                target_score = target_score * mask
                
            intersection = (pred_score * target_score).sum(dim=(-2, -1))
            total_pred = pred_score.sum(dim=(-2, -1))
            total_target = target_score.sum(dim=(-2, -1))
            union = total_pred + total_target - intersection
            
            csi = (intersection + self.smooth) / (union + self.smooth)
            loss_map = 1.0 - csi
            
            weighted_loss_t = (loss_map * current_time_weights.squeeze(-1).squeeze(-1)).mean()
            total_weighted_loss += weighted_loss_t * w
            total_weight_sum += w

        return total_weighted_loss / total_weight_sum

class LogSpectralDistanceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target, mask=None):
        pred_fp32 = pred.float()
        target_fp32 = target.float()
        pred_fft = torch.fft.rfft2(pred_fp32, dim=(-2, -1), norm='ortho')
        target_fft = torch.fft.rfft2(target_fp32, dim=(-2, -1), norm='ortho')
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        loss = F.l1_loss(torch.log(pred_mag + self.epsilon), torch.log(target_mag + self.epsilon))
        return loss

class WeightedEvolutionLoss(nn.Module):
    def __init__(self, weight_scale=5.0):
        super().__init__()
        self.weight_scale = weight_scale

    def forward(self, pred, target, mask=None):
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        diff_error = torch.abs(pred_diff - target_diff)
        weight_map = 1.0 + self.weight_scale * target[:, 1:]
        
        if mask is not None:
            if mask.dim() == 5: mask = mask.squeeze(2)
            mask_t_plus_1 = mask[:, 1:] 
            diff_error = diff_error * mask_t_plus_1 
            weight_map = weight_map * mask_t_plus_1 
            count_valid = mask_t_plus_1.sum()
            if count_valid > 0:
                weighted_loss = (diff_error * weight_map).sum() / count_valid
            else:
                weighted_loss = 0.0 
        else:
            weighted_loss = (diff_error * weight_map).mean()
        return weighted_loss

class HybridLoss(nn.Module):
    def __init__(self, l1_weight=1.0, ssim_weight=0.5, csi_weight=1.0, spectral_weight=0.1, evo_weight=0.5):
        super().__init__()
        self.weights = {
            'l1': l1_weight, 'ssim': ssim_weight, 'csi': csi_weight, 'spec': spectral_weight, 'evo': evo_weight
        }
        self.l1 = nn.L1Loss(reduction='none')
        if TORCHMETRICS_AVAILABLE and ssim_weight > 0:
            self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, reduction='none')
        else:
            self.ms_ssim = None
        self.soft_csi = WeightedScoreSoftCSILoss()
        self.spectral = LogSpectralDistanceLoss()
        self.evolution = WeightedEvolutionLoss()

    def forward(self, logits, target, mask=None):
        if logits.dim() == 5: logits = logits.squeeze(2)
        if target.dim() == 5: target = target.squeeze(2)
        if mask is not None and mask.dim() == 5: mask = mask.squeeze(2)
        
        pred = torch.sigmoid(logits)
        loss_dict = {}
        total_loss = 0.0
        
        l1_loss_map = self.l1(pred, target)
        if mask is not None:
            masked_error = l1_loss_map * mask
            count_valid = mask.sum()
            l1_loss = masked_error.sum() / count_valid if count_valid > 0 else 0.0
        else:
            l1_loss = l1_loss_map.mean()
            
        total_loss += self.weights['l1'] * l1_loss
        loss_dict['l1'] = l1_loss.item() if isinstance(l1_loss, torch.Tensor) else l1_loss
        
        if self.weights['csi'] > 0:
            csi_loss = self.soft_csi(pred, target, mask)
            total_loss += self.weights['csi'] * csi_loss
            loss_dict['csi'] = csi_loss.item()
            
        if self.weights['spec'] > 0:
            spec_loss = self.spectral(pred, target, mask)
            total_loss += self.weights['spec'] * spec_loss
            loss_dict['spec'] = spec_loss.item()
            
        if self.weights['evo'] > 0 and pred.shape[1] > 1:
            evo_loss = self.evolution(pred, target, mask)
            total_loss += self.weights['evo'] * evo_loss
            loss_dict['evo'] = evo_loss.item()
            
        if self.ms_ssim is not None and self.weights['ssim'] > 0:
            pred_c = pred.view(-1, 1, pred.shape[-2], pred.shape[-1])
            target_c = target.view(-1, 1, target.shape[-2], target.shape[-1])
            if mask is not None:
                mask_c = mask.view(-1, 1, mask.shape[-2], mask.shape[-1])
                pred_c = pred_c * mask_c
                target_c = target_c * mask_c
            ssim_val = self.ms_ssim(pred_c, target_c).mean()
            ssim_loss = 1.0 - ssim_val
            total_loss += self.weights['ssim'] * ssim_loss
            loss_dict['ssim'] = ssim_loss.item()
        
        loss_dict['total'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        return total_loss, loss_dict