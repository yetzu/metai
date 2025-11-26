import torch
import torch.nn as nn
import torch.nn.functional as F
from metai.model.core.metrics import lat_weighted_mse_val, weighted_csi

class WeightedL1Loss(nn.Module):
    """
    [优化] 带权重的 L1 Loss，专注于结构保持和清晰度。
    相比 MSE，L1 更不容易产生模糊。
    """
    def __init__(self, threshold=0.1, heavy_weight=5.0):
        super().__init__()
        self.threshold = threshold
        # 将 mm 阈值转换为归一化数值 (假设 max=30.0)
        self.norm_threshold = threshold / 30.0
        self.heavy_weight = heavy_weight

    def forward(self, pred, target, mask=None):
        # 基础 L1
        loss = torch.abs(pred - target)
        
        # [策略] 对强降水区域给与更高权重
        # 只要 target 或 pred 任意一个超过阈值，就视为重点关注区域
        heavy_mask = (target > self.norm_threshold) | (pred > self.norm_threshold)
        
        # 创建权重矩阵：基础权重 1.0，强降水区域权重 heavy_weight
        weights = torch.ones_like(loss)
        weights[heavy_mask] = self.heavy_weight
        
        loss = loss * weights
        
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-6)
        else:
            return loss.mean()

class EvolutionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target, mask=None):
        # 简单的帧间差分损失，约束物理演变平滑性
        diff_pred = pred[:, 1:] - pred[:, :-1]
        diff_target = target[:, 1:] - target[:, :-1]
        loss = torch.abs(diff_pred - diff_target)
        
        if mask is not None:
            # mask 需要对齐时间维度
            mask_t = mask[:, 1:] & mask[:, :-1]
            loss = loss * mask_t
            return loss.sum() / (mask_t.sum() + 1e-6)
        return loss.mean()

class HybridLoss(nn.Module):
    def __init__(self, l1_weight=1.0, evo_weight=1.0, **kwargs):
        super().__init__()
        self.l1_loss = WeightedL1Loss(threshold=1.0, heavy_weight=2.0) # 适当加权
        self.evo_loss = EvolutionLoss()
        
        # 动态权重字典
        self.weights = {
            'l1': l1_weight,
            'evo': evo_weight
        }

    def forward(self, pred, target, mask=None):
        loss_dict = {}
        
        # 1. L1 Loss (主导)
        l1 = self.l1_loss(pred, target, mask)
        loss_dict['l1'] = l1
        
        # 2. Evolution Loss (物理约束)
        evo = self.evo_loss(pred, target, mask)
        loss_dict['evo'] = evo
        
        # 总 Loss
        total_loss = (
            self.weights['l1'] * l1 + 
            self.weights['evo'] * evo
        )
        
        loss_dict['total'] = total_loss
        return total_loss, loss_dict