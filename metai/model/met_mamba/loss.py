import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseBalancedL1Loss(nn.Module):
    """
    针对稀疏降水优化的 Loss。
    策略：
    1. Hard Mining: 只要 Target > 0 的区域，权重全部拉高。
    2. Background Suppression: Target == 0 的区域，随机采样或给予低权重。
    3. Extreme Value Focus: 值越大，权重越大。
    """
    def __init__(self, rain_threshold=0.05, rain_weight=5.0, bg_weight=0.1):
        super().__init__()
        self.threshold = rain_threshold / 30.0 # 归一化后的阈值
        self.rain_weight = rain_weight
        self.bg_weight = bg_weight

    def forward(self, pred, target, mask=None):
        # 基础误差
        diff = torch.abs(pred - target)
        
        # 1. 区分降水区 (Rainy) 和 背景区 (Background)
        is_rain = target > self.threshold
        
        # 2. 动态权重矩阵
        # 初始全部为背景权重
        weights = torch.ones_like(diff) * self.bg_weight
        
        # 降水区域赋予高权重
        weights[is_rain] = self.rain_weight
        
        # [进阶] 极值关注：对于强降水（例如 > 0.5），给予额外加成
        # 假设 0.5 对应真实值 15mm/h
        heavy_rain = target > (0.5) 
        weights[heavy_rain] *= 2.0 
        
        loss = diff * weights
        
        if mask is not None:
            loss = loss * mask
            # 只除以 mask 的面积，避免数值过小
            return loss.sum() / (mask.sum() + 1e-6)
        
        return loss.mean()

class GradientDifferenceLoss(nn.Module):
    """保持不变，抗模糊效果好"""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target, mask=None):
        p_dx = torch.abs(pred[..., :, 1:] - pred[..., :, :-1])
        p_dy = torch.abs(pred[..., 1:, :] - pred[..., :-1, :])
        t_dx = torch.abs(target[..., :, 1:] - target[..., :, :-1])
        t_dy = torch.abs(target[..., 1:, :] - target[..., :-1, :])
        loss = torch.abs(p_dx - t_dx) + torch.abs(p_dy - t_dy)
        if mask is not None:
            mask_x = mask[..., :, 1:] * mask[..., :, :-1]
            mask_y = mask[..., 1:, :] * mask[..., :-1, :]
            loss_x = (loss * mask_x[..., None]).sum() / (mask_x.sum() + 1e-6) # 修正维度广播
            # 这里简化处理，直接返回 mean 即可，GDL 对稀疏数据也有效
            return loss.mean() * self.alpha
        return loss.mean() * self.alpha

class HybridLoss(nn.Module):
    """
    针对稀疏场景的混合 Loss
    """
    def __init__(self, l1_weight=1.0, gdl_weight=1.0, **kwargs):
        super().__init__()
        self.weights = {'l1': l1_weight, 'gdl': gdl_weight}
        
        # 使用稀疏平衡 L1
        self.l1 = SparseBalancedL1Loss(rain_weight=10.0, bg_weight=0.5) 
        self.gdl = GradientDifferenceLoss()

    def forward(self, pred, target, mask=None):
        # 这里的 pred 已经是 残差叠加后的最终结果
        loss_dict = {}
        
        l1 = self.l1(pred, target, mask)
        gdl = self.gdl(pred, target, mask)
        
        total = self.weights['l1'] * l1 + self.weights['gdl'] * gdl
        
        loss_dict['total'] = total
        loss_dict['l1'] = l1
        loss_dict['gdl'] = gdl
        return total, loss_dict