import torch
import torch.nn as nn
import math

# 全局物理常量: 30.0mm 对应模型输出 1.0
MM_MAX = 30.0

# Log 归一化系数: log(30.0 + 1) ≈ 3.434
# 用于将物理阈值转换到 Log 归一化空间
LOG_NORM_FACTOR = math.log(MM_MAX + 1)


def mm_to_lognorm(mm_val: float) -> float:
    """
    将物理降水值 (mm) 转换为对数归一化值 (0-1)。
    公式: log(x + 1) / log(30 + 1)
    """
    return math.log(mm_val + 1) / LOG_NORM_FACTOR


class MAELoss(nn.Module):
    """
    加权平均绝对误差损失 (Weighted MAE Loss)。

    该损失函数在 Log 归一化空间上计算，基于降水强度分级进行加权。
    权重策略与竞赛评分标准严格对齐，并包含虚警 (False Alarm) 抑制机制。

    Args:
        zero_rain_weight (float): 背景（无雨）区域的权重。默认为 0.05。
        false_alarm_penalty (float): 虚警惩罚系数。当实况无雨但预测有雨时，
            Loss 权重将乘以该系数。默认为 2.0。
        loss_scale (float): 整体 Loss 缩放因子，防止梯度数值过小。默认为 10.0。
    """
    def __init__(self, 
                 zero_rain_weight: float = 0.05,
                 false_alarm_penalty: float = 2.0,
                 loss_scale: float = 10.0):
        super().__init__()
        
        # 比赛评分标准物理阈值 (mm)
        thresholds_mm = [0.1, 1.0, 2.0, 5.0, 8.0]
        
        # 将物理阈值转换为 Log 归一化阈值以匹配模型输出
        log_thresholds = [mm_to_lognorm(t) for t in thresholds_mm]
        
        # 评分对齐策略权重
        # 对应区间: [背景, 0.1-1.0, 1.0-2.0, 2.0-5.0, 5.0-8.0, >8.0]
        # 权重值对应评分标准 (0.1, 0.1, 0.2, 0.25, 0.35) x 10
        weights_val = [zero_rain_weight, 1.0, 1.0, 2.0, 2.5, 3.5]
        
        # 注册 Buffer
        self.register_buffer('thresholds', torch.tensor(log_thresholds))
        self.register_buffer('weights', torch.tensor([w * loss_scale for w in weights_val]))
        
        # 虚警判定阈值 (0.1mm 对应的 Log 值)
        self.rain_start_threshold = mm_to_lognorm(0.1)
        self.false_alarm_penalty = false_alarm_penalty

    def forward(self, pred, target, mask=None):
        # 计算 Log 空间的 L1 差异
        diff = torch.abs(pred - target)
        
        # 动态查表获取权重 (target 已是 Log 分布)
        indices = torch.bucketize(target, self.thresholds)
        weight_map = self.weights[indices]
        
        # 虚警惩罚机制：Target < 0.1mm 但 Pred > 0.1mm
        if self.false_alarm_penalty > 0:
            is_zero_rain = target < self.rain_start_threshold
            is_false_alarm = is_zero_rain & (pred > self.rain_start_threshold)
            
            if is_false_alarm.any():
                weight_map = weight_map.clone()
                weight_map[is_false_alarm] *= self.false_alarm_penalty

        loss = diff * weight_map
        
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-6)
        
        return loss.mean()


class GDLoss(nn.Module):
    """
    梯度差分损失 (Gradient Difference Loss)。

    计算预测场与真实场在水平和垂直方向上的梯度差异。
    用于保持降水云团的边缘锐度和纹理结构，辅助提升相关性指标。

    Args:
        alpha (float): Loss 缩放系数。默认为 1.0。
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target, mask=None):
        # 计算 X/Y 方向梯度
        p_dx = torch.abs(pred[..., :, 1:] - pred[..., :, :-1])
        t_dx = torch.abs(target[..., :, 1:] - target[..., :, :-1])
        p_dy = torch.abs(pred[..., 1:, :] - pred[..., :-1, :])
        t_dy = torch.abs(target[..., 1:, :] - target[..., :-1, :])
        
        gdl_x = torch.abs(p_dx - t_dx)
        gdl_y = torch.abs(p_dy - t_dy)
        
        if mask is not None:
            # Mask 需要对应裁剪以匹配梯度图尺寸
            mask_x = mask[..., :, 1:] * mask[..., :, :-1]
            mask_y = mask[..., 1:, :] * mask[..., :-1, :]
            
            loss_x = (gdl_x * mask_x).sum() / (mask_x.sum() + 1e-6)
            loss_y = (gdl_y * mask_y).sum() / (mask_y.sum() + 1e-6)
            return (loss_x + loss_y) * self.alpha
            
        return (gdl_x.mean() + gdl_y.mean()) * self.alpha


class CorrLoss(nn.Module):
    """
    Pearson 相关性损失 (Correlation Loss)。

    直接优化相关系数 R (Loss = 1 - R)。
    支持忽略双零背景，以确保优化方向与竞赛评测标准（仅关注有雨区域相关性）一致。

    Args:
        eps (float): 防止除零的微小量。
        ignore_zeros (bool): 是否剔除预测和真值均为背景的区域。默认为 True。
    """
    def __init__(self, eps=1e-6, ignore_zeros=True):
        super().__init__()
        self.eps = eps
        self.ignore_zeros = ignore_zeros
        self.threshold = mm_to_lognorm(0.1)

    def forward(self, pred, target, mask=None):
        if mask is not None:
            mask = mask.bool()
        else:
            mask = torch.ones_like(pred, dtype=torch.bool)
            
        # 剔除双零背景：仅关注有雨区域的相关性
        if self.ignore_zeros:
            is_valid_rain = (pred >= self.threshold) | (target >= self.threshold)
            mask = mask & is_valid_rain

        p = pred[mask]
        t = target[mask]

        # 避免样本过少导致计算异常
        if p.numel() < 5:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        p_mean = p.mean()
        t_mean = t.mean()

        num = ((p - p_mean) * (t - t_mean)).sum()
        den = torch.sqrt(((p - p_mean)**2).sum() * ((t - t_mean)**2).sum())

        r = num / (den + self.eps)
        r = torch.clamp(r, -1.0, 1.0)

        return 1.0 - r


class TSLoss(nn.Module):
    """
    威胁分数损失 (Threat Score Loss / Soft Dice Loss)。

    通过 Sigmoid 软化阈值，计算多个关键阈值下的 Soft Dice Loss，
    直接优化模型的 TS 评分能力。

    Args:
        weights (List[float]): 各阈值的加权系数。
    """
    def __init__(self, weights=[0.1, 0.1, 0.2, 0.25, 0.35]):
        super().__init__()
        # 将物理阈值转换为 Log 归一化阈值
        self.thresholds = [mm_to_lognorm(x) for x in [0.1, 1.0, 2.0, 5.0, 8.0]]
        self.weights = weights
        self.smooth = 1e-5
        self.temperature = 50.0 # Sigmoid 温度系数，值越大越接近阶跃函数

    def forward(self, pred, target, mask=None):
        loss = 0.0
        total_w = sum(self.weights)
        
        for i, thresh in enumerate(self.thresholds):
            # Soft Thresholding: 使用 Sigmoid 近似 I(x > thresh)
            p_mask = torch.sigmoid((pred - thresh) * self.temperature)
            t_mask = (target >= thresh).float()
            
            if mask is not None:
                p_mask = p_mask * mask
                t_mask = t_mask * mask

            intersection = (p_mask * t_mask).sum()
            union = p_mask.sum() + t_mask.sum()
            
            # Dice Loss = 1 - Dice Coefficient
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            loss += (1.0 - dice) * self.weights[i]
            
        return loss / total_w


class HybridLoss(nn.Module):
    """
    混合损失函数 (Hybrid Loss)。

    组合 MAE、GDL、Corr、TS 四种损失以最大化竞赛评分。
    
    默认权重策略 (针对数值量级平衡优化):
    - mae:  1.0  (基准)
    - gdl:  10.0 (放大以匹配 MAE 量级，强化纹理)
    - corr: 0.5  (R 是全局乘数项，需重点优化但要防止震荡)
    - ts:   1.0  (直接优化 TS 得分)

    Args:
        l1_weight (float): MAE Loss 权重。
        gdl_weight (float): GDL Loss 权重。
        corr_weight (float): Correlation Loss 权重。
        dice_weight (float): TS/Dice Loss 权重。
    """
    def __init__(self, 
                 l1_weight=1.0, 
                 gdl_weight=10.0, 
                 corr_weight=0.5, 
                 dice_weight=1.0, 
                 **kwargs):
        super().__init__()
        self.weights = {
            'mae': l1_weight, 
            'gdl': gdl_weight,
            'corr': corr_weight,
            'ts': dice_weight
        }
        
        self.mae = MAELoss()
        self.gdl = GDLoss()
        self.corr = CorrLoss(ignore_zeros=True)
        self.ts = TSLoss()

    def forward(self, pred, target, mask=None):
        loss_dict = {}
        
        # 计算各分项
        mae = self.mae(pred, target, mask)
        gdl = self.gdl(pred, target, mask)
        corr = self.corr(pred, target, mask)
        ts = self.ts(pred, target, mask)
        
        # 加权求和
        total = (self.weights['mae'] * mae + 
                 self.weights['gdl'] * gdl + 
                 self.weights['corr'] * corr + 
                 self.weights['ts'] * ts)
        
        # 记录日志 (保留 l1 键名以兼容 trainer 的 logging 逻辑)
        loss_dict['total'] = total
        loss_dict['l1'] = mae 
        loss_dict['gdl'] = gdl
        loss_dict['corr'] = corr
        loss_dict['ts'] = ts
        
        return total, loss_dict