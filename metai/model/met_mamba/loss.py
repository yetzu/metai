# metai/model/met_mamba/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# === 全局物理常量 ===
# 定义物理最大值 (30.0mm 对应模型输出 1.0)
# 这是连接归一化空间 [0,1] 与真实物理空间 [0, 30mm] 的锚点
MM_MAX = 30.0

class WeightedL1Loss(nn.Module):
    """
    [Weighted L1 Loss]
    基于降水强度分级加权的 L1 损失函数。
    
    核心机制：
    1. 物理对齐：内置 MM_MAX，自动将归一化输入映射回 mm 进行阈值判断。
    2. 比赛标准：严格按照比赛规定的 5 个量级区间 ([0.1-0.9], [1-1.9]...) 分配权重。
    3. 虚警抑制：对“背景区报有雨”的像素施加额外惩罚。
    """
    def __init__(self, 
                 zero_rain_weight: float = 0.05,   # 背景(无雨)区域的权重，保持较低以聚焦降水
                 false_alarm_penalty: float = 2.0, # 虚警惩罚倍率 (False Alarm Penalty)
                 loss_scale: float = 10.0):        # Loss 整体缩放因子，防止梯度数值过小
        super().__init__()
        
        # === 1. 定义比赛评分标准 (单位: mm) ===
        # 阈值边界 (左闭右开):
        # Index 0: (-inf, 0.1) -> 背景
        # Index 1: [0.1, 1.0)
        # Index 2: [1.0, 2.0)
        # Index 3: [2.0, 5.0)
        # Index 4: [5.0, 8.0)
        # Index 5: [8.0, +inf)
        thresholds_mm = [0.1, 1.0, 2.0, 5.0, 8.0]
        
        # 对应区间权重 (参考比赛表格):
        # [背景, 0.1-0.9, 1.0-1.9, 2.0-4.9, 5.0-7.9, >=8.0]
        # weights_val = [zero_rain_weight, 0.1, 0.1, 0.2, 0.25, 0.35]
        weights_val = [zero_rain_weight, 1.0, 2.0, 5.0, 10.0, 20.0]
        
        # === 2. 内部预处理 (使用全局 MM_MAX) ===
        # 将物理阈值转换为模型输出的归一化空间 [0, 1]
        normalized_thresholds = torch.tensor(thresholds_mm) / MM_MAX
        
        # 预先缩放权重，确保 Loss 在合理数值范围 (如 1.0~10.0)
        scaled_weights = torch.tensor([w * loss_scale for w in weights_val])
        
        # 注册为 Buffer (自动随模型 save/load 和 device 移动，但不作为参数更新)
        self.register_buffer('thresholds', normalized_thresholds)
        self.register_buffer('weights', scaled_weights)
        
        # 虚警判定阈值 (0.1mm 对应的归一化值)
        self.rain_start_threshold = 0.1 / MM_MAX
        self.false_alarm_penalty = false_alarm_penalty

    def forward(self, pred, target, mask=None):
        # 1. 基础绝对误差
        diff = torch.abs(pred - target)
        
        # 2. 查表获取动态权重 (基于 Target 真实强度)
        # torch.bucketize: 返回 target 落在哪个区间索引 (0 ~ 5)
        indices = torch.bucketize(target, self.thresholds)
        weight_map = self.weights[indices]
        
        # 3. 虚警惩罚 (False Alarm Penalty)
        # 逻辑：真实值是背景(<0.1mm)，但预测值报了雨(>0.1mm)
        if self.false_alarm_penalty > 0:
            is_zero_rain = target < self.rain_start_threshold
            is_false_alarm = is_zero_rain & (pred > self.rain_start_threshold)
            
            # 对虚警区域的权重进行倍增 (clone 避免 inplace 操作风险)
            weight_map = weight_map.clone()
            weight_map[is_false_alarm] *= self.false_alarm_penalty

        # 4. 计算最终加权 Loss
        loss = diff * weight_map
        
        if mask is not None:
            loss = loss * mask
            # 归一化：只除以 mask 的有效面积，避免数值过小
            return loss.sum() / (mask.sum() + 1e-6)
        
        return loss.mean()

class GradientDifferenceLoss(nn.Module):
    """
    梯度差分损失 (Gradient Difference Loss)
    分别计算 H(dy) 和 W(dx) 方向的梯度差异，用于保持降水云团的边缘锐度。
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target, mask=None):
        # pred/target shape: (B, T, H, W) or (B, T, C, H, W)
        # 这里的切片操作兼容上述两种维度
        
        # 计算 X 方向梯度 (沿 W 维度差分) -> [..., :, 1:] - [..., :, :-1]
        p_dx = torch.abs(pred[..., :, 1:] - pred[..., :, :-1])
        t_dx = torch.abs(target[..., :, 1:] - target[..., :, :-1])
        
        # 计算 Y 方向梯度 (沿 H 维度差分) -> [..., 1:, :] - [..., :-1, :]
        p_dy = torch.abs(pred[..., 1:, :] - pred[..., :-1, :])
        t_dy = torch.abs(target[..., 1:, :] - target[..., :-1, :])
        
        # 计算梯度误差
        gdl_x = torch.abs(p_dx - t_dx)
        gdl_y = torch.abs(p_dy - t_dy)
        
        if mask is not None:
            # Mask 也许相应裁剪
            # Mask X: (..., H, W-1)
            mask_x = mask[..., :, 1:] * mask[..., :, :-1]
            # Mask Y: (..., H-1, W)
            mask_y = mask[..., 1:, :] * mask[..., :-1, :]
            
            # 加权求和并归一化
            loss_x = (gdl_x * mask_x).sum() / (mask_x.sum() + 1e-6)
            loss_y = (gdl_y * mask_y).sum() / (mask_y.sum() + 1e-6)
            
            return (loss_x + loss_y) * self.alpha
            
        # 无 Mask 情况，直接求均值
        return (gdl_x.mean() + gdl_y.mean()) * self.alpha

class HybridLoss(nn.Module):
    """
    针对短临降水场景的混合 Loss
    
    组合策略：
    1. WeightedL1Loss: 负责像素级准确性，尤其是强降水和虚警抑制。
    2. GradientDifferenceLoss: 负责保持降水云团的纹理和边缘锐度。
    """
    def __init__(self, l1_weight=1.0, gdl_weight=1.0, **kwargs):
        super().__init__()
        self.weights = {'l1': l1_weight, 'gdl': gdl_weight}
        
        # 初始化 WeightedL1Loss (参数已内置优化，无需手动传入)
        self.l1 = WeightedL1Loss(
            zero_rain_weight=0.05, # 背景权重
            false_alarm_penalty=2.0, # 虚警惩罚
            loss_scale=10.0 # 缩放
        )
        
        self.gdl = GradientDifferenceLoss()

    def forward(self, pred, target, mask=None):
        loss_dict = {}
        
        # 计算分项损失
        l1 = self.l1(pred, target, mask)
        gdl = self.gdl(pred, target, mask)
        
        # 加权总和
        total = self.weights['l1'] * l1 + self.weights['gdl'] * gdl
        
        # 记录用于监控
        loss_dict['total'] = total
        loss_dict['l1'] = l1
        loss_dict['gdl'] = gdl
        
        return total, loss_dict