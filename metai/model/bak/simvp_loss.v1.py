from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    MultiScaleStructuralSimilarityIndexMeasure = None


def create_threshold_weights_optimized(target: torch.Tensor, 
                                     thresholds: List[float],
                                     weights: Optional[List[float]] = None) -> torch.Tensor:
    """
    [优化版] 根据降水阈值创建权重张量。
    使用 torch.bucketize 替代循环掩码，大幅提升 GPU 计算效率。
    
    Args:
        target: 真实值张量，形状为 [B, T, H, W] 或 [B, T, C, H, W]
        thresholds: 降水阈值列表，按升序排列
        weights: 对应每个阈值区间的权重列表（默认 None，自动生成）
    
    Returns:
        权重张量，形状与target相同
    """
    if weights is None:
        n_intervals = len(thresholds) + 1
        weights = [0.5 + i * 0.5 for i in range(n_intervals)]
    
    if len(weights) != len(thresholds) + 1:
        raise ValueError(f"权重数量({len(weights)})应该比阈值数量({len(thresholds)})多1")

    # 转换为 tensor 以使用 bucketize (确保在同一设备)
    thresholds_tensor = torch.tensor(thresholds, device=target.device, dtype=target.dtype)
    weights_tensor = torch.tensor(weights, device=target.device, dtype=target.dtype)
    
    # 使用 bucketize 快速查找索引
    # boundaries: [t1, t2, t3]
    # input < t1 -> index 0
    # t1 <= input < t2 -> index 1
    # ...
    indices = torch.bucketize(target, thresholds_tensor)
    
    # 根据索引获取权重
    weight_map = weights_tensor[indices]
    
    return weight_map


def create_threshold_weights(target: torch.Tensor, 
                             thresholds: List[float] = [0.1, 1.0, 2.0, 5.0],
                             weights: Optional[List[float]] = None) -> torch.Tensor:
    """
    根据降水阈值创建权重张量
    
    物理意义：
    对不同强度的降水给予不同权重，增强对强降水的关注。
    根据阈值将降水强度分为多个等级，每个等级给予不同的权重。
    
    Args:
        target: 真实值张量，形状为 [B, T, H, W] 或 [B, T, C, H, W]
        thresholds: 降水阈值列表，按升序排列
        weights: 对应每个阈值区间的权重列表（默认 None，自动生成）
    
    Returns:
        权重张量，形状与target相同
    """
    original_shape = target.shape
    if len(target.shape) == 5:
        # [B, T, C, H, W] -> [B*T*C, H, W]
        target_flat = target.view(-1, target.shape[-2], target.shape[-1])
    elif len(target.shape) == 4:
        # [B, T, H, W] -> [B*T, H, W]
        target_flat = target.view(-1, target.shape[-2], target.shape[-1])
    else:
        raise ValueError(f"不支持的输入维度: {len(target.shape)}，期望4或5维")
    
    # 默认权重
    if weights is None:
        n_intervals = len(thresholds) + 1
        base_weight = 0.5
        weight_step = 0.5
        weights = [base_weight + i * weight_step for i in range(n_intervals)]
    else:
        if len(weights) != len(thresholds) + 1:
            raise ValueError(f"权重数量({len(weights)})应该比阈值数量({len(thresholds)})多1")
    
    # 创建权重张量
    weight_tensor = torch.zeros_like(target_flat)
    
    # 从最高阈值开始，逐步向下分配
    weight_tensor.fill_(weights[0])  # 最低区间（无降水或微量降水）
    
    for i in range(len(thresholds)):
        # 当前阈值对应的区间使用对应的权重
        mask = target_flat >= thresholds[i]
        weight_tensor[mask] = weights[i + 1]
    
    # 重塑回原始形状
    weight_tensor = weight_tensor.view(original_shape)
    
    return weight_tensor


class SparsePrecipitationLoss(nn.Module):
    """
    稀疏降水损失函数 - 专门用于保持降水预测的稀疏性
    (包含 WeightedMSE, SparsityRegularization, L1Loss, TimeWeight, ThresholdWeight, MS-SSIM, TemporalConsistency)
    """
    
    def __init__(self, 
                 positive_weight: float = 100.0,
                 sparsity_weight: float = 10.0,
                 l1_weight: float = 0.1,
                 threshold: float = 0.01,
                 precipitation_thresholds: Optional[List[float]] = None,
                 precipitation_weights: Optional[List[float]] = None,
                 reduction: str = 'mean',
                 eps: float = 1e-6,
                 temporal_weight_enabled: bool = True,
                 temporal_weight_max: float = 2.0,
                 ssim_weight: Optional[float] = None,
                 temporal_consistency_weight: float = 0.5):
        super(SparsePrecipitationLoss, self).__init__()
        
        self.positive_weight = positive_weight
        self.sparsity_weight = sparsity_weight
        self.l1_weight = l1_weight
        self.threshold = threshold
        self.reduction = reduction
        self.eps = eps
        self.temporal_weight_enabled = temporal_weight_enabled
        self.temporal_weight_max = temporal_weight_max
        self.ssim_weight = ssim_weight if ssim_weight is not None and ssim_weight > 0 else None
        self.temporal_consistency_weight = temporal_consistency_weight
        
        # 降水阈值配置
        if precipitation_thresholds is None:
            # 假设输入已归一化 (例如 0-1 代表 0-30mm/h)
            precipitation_thresholds = [0.1/30, 1.0/30, 2.0/30, 5.0/30, 8.0/30]
            precipitation_weights = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]  # 强降水给予更高权重
        self.precipitation_thresholds = precipitation_thresholds
        self.precipitation_weights = precipitation_weights
        
        # MS-SSIM 初始化
        self.use_ssim = False
        if self.ssim_weight is not None and self.ssim_weight > 0:
            if TORCHMETRICS_AVAILABLE and MultiScaleStructuralSimilarityIndexMeasure is not None:
                self.use_ssim = True
                # MS-SSIM 作为一个子模块，会自动随主模块移动到 GPU
                self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
                    data_range=1.0,
                    kernel_size=7, 
                    betas=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)[:3],
                    normalize="relu"
                )
            else:
                import warnings
                warnings.warn("Warning: torchmetrics not found. SSIM loss disabled.")
                self.ms_ssim = None
                self.use_ssim = False
        else:
            self.ms_ssim = None

    def _forward_core(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算核心损失（WeightedMSE + SparsityRegularization + L1Loss），返回未归约的 [B, T, H, W] 张量。
        """
        B, T, H, W = pred.shape
        
        # ========== 0. 计算时间步权重：后期时间步权重更高 ==========
        if self.temporal_weight_enabled:
            time_weights = torch.linspace(1.0, self.temporal_weight_max, T, device=pred.device)  # [T]
            time_weights = time_weights.view(1, T, 1, 1)  # [1, T, 1, 1] 用于广播
        else:
            time_weights = None
        
        # ========== 1. 计算基础权重（非零值增强）==========
        is_positive = (target > self.threshold).float() 
        base_weight = torch.ones_like(target)
        base_weight = base_weight + is_positive * (self.positive_weight - 1.0)
        
        # ========== 2. 阈值权重（使用优化版本）==========
        if self.precipitation_thresholds is not None:
            # 使用优化版本的 bucketize 方法，提升 GPU 计算效率
            threshold_weights = create_threshold_weights_optimized(
                target,
                thresholds=self.precipitation_thresholds,
                weights=self.precipitation_weights
            )
            base_weight = base_weight * threshold_weights
        
        # ========== 3. 加权MSE损失 ==========
        error = pred - target
        weighted_mse = base_weight * (error ** 2)
        
        # ========== 4. 稀疏性正则化（惩罚虚警: target无雨但pred有雨）==========
        # 使用 ReLU 简化逻辑: max(0, pred - threshold) 当 target <= threshold
        no_rain_mask = (target <= self.threshold).float()
        pred_noise = F.relu(pred - self.threshold)
        sparsity_penalty = no_rain_mask * (pred_noise ** 2) 
        
        # ========== 5. L1损失（鼓励稀疏性）==========
        l1_loss = torch.abs(error)
        
        # ========== 6. 组合损失（应用时间步权重）==========
        total_loss = weighted_mse + self.sparsity_weight * sparsity_penalty + self.l1_weight * l1_loss
        
        if time_weights is not None:
            total_loss = total_loss * time_weights
        
        return total_loss # [B, T, H, W] 张量

    def _compute_temporal_consistency_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算时序一致性损失（Temporal Consistency Loss）
        
        物理意义：
        惩罚预测序列中相邻时间步变化率与真实序列变化率的差异，减少时序抖动（Temporal Flickering），
        提高预测的时序平滑度，避免推理时使用移动平均平滑导致的相位滞后。
        
        Args:
            pred: 预测值，形状为 [B, T, H, W]
            target: 真实值，形状为 [B, T, H, W]
        
        Returns:
            时序一致性损失值（标量）
        """
        # 计算一阶时序差分损失
        if pred.shape[1] < 2:
            return torch.tensor(0.0, device=pred.device)
            
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        return F.l1_loss(pred_diff, target_diff, reduction='mean')
    
    def _combine_and_reduce_loss(self, loss_raw: torch.Tensor, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """统一处理最终归约、MS-SSIM 损失和时序一致性损失。"""
        
        # 核心损失归约
        if self.reduction == 'mean':
            final_loss = loss_raw.mean()
        else:
            final_loss = loss_raw.sum()
            
        # MS-SSIM 损失（如果启用）
        if self.use_ssim and self.ssim_weight is not None and self.ms_ssim is not None:
            ssim_loss = self._compute_ssim_loss(pred, target)
            final_loss = final_loss + self.ssim_weight * ssim_loss
        
        # 时序一致性损失（如果启用且权重 > 0）
        if self.temporal_consistency_weight > 0:
            temporal_loss = self._compute_temporal_consistency_loss(pred, target)
            final_loss = final_loss + self.temporal_consistency_weight * temporal_loss
            
        return final_loss


    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测值，形状为 [B, T, C, H, W] 或 [B, T, H, W]
            target: 真实值，形状为 [B, T, C, H, W] 或 [B, T, H, W]
        
        Returns:
            损失值（标量）
        """
        # ========== 输入验证和维度处理 ==========
        
        if pred.shape != target.shape:
            if len(pred.shape) == 5 and len(target.shape) == 5:
                if pred.shape[2] > target.shape[2] and target.shape[2] == 1:
                    # 预测值通道数大于目标值通道数 (预测多个特征，只取第一个通道)
                    pred = pred[:, :, 0:1, :, :] 
                else:
                    raise ValueError(f"pred和target的形状不匹配: {pred.shape} vs {target.shape}")
            else:
                raise ValueError(f"pred和target的形状不匹配: {pred.shape} vs {target.shape}")
        
        # 处理不同的输入维度
        if len(pred.shape) == 5:  # [B, T, C, H, W]
            if pred.shape[2] == 1:
                # 单通道情况 (C=1): Squeeze 转换为 4D
                pred_4d = pred.squeeze(2)  # [B, T, H, W]
                target_4d = target.squeeze(2)  # [B, T, H, W]
                loss_raw = self._forward_core(pred_4d, target_4d)
                return self._combine_and_reduce_loss(loss_raw, pred_4d, target_4d)
            else:
                # 多通道情况 (C>1): 对每个通道分别计算核心损失
                losses_raw = []
                for c in range(pred.shape[2]):
                    # _forward_core 返回的是未归约的 [B, T, H, W] 损失张量
                    loss_c_raw = self._forward_core(
                        pred[:, :, c, :, :], 
                        target[:, :, c, :, :]
                    )
                    losses_raw.append(loss_c_raw)
                
                # 核心损失：对所有通道的原始损失张量求平均，得到一个 [B, T, H, W] 张量
                total_loss_raw = torch.stack(losses_raw).mean(dim=0)
                
                # MS-SSIM 损失仅对通道 0 (降水) 计算
                pred_channel0 = pred[:, :, 0, :, :]
                target_channel0 = target[:, :, 0, :, :]
                
                return self._combine_and_reduce_loss(total_loss_raw, pred_channel0, target_channel0)
                
        elif len(pred.shape) != 4:  # 不是 [B, T, H, W]
            raise ValueError(f"不支持的输入维度: {len(pred.shape)}，期望4或5维")
        
        # 4D 输入: [B, T, H, W]
        loss_raw = self._forward_core(pred, target)
        return self._combine_and_reduce_loss(loss_raw, pred, target)
    
    
    def _compute_ssim_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算 MS-SSIM 损失（内部方法）
        
        Args:
            pred: [B, T, H, W]
            target: [B, T, H, W]
        
        Returns:
            SSIM 损失值（标量）：(1 - MS-SSIM) 的平均值
        """
        B, T, H, W = pred.shape
        min_side = min(H, W)
        
        # MS-SSIM 需要特定尺寸，如果太小则跳过
        if min_side < 32: 
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Reshape for metric: [B*T, C=1, H, W]
        pred_flat = torch.clamp(pred, 0, 1).view(-1, 1, H, W)
        target_flat = torch.clamp(target, 0, 1).view(-1, 1, H, W)
        
        if self.ms_ssim is None:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        try:
            ssim_score = self.ms_ssim(pred_flat, target_flat)
            return 1.0 - ssim_score
        except Exception:
            # 捕获可能的尺寸不匹配错误
            return torch.tensor(0.0, device=pred.device, requires_grad=True)