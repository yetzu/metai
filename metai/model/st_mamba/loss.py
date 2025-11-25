from typing import List,Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

try:
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False

class SpatialFocalMseLoss(nn.Module):
    """
    空间感知的Focal MSE损失函数
    
    物理意义：
    该损失函数结合了三种权重机制，用于处理时空预测任务中的特殊挑战：
    
    1. **类别权重 (Class Weight)**：
       - 物理意义：处理类别不平衡问题（如降雨预测中零值vs非零值）
       - 原理：自动计算零值/非零值的比例，给少数类更高的权重
       - 公式：w_class = n_minority / n_majority
       - 作用：防止模型过度关注多数类（如无雨区域），忽略少数类（如降雨区域）
    
    2. **Focal权重 (Focal Weight)**：
       - 物理意义：关注难样本，降低易样本的贡献
       - 原理：当预测误差大时（难样本），权重接近1；误差小时（易样本），权重接近0
       - 公式：w_focal = (1 - sigmoid(pred - target))^gamma
       - 作用：让模型专注于学习难以预测的样本，提高对边界情况和异常值的敏感性
       - gamma参数：控制难易样本的区分度，gamma越大，对难样本的聚焦越强
    
    3. **空间权重 (Spatial Weight)**：
       - 物理意义：考虑空间邻域信息，强调空间结构的重要性
       - 原理：使用3x3平均池化核计算每个像素周围邻域的平均值
       - 公式：w_spatial = mean(neighborhood(target))
       - 作用：
         * 高值区域（如降雨中心）：邻域值高 → 权重高 → 更关注
         * 低值区域（如无雨区）：邻域值低 → 权重低 → 降低关注
         * 边界区域：邻域值中等 → 中等权重 → 保持关注
       - 物理直觉：降雨等气象现象具有空间连续性，邻域信息有助于稳定训练
    
    4. **总损失公式**：
       Loss = mean( w_class × w_focal × w_spatial × (pred - target)^2 )
       
       物理意义：
       - 基础项：(pred - target)^2 是MSE损失，衡量预测误差
       - 三重权重：三个权重相乘，形成自适应的重要性分配
       - 综合效果：同时解决类别不平衡、难样本学习和空间结构保持三个问题
    
    适用场景：
    - 时空序列预测（如降雨、温度等气象数据）
    - 稀疏事件预测（零值占主导的数据）
    - 需要保持空间连续性的预测任务
    

    """
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, eps: float = 1e-6,
                 kernel_size: int = 3, threshold: float = 0.0, reduction: str = 'mean',
                 threshold_weights: Optional[torch.Tensor] = None,
                 precipitation_thresholds: Optional[List[float]] = None,
                 precipitation_weights: Optional[List[float]] = None):
        """
        Args:
            alpha: 类别权重的全局缩放因子，控制类别平衡的强度
            gamma: Focal权重的指数参数，控制难易样本的区分度（gamma越大，越关注难样本）
            eps: 数值稳定性参数，防止除零
            kernel_size: 空间卷积核大小（必须是奇数，默认3）
            threshold: 类别分割阈值，用于区分"零值"和"非零值"（默认0.0）
            reduction: 损失归约方式，'mean'或'sum'（默认'mean'）
            threshold_weights: 可选的阈值权重张量，形状与target相同，用于对不同强度的降水给予不同权重
            precipitation_thresholds: 降水阈值列表，用于自动生成权重（默认 [0.1, 1.0, 2.0, 5.0]）
                                     如果提供此参数，将在forward时根据target自动生成权重
            precipitation_weights: 对应每个阈值区间的权重列表（默认 None，自动生成）
                                   如果为None，将根据阈值自动生成递增的权重
        """
        super(SpatialFocalMseLoss, self).__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.reduction = reduction
        self.threshold_weights = threshold_weights
        
        # 降水阈值配置（用于自动生成权重）
        if precipitation_thresholds is None:
            precipitation_thresholds = [0.1, 1.0, 2.0, 5.0]  # 默认阈值
        self.precipitation_thresholds = precipitation_thresholds
        self.precipitation_weights = precipitation_weights
        
        # 空间卷积核（适配单通道）
        # 物理意义：平均池化核，用于计算空间邻域的平均值
        # 注册为buffer，确保能正确移动到GPU
        kernel_value = 1.0 / (kernel_size * kernel_size)
        self.register_buffer('spatial_kernel', 
                            torch.ones(1, 1, kernel_size, kernel_size) * kernel_value)
        pad_size = kernel_size // 2
        self.pad = nn.ZeroPad2d(pad_size)
    
    @staticmethod
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
            thresholds: 降水阈值列表，按升序排列（默认 [0.1, 1.0, 2.0, 5.0]）
                       阈值含义：
                       - 0.1: 微量降水阈值（小雨）
                       - 1.0: 小到中雨阈值
                       - 2.0: 中到大雨阈值
                       - 5.0: 大雨到暴雨阈值
            weights: 对应每个阈值区间的权重列表（默认 None，自动生成）
                     如果为None，将根据阈值自动生成递增的权重
                     权重数量应该比阈值数量多1（因为有n+1个区间）
        
        Returns:
            权重张量，形状与target相同
        
        示例:
            >>> target = torch.randn(2, 10, 100, 100)  # [B, T, H, W]
            >>> # 使用默认阈值和权重
            >>> weights = SpatialFocalMseLoss.create_threshold_weights(target)
            >>> # 自定义阈值和权重
            >>> weights = SpatialFocalMseLoss.create_threshold_weights(
            ...     target, 
            ...     thresholds=[0.1, 1.0, 2.0, 5.0],
            ...     weights=[0.5, 1.0, 1.5, 2.0, 3.0]  # 5个区间对应5个权重
            ... )
        """
        # 处理通道维度：如果是5维，先处理
        original_shape = target.shape
        if len(target.shape) == 5:
            # [B, T, C, H, W] -> [B, T, H, W]（对每个通道分别处理）
            B, T, C, H, W = target.shape
            target_flat = target.view(B * T * C, H, W)
        elif len(target.shape) == 4:
            # [B, T, H, W]
            B, T, H, W = target.shape
            target_flat = target.view(B * T, H, W)
        else:
            raise ValueError(f"不支持的输入维度: {len(target.shape)}，期望4或5维")
        
        # 默认权重：根据阈值自动生成递增权重
        if weights is None:
            n_intervals = len(thresholds) + 1
            # 权重设计：从0.5开始，每个区间递增0.5
            # 对于4个阈值[0.1, 1.0, 2.0, 5.0]，有5个区间：
            # 无降水[0, 0.1): 0.5, 微量[0.1, 1.0): 1.0, 小雨[1.0, 2.0): 1.5, 中雨[2.0, 5.0): 2.0, 大雨[5.0, inf): 2.5
            base_weight = 0.5
            weight_step = 0.5
            weights = [base_weight + i * weight_step for i in range(n_intervals)]
        else:
            if len(weights) != len(thresholds) + 1:
                raise ValueError(f"权重数量({len(weights)})应该比阈值数量({len(thresholds)})多1")
        
        # 创建权重张量
        weight_tensor = torch.zeros_like(target_flat)
        
        # 根据阈值分配权重（从高到低，确保高值覆盖低值）
        # 区间划分：
        # - 区间0: [0, thresholds[0]) -> weights[0]
        # - 区间1: [thresholds[0], thresholds[1]) -> weights[1]
        # - 区间2: [thresholds[1], thresholds[2]) -> weights[2]
        # - ...
        # - 区间n: [thresholds[n-1], inf) -> weights[n]
        
        # 从最高阈值开始，逐步向下分配
        weight_tensor.fill_(weights[0])  # 最低区间（无降水或微量降水）
        
        for i in range(len(thresholds)):
            # 当前阈值对应的区间使用对应的权重
            mask = target_flat >= thresholds[i]
            weight_tensor[mask] = weights[i + 1]
        
        # 重塑回原始形状
        if len(original_shape) == 5:
            # 从original_shape获取C维度
            _, _, C_orig, _, _ = original_shape
            weight_tensor = weight_tensor.view(B, T, C_orig, H, W)
        else:
            weight_tensor = weight_tensor.view(B, T, H, W)
        
        return weight_tensor
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测值，形状为 [B, T, C, H, W]（默认）或 [B, T, H, W]
            target: 真实值，形状为 [B, T, C, H, W] 或 [B, T, H, W]
                    如果pred有多个通道而target只有1个通道，则只使用pred的第一个通道计算损失
        
        Returns:
            损失值（标量）
        
        Note:
            - 如果输入是 [B, T, C, H, W] 且 C=1，会自动squeeze为 [B, T, H, W]
            - 如果输入是 [B, T, C, H, W] 且 C>1，会对每个通道分别计算损失后平均
            - 如果pred有多个通道而target只有1个通道，只使用pred的第一个通道与target计算损失
        """
        # ========== 输入验证和维度处理 ==========
        # 处理通道数不匹配的情况：pred有多个通道，target只有1个通道
        if len(pred.shape) == 5 and len(target.shape) == 5:
            if pred.shape[2] > 1 and target.shape[2] == 1:
                # 只使用pred的第一个通道（通常是CR通道，最重要）
                pred = pred[:, :, 0:1, :, :]  # 保持通道维度 [B, T, 1, H, W]
        
        # 检查其他维度是否匹配
        if pred.shape != target.shape:
            raise ValueError(f"pred和target的形状不匹配: {pred.shape} vs {target.shape}")
        
        # 使用局部变量处理threshold_weights，避免修改实例变量
        threshold_weights = self.threshold_weights
        
        # 如果threshold_weights为None，但提供了precipitation_thresholds，则自动生成权重
        if threshold_weights is None and self.precipitation_thresholds is not None:
            threshold_weights = self.create_threshold_weights(
                target, 
                thresholds=self.precipitation_thresholds,
                weights=self.precipitation_weights
            )
        
        # 处理不同的输入维度
        if len(pred.shape) == 5:  # [B, T, C, H, W]
            if pred.shape[2] == 1:
                # 单通道：squeeze掉通道维度
                pred = pred.squeeze(2)  # [B, T, H, W]
                target = target.squeeze(2)  # [B, T, H, W]
                # 如果threshold_weights存在，也需要squeeze通道维度
                if threshold_weights is not None:
                    if len(threshold_weights.shape) == 5 and threshold_weights.shape[2] == 1:
                        threshold_weights = threshold_weights.squeeze(2)  # [B, T, H, W]
            else:
                # 多通道：对每个通道分别计算损失后平均
                losses = []
                for c in range(pred.shape[2]):
                    # 为每个通道准备threshold_weights（使用局部变量）
                    channel_threshold_weights = None
                    if threshold_weights is not None:
                        if len(threshold_weights.shape) == 5:
                            channel_threshold_weights = threshold_weights[:, :, c, :, :]
                        else:
                            channel_threshold_weights = threshold_weights
                    loss_c = self._forward_single(pred[:, :, c, :, :], target[:, :, c, :, :], 
                                                  threshold_weights=channel_threshold_weights)
                    losses.append(loss_c)
                return torch.stack(losses).mean()
        elif len(pred.shape) == 4:  # [B, T, H, W]
            pass  # 已经是正确形状
        else:
            raise ValueError(f"不支持的输入维度: {len(pred.shape)}，期望4或5维")
        
        return self._forward_single(pred, target, threshold_weights=threshold_weights)
    
    def _forward_single(self, pred: torch.Tensor, target: torch.Tensor, 
                       threshold_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        单通道前向传播（内部方法）
        Args:
            pred: [B, T, H, W]
            target: [B, T, H, W]
            threshold_weights: 可选的阈值权重张量，如果为None则使用self.threshold_weights
        """
        B, T, H, W = pred.shape
        
        # 使用传入的threshold_weights，如果没有则使用实例变量
        if threshold_weights is None:
            threshold_weights = self.threshold_weights
        
        # ========== 1. 类别权重计算 ==========
        # 物理意义：自动平衡零值和非零值的权重，解决类别不平衡问题
        # 计算零值和非零值的数量（使用阈值而非固定0）
        is_positive = (target > self.threshold).float()
        n0 = (1.0 - is_positive).sum()  # 零值（多数类）的数量
        n1 = is_positive.sum()           # 非零值（少数类）的数量
        
        # 数值稳定性：如果某一类完全不存在，使用eps避免除零
        n0_safe = torch.clamp(n0, min=self.eps)
        n1_safe = torch.clamp(n1, min=self.eps)
        
        # 类别权重：少数类权重 = 多数类数量/少数类数量（放大少数类）
        #           多数类权重 = 少数类数量/多数类数量（缩小多数类）
        class_weight = torch.where(
            is_positive > 0.5,
            n0_safe / n1_safe,  # 少数类：放大权重
            n1_safe / n0_safe   # 多数类：缩小权重
        )
        # 结合用户指定的alpha参数，全局控制类别平衡的强度
        alpha_weight = self.alpha * class_weight
        
        # ========== 2. Focal权重计算 ==========
        # 物理意义：关注难样本（预测误差大的样本），降低易样本的贡献
        # 计算相对误差（归一化到合理范围，提高数值稳定性）
        error = pred - target
        error_abs = torch.abs(error)
        error_max = error_abs.max().detach()  # 避免梯度传播
        error_normalized = error_abs / (error_max + self.eps)
        
        # prob: 预测正确的概率（通过sigmoid将误差映射到[0,1]）
        # 使用归一化误差，提高数值稳定性
        # 修改：使用更合理的映射，避免prob过大导致focal_weight过小
        prob = torch.sigmoid(-error_normalized * 5.0 + 2.0)  # 调整sigmoid参数，使prob分布更合理
        
        # focal_weight: 当prob接近0（难样本，误差大）时，权重接近1
        #                当prob接近1（易样本，误差小）时，权重接近0
        # 修改：添加最小权重，避免focal_weight过小导致损失过小
        focal_weight_raw = (1 - prob) ** self.gamma
        focal_weight = torch.clamp(focal_weight_raw, min=0.1)  # 设置最小权重为0.1，避免过度缩小
        
        # ========== 3. 空间权重计算（向量化优化）==========
        # 物理意义：考虑空间邻域信息，强调空间结构的重要性
        # 改进：使用reshape和group conv实现向量化，避免循环
        # 将 [B, T, H, W] 重塑为 [B*T, 1, H, W] 以便批量卷积
        target_reshaped = target.view(B * T, 1, H, W)  # [B*T, 1, H, W]
        target_padded = self.pad(target_reshaped)  # [B*T, 1, H+pad, W+pad]
        
        # 批量卷积计算所有时间步的空间权重
        spatial_weight_flat = nn.functional.conv2d(
            target_padded, self.spatial_kernel, bias=None  # type: ignore
        )  # [B*T, 1, H, W]
        
        # 重塑回原始形状
        spatial_weight = spatial_weight_flat.view(B, T, H, W)  # [B, T, H, W]
        
        # ========== 4. 阈值权重（如果提供）==========
        # 物理意义：对不同强度的降水给予不同权重，增强对强降水的关注
        if threshold_weights is not None:
            # 确保阈值权重形状匹配
            if threshold_weights.shape != target.shape:
                raise ValueError(f"阈值权重形状不匹配: {threshold_weights.shape} vs {target.shape}")
            threshold_weight = threshold_weights
        else:
            threshold_weight = torch.ones_like(target)
        
        # ========== 5. 总损失计算 ==========
        # 物理意义：四重权重相乘，形成自适应的重要性分配
        # 总权重 = 类别权重 × Focal权重 × 空间权重 × 阈值权重
        # 修改：对空间权重进行归一化，避免权重过小
        # 空间权重归一化：将 [0, 1] 范围映射到 [0.5, 1.5] 范围，保持相对重要性但避免过度缩小
        spatial_weight_normalized = 0.5 + spatial_weight  # [0, 1] -> [0.5, 1.5]
        total_weight = alpha_weight * focal_weight * spatial_weight_normalized * threshold_weight
        
        # 加权MSE损失：权重越大，该样本的误差对总损失的贡献越大
        weighted_mse = total_weight * (error ** 2)
        
        # 根据reduction参数返回结果
        if self.reduction == 'mean':
            return weighted_mse.mean()
        else:  # 'sum'
            return weighted_mse.sum()

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
    def __init__(self, 
                 l1_weight=1.0, 
                 ssim_weight=0.5, 
                 csi_weight=1.0, 
                 spectral_weight=0.1, 
                 evo_weight=0.5,
                 focal_weight=0.0): # [New] 新增 Focal 权重
        super().__init__()
        self.weights = {
            'l1': l1_weight, 
            'ssim': ssim_weight, 
            'csi': csi_weight, 
            'spec': spectral_weight, 
            'evo': evo_weight,
            'focal': focal_weight # [New]
        }
        self.l1 = nn.L1Loss(reduction='none')
        
        if TORCHMETRICS_AVAILABLE and ssim_weight > 0:
            self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, reduction='none')
        else:
            self.ms_ssim = None
            
        self.soft_csi = WeightedScoreSoftCSILoss()
        self.spectral = LogSpectralDistanceLoss()
        self.evolution = WeightedEvolutionLoss()
        
        # [New] 初始化 Spatial Focal Loss
        self.focal_loss = SpatialFocalMseLoss(
            alpha=0.5, gamma=2.0, kernel_size=3, threshold=0.05
        )

    def forward(self, logits, target, mask=None):
        if logits.dim() == 5: logits = logits.squeeze(2)
        if target.dim() == 5: target = target.squeeze(2)
        if mask is not None and mask.dim() == 5: mask = mask.squeeze(2)
        
        pred = torch.sigmoid(logits)
        loss_dict = {}
        total_loss = 0.0
        
        # 1. L1 Loss
        l1_loss_map = self.l1(pred, target)
        if mask is not None:
            masked_error = l1_loss_map * mask
            count_valid = mask.sum()
            l1_loss = masked_error.sum() / count_valid if count_valid > 0 else 0.0
        else:
            l1_loss = l1_loss_map.mean()
            
        total_loss += self.weights['l1'] * l1_loss
        loss_dict['l1'] = l1_loss.item() if isinstance(l1_loss, torch.Tensor) else l1_loss
        
        # 2. Focal Loss [New]
        if self.weights['focal'] > 0:
            # SpatialFocalMseLoss 内部实现了基于内容的加权，通常无需额外 mask
            focal_val = self.focal_loss(pred, target)
            total_loss += self.weights['focal'] * focal_val
            loss_dict['focal'] = focal_val.item()
            
        # 3. CSI Loss
        if self.weights['csi'] > 0:
            csi_loss = self.soft_csi(pred, target, mask)
            total_loss += self.weights['csi'] * csi_loss
            loss_dict['csi'] = csi_loss.item()
            
        # 4. Spectral Loss
        if self.weights['spec'] > 0:
            spec_loss = self.spectral(pred, target, mask)
            total_loss += self.weights['spec'] * spec_loss
            loss_dict['spec'] = spec_loss.item()
            
        # 5. Evolution Loss
        if self.weights['evo'] > 0 and pred.shape[1] > 1:
            evo_loss = self.evolution(pred, target, mask)
            total_loss += self.weights['evo'] * evo_loss
            loss_dict['evo'] = evo_loss.item()
            
        # 6. SSIM Loss
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