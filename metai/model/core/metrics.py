from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
try:
    from pytorch_msssim import ssim, ms_ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    
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


class SparsePrecipitationLoss(nn.Module):
    """
    稀疏降水损失函数 - 专门用于保持降水预测的稀疏性
    
    核心设计思想：
    1. **大幅增强非零值权重**：对降水区域给予极高的权重，强制模型学习降水模式
    2. **稀疏性正则化**：惩罚预测中的虚假非零值（当真实值为0时预测非0），鼓励预测保持稀疏
    3. **阈值权重**：对不同强度的降水给予不同权重，增强对强降水的关注
    4. **L1稀疏性约束**：L1损失天然鼓励稀疏性，有助于保持预测的稀疏特征
    
    损失函数组成：
    Loss = λ1 * WeightedMSE + λ2 * SparsityRegularization + λ3 * L1Loss
    
    其中：
    - WeightedMSE: 加权MSE损失，对非零值给予极高权重
    - SparsityRegularization: 稀疏性正则化，惩罚虚假非零预测
    - L1Loss: L1损失，鼓励预测保持稀疏
    
    适用场景：
    - 降水预测（数据极度稀疏，大部分区域为0）
    - 其他稀疏事件预测任务
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
                 temporal_weight_max: float = 2.0):
        """
        Args:
            positive_weight: 非零值的权重倍数（默认100.0，大幅增强对降水的关注）
            sparsity_weight: 稀疏性正则化项的权重（默认10.0）
            l1_weight: L1损失的权重（默认0.1）
            threshold: 区分"零值"和"非零值"的阈值（默认0.01）
            precipitation_thresholds: 降水阈值列表，用于生成阈值权重（默认 [0.1, 1.0, 2.0, 5.0]）
            precipitation_weights: 对应每个阈值区间的权重列表（默认 None，自动生成）
            reduction: 损失归约方式，'mean'或'sum'（默认'mean'）
            eps: 数值稳定性参数，防止除零
            temporal_weight_enabled: 是否启用时间步加权（默认True），后期时间步权重更高
            temporal_weight_max: 时间步权重的最大值（默认2.0），权重从1.0线性递增到该值
        """
        super(SparsePrecipitationLoss, self).__init__()
        
        self.positive_weight = positive_weight
        self.sparsity_weight = sparsity_weight
        self.l1_weight = l1_weight
        self.threshold = threshold
        self.reduction = reduction
        self.eps = eps
        self.temporal_weight_enabled = temporal_weight_enabled
        self.temporal_weight_max = temporal_weight_max
        
        # 降水阈值配置
        if precipitation_thresholds is None:
            precipitation_thresholds = [0.1/30, 1.0/30, 2.0/30, 5.0/30, 8/30]
            precipitation_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.precipitation_thresholds = precipitation_thresholds
        self.precipitation_weights = precipitation_weights
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测值，形状为 [B, T, C, H, W] 或 [B, T, H, W]
            target: 真实值，形状为 [B, T, C, H, W] 或 [B, T, H, W]
        
        Returns:
            损失值（标量）
        """
        # ========== 输入验证和维度处理 ==========
        # 处理通道数不匹配的情况
        if len(pred.shape) == 5 and len(target.shape) == 5:
            if pred.shape[2] > 1 and target.shape[2] == 1:
                pred = pred[:, :, 0:1, :, :]
        
        if pred.shape != target.shape:
            raise ValueError(f"pred和target的形状不匹配: {pred.shape} vs {target.shape}")
        
        # 处理不同的输入维度
        if len(pred.shape) == 5:  # [B, T, C, H, W]
            if pred.shape[2] == 1:
                pred = pred.squeeze(2)  # [B, T, H, W]
                target = target.squeeze(2)  # [B, T, H, W]
            else:
                # 多通道：对每个通道分别计算损失后平均
                losses = []
                for c in range(pred.shape[2]):
                    loss_c = self._forward_single(
                        pred[:, :, c, :, :], 
                        target[:, :, c, :, :]
                    )
                    losses.append(loss_c)
                return torch.stack(losses).mean()
        elif len(pred.shape) != 4:  # 不是 [B, T, H, W]
            raise ValueError(f"不支持的输入维度: {len(pred.shape)}，期望4或5维")
        
        return self._forward_single(pred, target)
    
    def _forward_single(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        单通道前向传播（内部方法）
        Args:
            pred: [B, T, H, W]
            target: [B, T, H, W]
        """
        B, T, H, W = pred.shape
        
        # ========== 0. 计算时间步权重：后期时间步权重更高 ==========
        if self.temporal_weight_enabled:
            # 线性递增权重：从1.0递增到temporal_weight_max
            time_weights = torch.linspace(1.0, self.temporal_weight_max, T, device=pred.device)  # [T]
            time_weights = time_weights.view(1, T, 1, 1)  # [1, T, 1, 1] 用于广播
        else:
            time_weights = None
        
        # ========== 1. 计算基础权重 ==========
        # 识别非零值（降水区域）
        is_positive = (target > self.threshold).float()  # [B, T, H, W]
        
        # 基础权重：非零值区域给予极高权重
        base_weight = torch.ones_like(target)
        base_weight = base_weight + is_positive * (self.positive_weight - 1.0)
        
        # ========== 2. 阈值权重（如果启用）==========
        if self.precipitation_thresholds is not None:
            # create_threshold_weights 可以处理 [B, T, H, W] 形状
            threshold_weights = SpatialFocalMseLoss.create_threshold_weights(
                target,
                thresholds=self.precipitation_thresholds,
                weights=self.precipitation_weights
            )
            # 如果返回的是5维，需要squeeze通道维度
            if len(threshold_weights.shape) == 5 and threshold_weights.shape[2] == 1:
                threshold_weights = threshold_weights.squeeze(2)
            base_weight = base_weight * threshold_weights
        
        # ========== 3. 加权MSE损失 ==========
        error = pred - target
        weighted_mse = base_weight * (error ** 2)
        
        # ========== 4. 稀疏性正则化 ==========
        # 惩罚虚假非零预测：当真实值为0时，如果预测值>0，则给予惩罚
        # 这鼓励模型在无降水区域预测0，保持稀疏性
        false_positive_mask = ((target <= self.threshold) & (pred > self.threshold)).float()
        sparsity_penalty = false_positive_mask * (pred ** 2)  # 惩罚虚假非零预测
        
        # ========== 5. L1损失（鼓励稀疏性）==========
        l1_loss = torch.abs(error)
        
        # ========== 6. 组合损失（应用时间步权重）==========
        total_loss = weighted_mse + self.sparsity_weight * sparsity_penalty + self.l1_weight * l1_loss
        
        # 应用时间步权重（如果启用）
        if time_weights is not None:
            total_loss = total_loss * time_weights
        
        # 根据reduction参数返回结果
        if self.reduction == 'mean':
            return total_loss.mean()
        else:  # 'sum'
            return total_loss.sum()