from typing import List, Optional, Union, Any
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

try:
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    MultiScaleStructuralSimilarityIndexMeasure = None
    # 占位符函数 (保持简洁，防止导入失败)
    def ssim_loss(pred, target): return F.mse_loss(pred, target)


def temporal_consistency_loss(pred: torch.Tensor) -> torch.Tensor:
    """
    计算时序一致性损失（Temporal Consistency Loss）
    
    物理意义：
    惩罚预测序列中相邻时间步的剧烈变化，减少时序抖动（Temporal Flickering），
    提高预测的时序平滑度。
    
    Args:
        pred: 预测值，形状为 [B, T, C, H, W] 或 [B, T, H, W]
    
    Returns:
        时序一致性损失值（标量）
    """
    # 处理不同的输入维度
    if len(pred.shape) == 5:
        # [B, T, C, H, W] -> [B, T, H, W] (取第一个通道或平均)
        if pred.shape[2] == 1:
            pred = pred.squeeze(2)  # [B, T, H, W]
        else:
            pred = pred.mean(dim=2)  # [B, T, H, W]
    
    # 如果时间步数小于2，返回0
    if pred.shape[1] < 2:
        return torch.tensor(0.0, device=pred.device)
    
    # 计算相邻时间步的差分
    pred_diff = pred[:, 1:] - pred[:, :-1]  # [B, T-1, H, W]
    
    # 计算差分的L2范数（鼓励平滑变化）
    temporal_loss = torch.mean(pred_diff ** 2)
    
    return temporal_loss

class EvolutionLoss(nn.Module):
    """
    [新增] 物理演变损失 (Physics-Guided Evolution Loss)
    
    理论依据: 
    基于雷达回波的平流方程 (Advection Equation) 近似: dI/dt + v * grad(I) = 0。
    如果模型的位置预测出现偏差，会导致预测场的时间导数 (dI/dt) 与真实场不一致。
    通过最小化演变梯度的误差，我们引入了隐式的运动约束，强迫模型修正位置偏差。
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.l1 = nn.L1Loss(reduction='mean')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, T, H, W] (已归一化 0-1)
            target: [B, T, H, W]
        """
        # 维度兼容处理
        if pred.dim() == 5: pred = pred.squeeze(2)
        if target.dim() == 5: target = target.squeeze(2)
            
        if pred.shape[1] < 2:
            return torch.tensor(0.0, device=pred.device)

        # 计算一阶时间差分 (Finite Difference)
        # Pred 变化量
        pred_diff = pred[:, 1:] - pred[:, :-1]
        # True 变化量
        target_diff = target[:, 1:] - target[:, :-1]

        # 惩罚两者的差异
        loss = self.l1(pred_diff, target_diff)
        
        return self.weight * loss

def create_threshold_weights(target: torch.Tensor, 
                             thresholds: List[float],
                             weights: Optional[List[float]] = None) -> torch.Tensor:
    """[优化版] 根据降水阈值创建权重张量，使用 torch.bucketize。"""
    if weights is None:
        n_intervals = len(thresholds) + 1
        weights = [0.5 + i * 0.5 for i in range(n_intervals)]
    
    if len(weights) != len(thresholds) + 1:
        raise ValueError(f"权重数量({len(weights)})应该比阈值数量({len(thresholds)})多1")

    thresholds_tensor = torch.tensor(thresholds, device=target.device, dtype=target.dtype)
    weights_tensor = torch.tensor(weights, device=target.device, dtype=target.dtype)
    
    indices = torch.bucketize(target, thresholds_tensor)
    weight_map = weights_tensor[indices]
    
    return weight_map


class SparsePrecipitationLoss(nn.Module):
    """
    稀疏降水损失函数 - 专为保持降水预测的稀疏性设计，与 Logit Space 和裁判评分 W_k 对齐。
    """
    
    def __init__(self, 
                 positive_weight: float = 100.0,
                 sparsity_weight: float = 5.0,     # 修正：降低对虚警的惩罚强度
                 l1_weight: float = 0.5,           # 修正：L1 Hard Start 权重
                 bce_weight: float = 8.0,
                 threshold: float = 0.01,
                 precipitation_thresholds: Optional[List[float]] = None,
                 precipitation_weights: Optional[List[float]] = None,
                 reduction: str = 'mean',
                 eps: float = 1e-6,
                 temporal_weight_enabled: bool = False,
                 temporal_weight_max: float = 2.0,
                 evolution_weight: float = 0.0,
                 ssim_weight: Optional[float] = 0.3,
                 temporal_consistency_weight: float = 0.1, # 修正：降低平滑偏好
                 referee_weights_w_k: Optional[List[float]] = None):
        super(SparsePrecipitationLoss, self).__init__()
        
        self.positive_weight = positive_weight
        self.sparsity_weight = sparsity_weight
        self.l1_weight = l1_weight
        self.bce_weight = bce_weight
        self.threshold = threshold
        self.reduction = reduction
        self.eps = eps
        self.temporal_weight_enabled = temporal_weight_enabled
        self.temporal_weight_max = temporal_weight_max
        self.ssim_weight = ssim_weight if ssim_weight is not None and ssim_weight > 0 else None
        self.evolution_weight = evolution_weight
        if self.evolution_weight > 0:
            self.evo_loss = EvolutionLoss(weight=self.evolution_weight)
        else:
            self.evo_loss = None
        self.temporal_consistency_weight = temporal_consistency_weight
        
        # 🚨 核心修正: Logit Space Loss - BCEWithLogitsLoss 替代 MSELoss (BCE代理)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none') 
        self.l1_loss = nn.L1Loss(reduction='none')

        # 降水阈值配置
        if precipitation_thresholds is None:
            # 竞赛默认阈值 (归一化)
            self.precipitation_thresholds = [0.1/30.0, 1.0/30.0, 2.0/30.0, 5.0/30.0, 8.0/30.0]
            self.precipitation_weights = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0] 
        else:
             self.precipitation_thresholds = precipitation_thresholds
             self.precipitation_weights = precipitation_weights
        
        # 时序权重 W_k
        if referee_weights_w_k is not None:
            self.register_buffer('w_k', torch.tensor(referee_weights_w_k, dtype=torch.float32).view(1, -1, 1, 1))
        else:
             self.w_k = None
        
        # MS-SSIM 初始化
        self.use_ssim = False
        if self.ssim_weight is not None and self.ssim_weight > 0 and TORCHMETRICS_AVAILABLE:
            self.use_ssim = True
            # Type assertion: MultiScaleStructuralSimilarityIndexMeasure is guaranteed to be available here
            assert MultiScaleStructuralSimilarityIndexMeasure is not None, "MultiScaleStructuralSimilarityIndexMeasure should be available when TORCHMETRICS_AVAILABLE is True"
            self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
                data_range=1.0, kernel_size=7, betas=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)[:3], normalize="relu"
            )
        else:
             self.ms_ssim = None


    def forward(self, logits_pred: torch.Tensor, target: torch.Tensor, target_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算稀疏降水预测的组合损失。
        Args:
            logits_pred: 模型输出的 Logits Z (B, T, C, H, W)
            target: 真实目标值 (B, T, C, H, W)
            target_mask: 掩码 (B, T, C, H, W)
        """
        
        # 1. 数据维度处理 (将 C 维度压平或移除，针对 C=1 降水通道)
        if len(logits_pred.shape) == 5:
            logits_pred_4d = logits_pred.squeeze(2) 
            target_4d = target.squeeze(2)
            target_mask_4d = target_mask.squeeze(2) if target_mask is not None else None
        else:
             logits_pred_4d = logits_pred
             target_4d = target
             target_mask_4d = target_mask

        # 2. 核心预测 (Pred Space, [0, 1])
        # Pred 用于 L1, SSIM, Sparsity 惩罚
        pred_4d = torch.sigmoid(logits_pred_4d)
        pred_clamped_4d = torch.clamp(pred_4d, 0.0, 1.0) 
        
        # 3. 损失项计算 (基础损失，reduction='none')
        
        # L1 Loss (在 Pred Space, 使用 clamped output)
        l1_comp = self.l1_loss(pred_clamped_4d, target_4d) 
        
        # BCE Loss (在 Logit Space, 避免梯度截断)
        bce_comp = self.bce_loss(logits_pred_4d, target_4d) 

        # 4. 动态权重计算 (Positive + Sparsity)
        
        # 降水区域掩码 (Positives)
        mask_pos = (target_4d > self.threshold)
        mask_neg = ~mask_pos

        # 虚警区域掩码 (False Positives): 真实为0，预测高于阈值
        mask_false_pos = torch.logical_and(mask_neg, pred_clamped_4d > self.threshold)
        
        # 初始化权重图
        weight_map = torch.ones_like(target_4d, dtype=target_4d.dtype)
        
        # 应用 Positive Weight
        weight_map[mask_pos] *= self.positive_weight
        
        # 应用 Sparsity Weight (惩罚虚警)
        weight_map[mask_false_pos] += self.sparsity_weight
        
        # 5. 组合像素级损失 (加权 L1 + 加权 BCE)
        
        # 核心损失项：(L1 * w_l1) + (BCE * w_bce)
        pixel_loss = l1_comp * self.l1_weight + bce_comp * self.bce_weight
        
        # 应用动态权重
        loss_weighted = pixel_loss * weight_map
        
        # 6. 归约和时间步权重
        
        # 应用 Target Mask (忽略无效区域)
        if target_mask_4d is not None:
             valid_mask = target_mask_4d.bool() 
             loss_weighted = loss_weighted * valid_mask.float()
             count = valid_mask.sum() + self.eps 
        else:
             count = torch.numel(target_4d) + self.eps 
        
        # 应用 W_k 时间权重 (裁判评分权重)
        if self.w_k is not None:
             # 确保 w_k 的时间维度与数据匹配 (T)
             time_weights_expanded = self.w_k.to(loss_weighted.device)
             loss_weighted = loss_weighted * time_weights_expanded 
        
        # 最终归约到单个 Loss 值
        total_loss = loss_weighted.sum() / count

        # 7. 结构和时序损失 (使用 Pred Space Clamp后的输出)
        
        # MS-SSIM Loss
        if self.use_ssim:
             ssim_score = self._compute_ssim_score(pred_clamped_4d, target_4d)
             ssim_loss_val = 1.0 - ssim_score
             total_loss += self.ssim_weight * ssim_loss_val

        # 🆕 应用物理演变损失 (Evolution Loss)
        if self.evo_loss is not None:
            # 注意：必须传入 [0,1] 范围的预测值 (pred_clamped_4d)
            e_loss = self.evo_loss(pred_clamped_4d, target_4d)
            total_loss += e_loss
        
        # Temporal Consistency Loss
        if self.temporal_consistency_weight > 0:
             # 注意: temporal_consistency_loss 内部通常会做 mean/sum 归约，需要谨慎使用乘法权重
             t_cons = temporal_consistency_loss(pred_clamped_4d.unsqueeze(2)) # 确保输入是 5D
             total_loss += self.temporal_consistency_weight * t_cons
        
        return total_loss
    
    
    def _compute_ssim_score(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算 MS-SSIM 分数（1.0 最好）"""
        H, W = pred.shape[-2:]
        if min(H, W) < 32: return torch.tensor(1.0, device=pred.device)

        # 增加通道维度 C=1
        pred_flat = pred.view(-1, 1, H, W)
        target_flat = target.view(-1, 1, H, W)
        
        if self.ms_ssim is None: return torch.tensor(1.0, device=pred.device)
        
        try:
            self.ms_ssim = self.ms_ssim.to(pred.device)
            ssim_score = self.ms_ssim(pred_flat, target_flat)
            return ssim_score
        except Exception:
            return torch.tensor(1.0, device=pred.device)