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



import torch
import torch.nn as nn

class WeightedScoreSoftCSILoss(nn.Module):
    """
    [针对 SimVP/Mamba 优化] 评分规则对齐的 Soft-CSI 损失函数
    
    设计依据：
    严格遵循 Score_k 评分公式，引入：
    1. 强度加权 (Intensity Weights): 强降水权重更高 (0.3 vs 0.1)。
    2. 时效加权 (Temporal Weights): 关键时效权重更高 (60min权重是120min的20倍)。
    
    这能引导 Mamba 模型将有限的容量分配给得分贡献最大的时空区域。
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.MM_MAX = 30.0  # 数据归一化常数
        
        # --- 1. 对齐表2：要素分级及权重 ---
        # 注意：这里选取区间的左端点作为阈值。
        # 0.1-0.9 -> 0.1
        # 1.0-1.9 -> 1.0
        # 2.0-4.9 -> 2.0
        # 5.0-7.9 -> 5.0
        # >=8.0   -> 8.0
        thresholds_raw = [0.1, 1.0, 2.0, 5.0, 8.0]
        weights_raw    = [0.1, 0.1, 0.2, 0.2, 0.3] 
        
        # 归一化阈值
        self.register_buffer('thresholds', torch.tensor(thresholds_raw) / self.MM_MAX)
        # 归一化权重 (确保和为1，或者保持相对比例即可，这里保持原始比例)
        self.register_buffer('intensity_weights', torch.tensor(weights_raw))
        
        # --- 2. 对齐表1：预报时效及权重 ---
        # 对应序号 1 (6min) 到 20 (120min)
        time_weights_raw = [
            0.0075, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,  # 1-10 (60min最重要)
            0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.0075, 0.005 # 11-20
        ]
        # 转为 [1, T, 1, 1] 以便广播
        self.register_buffer('time_weights', torch.tensor(time_weights_raw).view(1, -1, 1, 1))
        
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred: [B, T, H, W], 范围 [0, 1]
        target: [B, T, H, W], 范围 [0, 1]
        """
        # 确保时间步长度匹配 (防止 pred 长度与权重表不一致)
        T = pred.shape[1]
        current_time_weights = self.time_weights[:, :T, :, :]
        
        # 归一化时间权重，使其平均值为 1，避免 Loss 数值过小
        # 这样做是为了让 Loss 的数值量级与不加权时保持在一个水平，方便调参
        current_time_weights = current_time_weights / current_time_weights.mean()

        total_weighted_loss = 0.0
        total_weight_sum = 0.0

        # 遍历每个强度阈值
        for i, t in enumerate(self.thresholds):
            w = self.intensity_weights[i]
            
            # --- Soft CSI 计算 ---
            # 1. 软二值化：使用 steep sigmoid (k=50) 模拟阶跃
            # 预测值 > 阈值 -> 1, 否则 -> 0
            pred_score = torch.sigmoid((pred - t) * 50)
            target_score = (target > t).float()
            
            # 2. 计算 Intersection (TP) 和 Union (TP + FN + FP)
            # 在 Spatial (H, W) 维度求和，保留 (B, T) 维度，以便应用时间权重
            intersection = (pred_score * target_score).sum(dim=(-2, -1))
            union = pred_score.sum(dim=(-2, -1)) + target_score.sum(dim=(-2, -1)) - intersection
            
            # 3. 计算每个时间步、每个样本的 CSI
            # csi: [B, T]
            csi = (intersection + self.smooth) / (union + self.smooth)
            
            # 4. 计算 Loss = 1 - CSI
            loss_map = 1.0 - csi
            
            # --- 关键改进：应用时间权重 ---
            # loss_map [B, T] * time_weights [1, T]
            # 结果是对 Time 加权后的 Loss
            weighted_loss_t = (loss_map * current_time_weights.squeeze(-1).squeeze(-1)).mean()
            
            # --- 关键改进：应用强度权重 ---
            total_weighted_loss += weighted_loss_t * w
            total_weight_sum += w

        # 返回加权平均后的 Loss
        return total_weighted_loss / total_weight_sum

class LogSpectralDistanceLoss(nn.Module):
    """
    频域损失。
    SimVP+Mamba 容易产生平滑纹理，此损失强制模型在频域保持高频分量，
    使生成的雷达回波图具有真实的锐度和纹理。
    """
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        # FFT 变换
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1), norm='ortho')
        target_fft = torch.fft.rfft2(target, dim=(-2, -1), norm='ortho')
        
        # 幅度谱
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # 对数距离 (平衡低频和高频的贡献)
        loss = F.l1_loss(torch.log(pred_mag + self.epsilon), torch.log(target_mag + self.epsilon))
        return loss

class WeightedEvolutionLoss(nn.Module):
    """
    物理感知的加权演变损失。
    原理：雷达回波的变化（一阶时间差分）应当符合物理规律。
    改进：对强回波区域的变化给予更高权重，因为强对流的生消是预报的难点。
    """
    def __init__(self, weight_scale=5.0):
        super().__init__()
        self.weight_scale = weight_scale

    def forward(self, pred, target):
        # 计算时间差分 (dI/dt)
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        
        # 计算误差
        diff_error = torch.abs(pred_diff - target_diff)
        
        # 动态加权：如果该位置是强回波（在 target 中），则赋予更高权重
        # target[:, 1:] 代表 t+1 时刻的真实强度
        weight_map = 1.0 + self.weight_scale * target[:, 1:]
        
        weighted_loss = (diff_error * weight_map).mean()
        return weighted_loss

class HybridLoss(nn.Module):
    """
    Mamba 物理感知混合损失函数
    
    设计理念：
    1. L1: 基础像素对齐 (Base)。
    2. MS-SSIM: 利用 Mamba 的全局视野，保证大尺度结构一致性 (Structure)。
    3. Soft-CSI: 直接优化竞赛指标，解决稀疏性 (Metric)。
    4. Spectral: 解决模糊，恢复高频细节 (Texture)。
    5. Evolution: 约束状态空间的演变符合物理规律 (Physics)。
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
        
        self.l1 = nn.L1Loss()
        
        # MS-SSIM (如果可用)
        if TORCHMETRICS_AVAILABLE and ssim_weight > 0:
            self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
        else:
            self.ms_ssim = None
            
        self.soft_csi = WeightedScoreSoftCSILoss()
        self.spectral = LogSpectralDistanceLoss()
        self.evolution = WeightedEvolutionLoss()

    def forward(self, logits, target, mask=None):
        """
        logits: [B, T, C, H, W] - 模型的原始输出
        target: [B, T, C, H, W] - 归一化后的真实值 [0, 1]
        """
        # 1. 预处理
        if logits.dim() == 5: logits = logits.squeeze(2)
        if target.dim() == 5: target = target.squeeze(2)
        
        # 映射到 [0, 1]
        pred = torch.sigmoid(logits)
        
        loss_dict = {}
        total_loss = 0.0
        
        # 2. L1 Loss (基础)
        l1_loss = self.l1(pred, target)
        total_loss += self.weights['l1'] * l1_loss
        loss_dict['l1'] = l1_loss.item()
        
        # 3. Soft-CSI Loss (指标优化)
        if self.weights['csi'] > 0:
            csi_loss = self.soft_csi(pred, target)
            total_loss += self.weights['csi'] * csi_loss
            loss_dict['csi'] = csi_loss.item()
            
        # 4. Spectral Loss (抗模糊)
        if self.weights['spec'] > 0:
            spec_loss = self.spectral(pred, target)
            total_loss += self.weights['spec'] * spec_loss
            loss_dict['spec'] = spec_loss.item()
            
        # 5. Evolution Loss (物理约束)
        if self.weights['evo'] > 0 and pred.shape[1] > 1:
            evo_loss = self.evolution(pred, target)
            total_loss += self.weights['evo'] * evo_loss
            loss_dict['evo'] = evo_loss.item()
            
        # 6. MS-SSIM Loss (结构一致性)
        if self.ms_ssim is not None and self.weights['ssim'] > 0:
            # SSIM 需要 Channel 维度
            pred_c = pred.view(-1, 1, pred.shape[-2], pred.shape[-1])
            target_c = target.view(-1, 1, target.shape[-2], target.shape[-1])
            ssim_val = self.ms_ssim(pred_c, target_c)
            ssim_loss = 1.0 - ssim_val
            total_loss += self.weights['ssim'] * ssim_loss
            loss_dict['ssim'] = ssim_loss.item()
            
        return total_loss