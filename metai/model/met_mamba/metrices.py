# metai/model/met_mamba/metrices.py

import torch
import torch.nn as nn
import math
from typing import Dict, Optional, Any

class MetScore(nn.Module):
    """
    竞赛评分计算器模块。
    
    支持 Log Space 反归一化，确保在物理空间 (mm) 计算指标。
    """
    
    def __init__(self, use_log_norm: bool = True, data_max: float = 30.0):
        super().__init__()
        self.use_log_norm = use_log_norm
        self.data_max = data_max
        
        # Log 反变换参数: log(30.0 + 1)
        self.log_factor = math.log(self.data_max + 1)
        
        # --- 注册常量参数 (Buffer) ---
        # 1. 时效权重 (对应 6min - 120min)
        time_weights = torch.tensor([
            0.0075, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
            0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.0075, 0.005
        ])
        self.register_buffer('time_weights_default', time_weights)

        # 2. 分级阈值 (mm)
        thresholds = torch.tensor([0.1, 1.0, 2.0, 5.0, 8.0])
        self.register_buffer('thresholds', thresholds)

        # 3. 分级区间上界 (用于 Interval TS 计算)
        inf_tensor = torch.tensor([float('inf')])
        highs = torch.cat([thresholds[1:], inf_tensor])
        self.register_buffer('highs', highs)

        # 4. 分级权重
        level_weights = torch.tensor([0.1, 0.1, 0.2, 0.25, 0.35])
        self.register_buffer('level_weights', level_weights)

    def forward(self, 
                pred_norm: torch.Tensor, 
                target_norm: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """计算各项评分指标"""
        with torch.no_grad():
            return self._compute(pred_norm, target_norm, mask)

    def _compute(self, pred_norm, target_norm, mask):
        # 1. 反归一化：还原为物理量 (mm)
        if self.use_log_norm:
            pred = torch.expm1(pred_norm * self.log_factor)
            target = torch.expm1(target_norm * self.log_factor)
            pred = torch.clamp(pred, 0.0, None)
            target = torch.clamp(target, 0.0, None)
        else:
            pred = pred_norm * self.data_max
            target = target_norm * self.data_max
        
        # 维度适配处理
        if pred.dim() == 5 and pred.shape[2] == 1:
            pred = pred.squeeze(2); target = target.squeeze(2)
        if mask is not None and mask.dim() == 5 and mask.shape[2] == 1:
            mask = mask.squeeze(2)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1); target = target.unsqueeze(1)
            if mask is not None and mask.dim() == 3: mask = mask.unsqueeze(1)

        B, T, H, W = pred.shape
        device = pred.device
        
        if mask is None:
            valid_mask = torch.ones((B, T, H, W), device=device, dtype=torch.bool)
        else:
            if mask.shape != pred.shape:
                if mask.dim() == 4 and mask.shape[1] == 1:
                     mask = mask.expand(-1, T, -1, -1)
            valid_mask = mask > 0.5

        scores_k_list, r_k_list, ts_matrix_list, mae_matrix_list = [], [], [], []

        for k in range(T):
            p_k, t_k, m_k = pred[:, k, ...].flatten(), target[:, k, ...].flatten(), valid_mask[:, k, ...].flatten()
            
            # === Correlation ===
            min_thresh = self.thresholds[0]
            is_double_zero = (p_k < min_thresh) & (t_k < min_thresh)
            mask_corr = m_k & (~is_double_zero)
            
            if mask_corr.sum() > 0:
                p_c, t_c = p_k[mask_corr], t_k[mask_corr]
                p_mean, t_mean = p_c.mean(), t_c.mean()
                num = ((p_c - p_mean) * (t_c - t_mean)).sum()
                den = torch.sqrt(((p_c - p_mean)**2).sum() * ((t_c - t_mean)**2).sum())
                R_k = torch.clamp(num / (den + 1e-6), -1.0, 1.0)
            else:
                R_k = torch.tensor(0.0, device=device)
            r_k_list.append(R_k)

            # === TS & MAE ===
            p_v, t_v = p_k[m_k], t_k[m_k]
            L = self.thresholds.shape[0]
            
            if p_v.numel() == 0:
                ts_vec, mae_vec = torch.zeros(L, device=device), torch.zeros(L, device=device)
            else:
                p_ex, t_ex = p_v.unsqueeze(0), t_v.unsqueeze(0)
                low_b, high_b = self.thresholds.unsqueeze(1), self.highs.unsqueeze(1)
                
                is_p_in = (p_ex >= low_b) & (p_ex < high_b)
                is_t_in = (t_ex >= low_b) & (t_ex < high_b)
                
                hits = (is_p_in & is_t_in).float().sum(dim=1)
                misses = ((~is_p_in) & is_t_in).float().sum(dim=1)
                fas = (is_p_in & (~is_t_in)).float().sum(dim=1)
                
                ts_vec = hits / (hits + misses + fas + 1e-8)
                
                mae_list = []
                for i in range(L):
                    mask_level = is_t_in[i, :]
                    mae_list.append((torch.abs(p_v - t_v) * mask_level).sum() / mask_level.sum() if mask_level.sum() > 0 else torch.tensor(0.0, device=device))
                mae_vec = torch.stack(mae_list)

            ts_matrix_list.append(ts_vec)
            mae_matrix_list.append(mae_vec)
            
            # === Score ===
            term_corr = torch.sqrt(torch.exp(R_k - 1))
            term_mae = torch.sqrt(torch.exp(-mae_vec / 100.0))
            sum_level_metrics = (self.level_weights * ts_vec * term_mae).sum()
            scores_k_list.append(term_corr * sum_level_metrics)

        scores_time = torch.stack(scores_k_list)
        ts_time_matrix = torch.stack(ts_matrix_list)
        mae_time_matrix = torch.stack(mae_matrix_list)
        
        # 时间加权
        if T == len(self.time_weights_default): w_time = self.time_weights_default
        elif T < len(self.time_weights_default): w_time = self.time_weights_default[:T] / self.time_weights_default[:T].sum()
        else: w_time = torch.ones(T, device=device) / T
            
        return {
            'total_score': (scores_time * w_time).sum(),
            'score_time': scores_time,
            'r_time': torch.stack(r_k_list),
            'ts_time': ts_time_matrix,
            'mae_time': mae_time_matrix,
            'ts_levels': ts_time_matrix.mean(dim=0),
            'mae_levels': mae_time_matrix.mean(dim=0)
        }

class MetricTracker:
    """指标累加器"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count = 0
        self.metrics_sum = {}
        
    def update(self, metrics: Dict[str, torch.Tensor]):
        self.count += 1
        for k, v in metrics.items():
            val = v.detach().cpu()
            if k not in self.metrics_sum: self.metrics_sum[k] = torch.zeros_like(val)
            self.metrics_sum[k] += val
            
    def compute(self) -> Dict[str, Any]:
        if self.count == 0: return {}
        return {k: v / self.count for k, v in self.metrics_sum.items()}

# ==========================================
# [新增] MetMetricCollection
# ==========================================
class MetMetricCollection(nn.Module):
    """
    指标集合封装类。
    
    功能：
    1. 组合 MetScore (计算) 和 MetricTracker (累积)。
    2. 处理指标名称前缀 (如 'train_', 'val_')。
    3. 适配 LightningModule 的调用方式。
    """
    def __init__(self, prefix: str = "", use_log_norm: bool = True):
        super().__init__()
        self.prefix = prefix
        # MetScore 是 nn.Module，包含 buffer，Lightning 会自动处理设备移动
        self.scorer = MetScore(use_log_norm=use_log_norm)
        # Tracker 是普通类，用于 CPU 累积
        self.tracker = MetricTracker()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算当前 Batch 的指标并更新 Tracker。
        """
        # 1. 计算当前 Batch 指标 (GPU)
        scores = self.scorer(pred, target, mask)
        
        # 2. 更新累积器 (CPU)
        self.tracker.update(scores)
        
        # 3. 返回带前缀的结果 (用于 step log)
        return {f"{self.prefix}{k}": v for k, v in scores.items()}

    def compute(self) -> Dict[str, Any]:
        """计算累积平均值并返回带前缀的字典"""
        metrics = self.tracker.compute()
        return {f"{self.prefix}{k}": v for k, v in metrics.items()}

    def reset(self):
        """重置累积器"""
        self.tracker.reset()