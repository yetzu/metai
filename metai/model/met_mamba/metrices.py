import torch
import torch.nn as nn
import math
from typing import Dict, Optional

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
        self.log_factor = math.log(30.0 + 1)
        
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
        """
        计算各项评分指标。
        """
        with torch.no_grad():
            return self._compute(pred_norm, target_norm, mask)

    def _compute(self, pred_norm, target_norm, mask):
        # 1. 反归一化：还原为物理量 (mm)
        if self.use_log_norm:
            pred = torch.expm1(pred_norm * self.log_factor)
            target = torch.expm1(target_norm * self.log_factor)
            
            # 安全截断，防止数值误差产生负数
            pred = torch.clamp(pred, 0.0, None)
            target = torch.clamp(target, 0.0, None)
        else:
            # Linear Space
            pred = pred_norm * self.data_max
            target = target_norm * self.data_max
        
        # ==========================================
        # [关键修复] 维度适配处理：处理 5D 输入
        # (B, T, C, H, W) -> (B, T, H, W)
        # ==========================================
        if pred.dim() == 5 and pred.shape[2] == 1:
            pred = pred.squeeze(2)
            target = target.squeeze(2)
            
        if mask is not None and mask.dim() == 5:
            if mask.shape[2] == 1:
                mask = mask.squeeze(2)

        # 2. 维度适配处理 (兼容旧逻辑)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)
            if mask is not None and mask.dim() == 3:
                mask = mask.unsqueeze(1)

        # 现在这里可以安全解包了，pred 必定是 4 维 (B, T, H, W)
        B, T, H, W = pred.shape
        device = pred.device
        
        # 3. 基础 Mask 处理
        if mask is None:
            valid_mask = torch.ones((B, T, H, W), device=device, dtype=torch.bool)
        else:
            # 确保 mask 维度与 pred 一致，如果 mask 是 (B, 1, H, W) 则扩展
            if mask.shape != pred.shape:
                if mask.dim() == 4 and mask.shape[1] == 1:
                     mask = mask.expand(-1, T, -1, -1)
            valid_mask = mask > 0.5

        # 4. 初始化容器
        scores_k_list = []   
        r_k_list = []        
        ts_matrix_list = []  
        mae_matrix_list = [] 

        # 5. 遍历时效 k 进行计算
        for k in range(T):
            p_k = pred[:, k, ...].flatten()
            t_k = target[:, k, ...].flatten()
            m_k = valid_mask[:, k, ...].flatten()
            
            # === Part A: 计算相关系数 R_k ===
            # 剔除预测值和观测值同时 < 0.1 的背景点
            min_thresh = self.thresholds[0]
            is_double_zero = (p_k < min_thresh) & (t_k < min_thresh)
            mask_corr = m_k & (~is_double_zero)
            
            if mask_corr.sum() > 0:
                p_c = p_k[mask_corr]
                t_c = t_k[mask_corr]
                p_mean, t_mean = p_c.mean(), t_c.mean()
                
                num = ((p_c - p_mean) * (t_c - t_mean)).sum()
                den = torch.sqrt(((p_c - p_mean)**2).sum() * ((t_c - t_mean)**2).sum())
                
                R_k = num / (den + 1e-6)
                R_k = torch.clamp(R_k, -1.0, 1.0)
            else:
                R_k = torch.tensor(0.0, device=device)
            
            r_k_list.append(R_k)
            term_corr = torch.sqrt(torch.exp(R_k - 1))

            # === Part B: 计算区间 TS & 区间 MAE ===
            low_b = self.thresholds.unsqueeze(1) # [L, 1]
            high_b = self.highs.unsqueeze(1)     # [L, 1]
            
            p_v = p_k[m_k]
            t_v = t_k[m_k]
            
            L = self.thresholds.shape[0]
            
            if p_v.numel() == 0:
                ts_vec = torch.zeros(L, device=device)
                mae_vec = torch.zeros(L, device=device)
            else:
                p_ex = p_v.unsqueeze(0) 
                t_ex = t_v.unsqueeze(0) 
                
                # --- 区间 TS (Interval TS) ---
                # 判定: 值必须落在 [low, high) 区间内
                is_p_in = (p_ex >= low_b) & (p_ex < high_b)
                is_t_in = (t_ex >= low_b) & (t_ex < high_b)
                
                # Hit: 预测和实况落在同一区间
                hits = (is_p_in & is_t_in).float().sum(dim=1)
                # Miss: 实况在区间，预测不在
                misses = ((~is_p_in) & is_t_in).float().sum(dim=1)
                # FA: 预测在区间，实况不在
                fas = (is_p_in & (~is_t_in)).float().sum(dim=1)
                
                ts_vec = hits / (hits + misses + fas + 1e-8)
                
                # --- 区间 MAE (Interval MAE) ---
                mae_list = []
                for i in range(L):
                    # 获取实况落在第 i 个区间的 Mask
                    mask_level = is_t_in[i, :] 
                    
                    n_b = mask_level.sum()
                    if n_b > 0:
                        err = (torch.abs(p_v - t_v) * mask_level).sum() / n_b
                    else:
                        err = torch.tensor(0.0, device=device)
                    mae_list.append(err)
                mae_vec = torch.stack(mae_list)

            ts_matrix_list.append(ts_vec)
            mae_matrix_list.append(mae_vec)
            
            # === Part C: 计算单项评分 Score_k ===
            term_mae = torch.sqrt(torch.exp(-mae_vec / 100.0))
            sum_level_metrics = (self.level_weights * ts_vec * term_mae).sum()
            
            Score_k = term_corr * sum_level_metrics
            scores_k_list.append(Score_k)

        # 6. 结果聚合
        scores_time = torch.stack(scores_k_list)
        r_time = torch.stack(r_k_list)
        ts_time_matrix = torch.stack(ts_matrix_list)
        mae_time_matrix = torch.stack(mae_matrix_list)
        
        ts_levels = ts_time_matrix.mean(dim=0)
        mae_levels = mae_time_matrix.mean(dim=0)
        
        # 7. 计算时间加权总分
        if T == len(self.time_weights_default):
            w_time = self.time_weights_default
        elif T < len(self.time_weights_default):
            w_time = self.time_weights_default[:T]
            w_time = w_time / w_time.sum()
        else:
            w_time = torch.ones(T, device=device) / T
            
        total_score = (scores_time * w_time).sum()
        
        return {
            'total_score': total_score,
            'score_time': scores_time,
            'r_time': r_time,
            'ts_time': ts_time_matrix,
            'mae_time': mae_time_matrix,
            'ts_levels': ts_levels,
            'mae_levels': mae_levels
        }


class MetricTracker:
    """
    指标累加器模块。

    用于在验证或测试循环中累积 Batch 结果并计算平均值。
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count = 0
        self.metrics_sum = {}
        
    def update(self, metrics: Dict[str, torch.Tensor]):
        """累积单个 Batch 的指标结果"""
        self.count += 1
        for k, v in metrics.items():
            val = v.detach().cpu()
            if k not in self.metrics_sum:
                self.metrics_sum[k] = torch.zeros_like(val)
            self.metrics_sum[k] += val
            
    def compute(self) -> Dict[str, object]:
        """计算所有累积 Batch 的平均指标"""
        if self.count == 0:
            return {}
        
        results = {}
        for k, v in self.metrics_sum.items():
            results[k] = v / self.count
            
        return results