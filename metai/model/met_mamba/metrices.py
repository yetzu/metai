import torch
import torch.nn as nn
from typing import Dict, Optional

class MetScore(nn.Module):
    """
    竞赛评分标准计算器模块
    
    功能描述:
    1. Score_total: 计算时间维度的加权平均总分，支持根据输入时效长度自动归一化权重。
    2. R_k (相关系数): 计算 Pearson 相关系数，自动剔除预测值与实况值均为背景( < 0.1)的格点。
    3. TS_ik (区间命中率): 采用【区间命中】(Interval TS) 逻辑，即预测值与实况值需落在同一等级区间才算命中。
    4. MAE_ik (区间绝对误差): 采用【分段区间】逻辑，仅统计实况落在该等级区间内的绝对误差。
    5. 维度支持: 自动适配序列输入 [B, T, H, W] 与单帧输入 [B, H, W]。
    6. 数据范围: 数据范围为 0-30mm。
    """
    
    def __init__(self, data_max: float = 30.0):
        """
        Args:
            data_max (float): 数据集归一化时使用的最大物理值 (mm)。默认值: 30.0。
        """
        super().__init__()
        self.data_max = data_max
        
        # --- 注册常量参数 (Buffer) ---
        
        # 1. 时效权重 (表1)
        # 对应: 6min, 12min, ..., 120min
        time_weights = torch.tensor([
            0.0075, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
            0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.0075, 0.005
        ])
        self.register_buffer('time_weights_default', time_weights)

        # 2. 分级阈值 (表2) - 区间下界
        # [0.1, 1.0, 2.0, 5.0, 8.0]
        thresholds = torch.tensor([0.1, 1.0, 2.0, 5.0, 8.0])
        self.register_buffer('thresholds', thresholds)

        # 3. 分级阈值 - 区间上界
        # [1.0, 2.0, 5.0, 8.0, inf]
        inf_tensor = torch.tensor([float('inf')])
        highs = torch.cat([thresholds[1:], inf_tensor])
        self.register_buffer('highs', highs)

        # 4. 分级权重 (表2)
        level_weights = torch.tensor([0.1, 0.1, 0.2, 0.25, 0.35])
        self.register_buffer('level_weights', level_weights)

    def forward(self, 
                pred_norm: torch.Tensor, 
                target_norm: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                override_data_max: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_norm: 归一化预测值，形状 [B, T, H, W] 或 [B, H, W]。
            target_norm: 归一化真值，形状 [B, T, H, W] 或 [B, H, W]。
            mask: 有效区域掩码 (1为有效)，形状同上或 [B, 1, H, W]。
            override_data_max: (可选) 临时覆盖初始化时的 data_max。
        """
        with torch.no_grad():
            return self._compute(pred_norm, target_norm, mask, override_data_max)

    def _compute(self, pred_norm, target_norm, mask, override_data_max):
        # 1. 维度适配处理
        if pred_norm.dim() == 3:
            pred_norm = pred_norm.unsqueeze(1)
            target_norm = target_norm.unsqueeze(1)
            if mask is not None and mask.dim() == 3:
                mask = mask.unsqueeze(1)

        # 2. 还原物理量
        scale = override_data_max if override_data_max is not None else self.data_max
        pred = pred_norm * scale
        target = target_norm * scale
        
        B, T, H, W = pred.shape
        device = pred.device
        
        # 3. 基础 Mask 处理
        if mask is None:
            valid_mask = torch.ones((B, T, H, W), device=device, dtype=torch.bool)
        else:
            if mask.dim() == 5: mask = mask.squeeze(2)
            if mask.shape[1] == 1: mask = mask.expand(-1, T, -1, -1)
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
    指标累加器模块

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