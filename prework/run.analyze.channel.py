"""
通道重要性分析脚本（基于数据相关性，与模型无关）

使用统计相关性方法分析各通道与降水的相关性
- Pearson 相关系数
- Spearman 秩相关系数
- 互信息（Mutual Information）

支持GPU加速计算
"""
import sys
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
import traceback
from sklearn.feature_selection import mutual_info_regression
from torch.utils.data import DataLoader, SubsetRandomSampler
import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.dataset.met_dataloader_scwds import ScwdsDataModule

# 通道映射
CHANNEL_MAPPING = {
    0: ("LABEL", "RA"),
    1: ("RADAR", "CR"), 2: ("RADAR", "CAP20"), 3: ("RADAR", "CAP30"),
    4: ("RADAR", "CAP40"), 5: ("RADAR", "CAP50"), 6: ("RADAR", "CAP60"),
    7: ("RADAR", "CAP70"), 8: ("RADAR", "HBR"), 9: ("RADAR", "VIL"),
    10: ("RADAR", "ET"),
    11: ("NWP", "CAPE"), 12: ("NWP", "LCL"), 13: ("NWP", "PE"),
    14: ("NWP", "PWAT"), 15: ("NWP", "Q700"), 16: ("NWP", "Q850"),
    17: ("NWP", "Q1000"), 18: ("NWP", "RH700"), 19: ("NWP", "RH1000"),
    20: ("NWP", "RH500"), 21: ("NWP", "TdSfc850"), 22: ("NWP", "WS500"),
    23: ("NWP", "WS700"), 24: ("NWP", "WS925"),
    25: ("GIS", "LAT"), 26: ("GIS", "LON"), 27: ("GIS", "DEM"),
    28: ("GIS", "MONTH"), 29: ("GIS", "HOUR"),
}


class ChannelCorrelationAnalyzer:
    """通道相关性分析器（基于数据，不依赖模型，支持GPU加速）"""
    
    def __init__(self, dataloader, precipitation_channel: int = 0, device: Optional[torch.device] = None):
        """
        Args:
            dataloader: 数据加载器
            precipitation_channel: 降水标签通道索引（默认0）
            device: 计算设备（None表示自动选择，'cuda'表示GPU，'cpu'表示CPU）
        """
        self.dataloader = dataloader
        self.precipitation_channel = precipitation_channel
        
        # 设备选择
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        if self.device.type == 'cuda':
            # 使用 self.device 的索引，而不是硬编码 0
            device_index = self.device.index if self.device.index is not None else 0
            print(f"GPU: {torch.cuda.get_device_name(device_index)}")
            print(f"GPU内存: {torch.cuda.get_device_properties(device_index).total_memory / 1024**3:.2f} GB")
        
    def collect_data(self, num_samples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        收集数据用于相关性分析（使用GPU加速）
        
        Returns:
            input_data: (N, C-1, T, H, W) 输入通道数据（排除降水通道），GPU tensor
            precipitation_data: (N, T, H, W) 降水数据，GPU tensor
        """
        input_channels = []
        precipitation_channels = []
        
        print("正在收集数据...")
        count = 0
        for batch_x, batch_y in tqdm(self.dataloader, desc="加载数据"):
            # batch_x: (B, T, C, H, W)
            # batch_y: (B, T, C, H, W) 或 (B, T, 1, H, W)
            
            # 将数据移到GPU
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 提取降水数据（从输入中提取，因为输入包含所有通道）
            # 输入数据的通道0是降水标签
            precip_input = batch_x[:, :, self.precipitation_channel, :, :]  # (B, T, H, W)
            
            # 提取目标降水（如果batch_y的通道维度是1，则直接使用；否则取通道0）
            if batch_y.shape[2] == 1:
                precip_target = batch_y[:, :, 0, :, :]  # (B, T, H, W)
            else:
                precip_target = batch_y[:, :, self.precipitation_channel, :, :]  # (B, T, H, W)
            
            # 提取输入通道（排除降水通道）
            # 输入通道索引：1 到 C-1
            input_chans = batch_x[:, :, 1:, :, :]  # (B, T, C-1, H, W)
            input_chans = input_chans.permute(0, 2, 1, 3, 4)  # (B, C-1, T, H, W)
            
            # 合并输入和目标的降水数据（用于分析输入通道与降水的相关性）
            # 使用输入时刻的降水作为参考
            precipitation_channels.append(precip_input)
            input_channels.append(input_chans)
            
            count += batch_x.shape[0]
            if num_samples is not None and count >= num_samples:
                break
        
        # 合并所有批次（在GPU上）
        input_data = torch.cat(input_channels, dim=0)  # (N, C-1, T, H, W)
        precipitation_data = torch.cat(precipitation_channels, dim=0)  # (N, T, H, W)
        
        print(f"收集完成: {input_data.shape[0]} 个样本")
        print(f"输入通道数: {input_data.shape[1]}")
        print(f"时间步数: {input_data.shape[2]}")
        print(f"空间尺寸: {input_data.shape[3]}x{input_data.shape[4]}")
        
        # 显示GPU内存使用情况
        if self.device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            print(f"GPU内存使用: {memory_allocated:.2f} GB (已分配) / {memory_reserved:.2f} GB (已保留)")
        
        return input_data, precipitation_data
    
    def _pearson_correlation_gpu(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GPU批量计算Pearson相关系数
        
        Args:
            x: (N, C) 或 (N*T*H*W, C) 通道数据
            y: (N,) 或 (N*T*H*W,) 降水数据
            
        Returns:
            correlations: (C,) Pearson相关系数
            p_values: (C,) p值（简化版本，使用t分布近似）
        """
        # 确保y是1D
        if y.dim() > 1:
            y = y.flatten()
        
        # 展平x的样本维度
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        
        # 确保x和y的长度一致
        min_len = min(x.shape[0], y.shape[0])
        x = x[:min_len]
        y = y[:min_len]
        
        # 移除NaN和无穷值
        valid_mask = torch.isfinite(x).all(dim=1) & torch.isfinite(y)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        if len(x_valid) < 10:
            return torch.zeros(x.shape[1], device=x.device), torch.ones(x.shape[1], device=x.device)
        
        # 中心化
        x_mean = x_valid.mean(dim=0, keepdim=True)
        y_mean = y_valid.mean()
        x_centered = x_valid - x_mean
        y_centered = y_valid - y_mean
        
        # 计算协方差和标准差
        numerator = (x_centered * y_centered.unsqueeze(1)).sum(dim=0)
        x_std = x_centered.norm(dim=0)
        y_std = y_centered.norm()
        
        # 避免除零
        denominator = x_std * y_std
        denominator = torch.where(denominator > 1e-8, denominator, torch.ones_like(denominator))
        
        correlations = numerator / denominator
        
        # 简化的p值计算（使用t分布近似）
        n = len(x_valid)
        t_stat = correlations * torch.sqrt((n - 2) / (1 - correlations.pow(2) + 1e-8))
        # 使用正态分布近似（对于大样本）
        p_values = 2 * (1 - torch.distributions.Normal(0, 1).cdf(t_stat.abs()))
        
        return correlations, p_values
    
    def _update_pearson_incremental(
        self, 
        stats: Dict[str, torch.Tensor],
        x_batch: torch.Tensor,
        y_batch: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        增量更新Pearson相关系数的统计量（在线算法）
        
        Args:
            stats: 包含 'sum_x', 'sum_y', 'sum_xy', 'sum_x2', 'sum_y2', 'count' 的字典
            x_batch: (N, C) 当前批次的通道数据
            y_batch: (N,) 当前批次的降水数据
            
        Returns:
            更新后的统计量字典
        """
        # 展平数据
        if y_batch.dim() > 1:
            y_batch = y_batch.flatten()
        if x_batch.dim() > 2:
            x_batch = x_batch.reshape(-1, x_batch.shape[-1])
        
        # 确保长度一致
        min_len = min(x_batch.shape[0], y_batch.shape[0])
        x_batch = x_batch[:min_len]
        y_batch = y_batch[:min_len]
        
        # 移除NaN和无穷值
        valid_mask = torch.isfinite(x_batch).all(dim=1) & torch.isfinite(y_batch)
        x_valid = x_batch[valid_mask]
        y_valid = y_batch[valid_mask]
        
        if len(x_valid) == 0:
            return stats
        
        # 初始化统计量（如果是第一次）
        if 'sum_x' not in stats:
            num_channels = x_valid.shape[1]
            stats = {
                'sum_x': torch.zeros(num_channels, device=x_valid.device),
                'sum_y': torch.zeros(1, device=x_valid.device),
                'sum_xy': torch.zeros(num_channels, device=x_valid.device),
                'sum_x2': torch.zeros(num_channels, device=x_valid.device),
                'sum_y2': torch.zeros(1, device=x_valid.device),
                'count': torch.zeros(1, device=x_valid.device, dtype=torch.long)
            }
        
        # 更新统计量
        stats['sum_x'] += x_valid.sum(dim=0)
        stats['sum_y'] += y_valid.sum()
        stats['sum_xy'] += (x_valid * y_valid.unsqueeze(1)).sum(dim=0)
        stats['sum_x2'] += (x_valid ** 2).sum(dim=0)
        stats['sum_y2'] += (y_valid ** 2).sum()
        stats['count'] += len(x_valid)
        
        return stats
    
    def _compute_pearson_from_stats(self, stats: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从统计量计算Pearson相关系数
        
        Args:
            stats: 包含统计量的字典
            
        Returns:
            correlations: (C,) Pearson相关系数
            p_values: (C,) p值
        """
        n = stats['count'].item()
        if n < 10:
            num_channels = stats['sum_x'].shape[0]
            return torch.zeros(num_channels, device=stats['sum_x'].device), \
                   torch.ones(num_channels, device=stats['sum_x'].device)
        
        # 计算均值
        x_mean = stats['sum_x'] / n
        y_mean = stats['sum_y'] / n
        
        # 计算协方差和方差
        cov_xy = stats['sum_xy'] / n - x_mean * y_mean
        var_x = stats['sum_x2'] / n - x_mean ** 2
        var_y = stats['sum_y2'] / n - y_mean ** 2
        
        # 计算相关系数
        denominator = torch.sqrt(var_x * var_y)
        denominator = torch.where(denominator > 1e-8, denominator, torch.ones_like(denominator))
        correlations = cov_xy / denominator
        
        # 计算p值
        t_stat = correlations * torch.sqrt((n - 2) / (1 - correlations.pow(2) + 1e-8))
        p_values = 2 * (1 - torch.distributions.Normal(0, 1).cdf(t_stat.abs()))
        
        return correlations, p_values
    
    def _spearman_correlation_gpu(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GPU批量计算Spearman秩相关系数
        
        Args:
            x: (N, C) 或 (N*T*H*W, C) 通道数据
            y: (N,) 或 (N*T*H*W,) 降水数据
            
        Returns:
            correlations: (C,) Spearman相关系数
            p_values: (C,) p值
        """
        # 确保y是1D
        if y.dim() > 1:
            y = y.flatten()
        
        # 展平x的样本维度
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        
        # 确保x和y的长度一致
        min_len = min(x.shape[0], y.shape[0])
        x = x[:min_len]
        y = y[:min_len]
        
        # 移除NaN和无穷值
        valid_mask = torch.isfinite(x).all(dim=1) & torch.isfinite(y)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        if len(x_valid) < 10:
            return torch.zeros(x.shape[1], device=x.device), torch.ones(x.shape[1], device=x.device)
        
        # 计算秩（批量处理）
        y_ranks = torch.argsort(torch.argsort(y_valid))
        
        # 对每个通道计算秩（批量处理，优化：使用torch.argsort批量计算）
        # 使用torch.argsort的argsort来批量计算所有通道的秩
        x_ranks = torch.argsort(torch.argsort(x_valid, dim=0), dim=0).float()
        
        # 使用Pearson公式计算秩相关系数
        x_ranks_mean = x_ranks.mean(dim=0, keepdim=True)
        y_ranks_mean = y_ranks.mean()
        x_ranks_centered = x_ranks - x_ranks_mean
        y_ranks_centered = y_ranks - y_ranks_mean
        
        numerator = (x_ranks_centered * y_ranks_centered.unsqueeze(1)).sum(dim=0)
        x_ranks_std = x_ranks_centered.norm(dim=0)
        y_ranks_std = y_ranks_centered.norm()
        
        denominator = x_ranks_std * y_ranks_std
        denominator = torch.where(denominator > 1e-8, denominator, torch.ones_like(denominator))
        
        correlations = numerator / denominator
        
        # 简化的p值计算
        n = len(x_valid)
        t_stat = correlations * torch.sqrt((n - 2) / (1 - correlations.pow(2) + 1e-8))
        p_values = 2 * (1 - torch.distributions.Normal(0, 1).cdf(t_stat.abs()))
        
        return correlations, p_values
    
    @staticmethod
    def _compute_mi_single_channel(args):
        """计算单个通道的互信息（用于并行计算）"""
        channel_idx, channel_data, precip_data = args
        try:
            valid_mask = np.isfinite(channel_data) & np.isfinite(precip_data)
            channel_valid = channel_data[valid_mask]
            precip_valid = precip_data[valid_mask]
            
            if len(channel_valid) >= 10:
                mi = mutual_info_regression(
                    channel_valid.reshape(-1, 1),
                    precip_valid,
                    random_state=42,
                    n_neighbors=3
                )[0]
                return channel_idx, float(mi)
            else:
                return channel_idx, 0.0
        except Exception as e:
            print(f"  警告: 通道 {channel_idx+1} 互信息计算失败: {e}")
            return channel_idx, 0.0
    
    def _compute_mutual_info_parallel(
        self, 
        input_data: np.ndarray, 
        precip_data: np.ndarray, 
        num_channels: int,
        max_workers: Optional[int] = None
    ) -> Dict[int, float]:
        """
        并行计算所有通道的互信息
        
        Args:
            input_data: (N, C) 输入通道数据
            precip_data: (N,) 降水数据
            num_channels: 通道数量
            max_workers: 最大工作进程数（None=自动选择）
            
        Returns:
            Dict[channel_idx, mutual_info_value]
        """
        if max_workers is None:
            # 使用CPU核心数，但不超过通道数
            max_workers = min(mp.cpu_count(), num_channels, 8)  # 最多8个进程
        
        # 准备任务参数
        tasks = []
        for channel_idx in range(num_channels):
            channel_data = input_data[:, channel_idx]
            tasks.append((channel_idx, channel_data, precip_data))
        
        # 并行计算
        results_dict = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._compute_mi_single_channel, task): task[0] 
                      for task in tasks}
            
            completed = 0
            for future in as_completed(futures):
                channel_idx, mi_value = future.result()
                results_dict[channel_idx] = mi_value
                completed += 1
                if completed % 5 == 0:
                    print(f"  已完成 {completed}/{num_channels} 个通道的互信息计算")
        
        return results_dict
    
    def compute_correlation_streaming(
        self,
        num_samples: Optional[int] = None,
        method: str = 'pearson',
        batch_size_for_spearman: int = 10000
    ) -> Dict[int, Dict]:
        """
        流式计算相关性（直接从dataloader读取，支持大样本量）
        
        Args:
            num_samples: 用于分析的样本数量（None表示使用全部）
            method: 相关性计算方法 ('pearson', 'spearman', 'mutual_info', 'all')
            batch_size_for_spearman: Spearman计算时的批次大小（用于内存优化）
            
        Returns:
            Dict[channel_idx, {pearson, spearman, mi, ...}]
        """
        print(f"\n开始流式计算通道相关性...")
        print(f"方法: {method}")
        print(f"使用设备: {self.device}")
        print(f"最大样本数: {num_samples if num_samples else '全部'}\n")
        
        # 初始化统计量
        pearson_stats = {}
        spearman_data_list = []  # 对于Spearman，需要收集数据计算秩
        mi_data_list = []  # 对于互信息，需要收集数据
        
        total_samples = 0
        num_channels = None
        
        # 流式处理数据
        print("流式处理数据...")
        for batch_idx, (batch_x, batch_y) in enumerate(tqdm(self.dataloader, desc="处理批次")):
            # 将数据移到GPU
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 提取降水数据
            precip_input = batch_x[:, :, self.precipitation_channel, :, :]  # (B, T, H, W)
            
            # 提取输入通道（排除降水通道）
            input_chans = batch_x[:, :, 1:, :, :]  # (B, T, C-1, H, W)
            input_chans = input_chans.permute(0, 2, 1, 3, 4)  # (B, C-1, T, H, W)
            
            # 确定通道数
            if num_channels is None:
                num_channels = input_chans.shape[1]
            
            # 展平空间和时间维度
            B, C, T, H, W = input_chans.shape
            input_flat = input_chans.reshape(B, C, T * H * W)  # (B, C-1, T*H*W)
            precip_flat = precip_input.reshape(B, T * H * W)  # (B, T*H*W)
            
            # 进一步展平为 (B*T*H*W, C-1) 和 (B*T*H*W,)
            input_flat_all = input_flat.permute(1, 0, 2).reshape(C, -1).T  # (B*T*H*W, C-1)
            precip_flat_all = precip_flat.flatten()  # (B*T*H*W,)
            
            # 增量更新Pearson统计量
            if method in ['pearson', 'all']:
                pearson_stats = self._update_pearson_incremental(
                    pearson_stats, input_flat_all, precip_flat_all
                )
            
            # 收集Spearman数据（分批处理以减少内存）
            if method in ['spearman', 'all']:
                # 将数据移到CPU以节省GPU内存
                input_cpu = input_flat_all.cpu()
                precip_cpu = precip_flat_all.cpu()
                spearman_data_list.append((input_cpu, precip_cpu))
            
            # 收集互信息数据（CPU）
            if method in ['mutual_info', 'all']:
                input_cpu = input_flat_all.cpu().numpy()
                precip_cpu = precip_flat_all.cpu().numpy()
                mi_data_list.append((input_cpu, precip_cpu))
            
            total_samples += batch_x.shape[0]
            
            # 显示进度
            if (batch_idx + 1) % 10 == 0:
                if self.device.type == 'cuda':
                    memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                    print(f"  已处理 {total_samples} 个样本, GPU内存: {memory_allocated:.2f} GB")
            
            if num_samples is not None and total_samples >= num_samples:
                break
            
            # 释放GPU内存
            del batch_x, batch_y, input_chans, precip_input, input_flat, precip_flat
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        print(f"\n数据处理完成，共 {total_samples} 个样本")
        
        # 计算Pearson相关系数
        pearson_corrs = None
        pearson_ps = None
        if method in ['pearson', 'all']:
            print("计算Pearson相关系数...")
            pearson_corrs, pearson_ps = self._compute_pearson_from_stats(pearson_stats)
            pearson_corrs = pearson_corrs.cpu().numpy()
            pearson_ps = pearson_ps.cpu().numpy()
        
        # 计算Spearman相关系数（合并所有数据）
        spearman_corrs = None
        spearman_ps = None
        if method in ['spearman', 'all']:
            print("计算Spearman秩相关系数...")
            if len(spearman_data_list) > 0:
                # 合并所有数据
                input_all = torch.cat([x for x, _ in spearman_data_list], dim=0)
                precip_all = torch.cat([y for _, y in spearman_data_list], dim=0)
                spearman_corrs, spearman_ps = self._spearman_correlation_gpu(input_all, precip_all)
                spearman_corrs = spearman_corrs.cpu().numpy()
                spearman_ps = spearman_ps.cpu().numpy()
                # 释放内存
                del input_all, precip_all
                spearman_data_list.clear()
        
        # 计算互信息（合并所有数据，并行计算）
        mi_results_dict = {}
        if method in ['mutual_info', 'all']:
            print("合并数据用于互信息计算...")
            if len(mi_data_list) > 0:
                input_cpu_all = np.concatenate([x for x, _ in mi_data_list], axis=0)
                precip_cpu_all = np.concatenate([y for _, y in mi_data_list], axis=0)
                mi_data_list.clear()
                
                # 并行计算互信息
                print("并行计算互信息（使用多进程）...")
                if num_channels is None:
                    raise ValueError("无法确定通道数量，请检查数据")
                assert num_channels is not None  # 类型检查
                mi_results_dict = self._compute_mutual_info_parallel(
                    input_cpu_all, precip_cpu_all, num_channels
                )
        
        # 整理结果
        print("\n整理结果...")
        results = {}
        valid_samples = pearson_stats.get('count', torch.tensor(0)).item() if pearson_stats else 0
        
        if num_channels is None:
            raise ValueError("无法确定通道数量，请检查数据")
        
        for channel_idx in range(num_channels):
            actual_channel_idx = channel_idx + 1
            category, name = CHANNEL_MAPPING[actual_channel_idx]
            
            result = {
                'category': category,
                'name': name,
                'valid_samples': valid_samples
            }
            
            # Pearson相关系数
            if method in ['pearson', 'all'] and pearson_corrs is not None and pearson_ps is not None:
                result['pearson'] = float(pearson_corrs[channel_idx])
                result['pearson_p'] = float(pearson_ps[channel_idx])
            
            # Spearman秩相关系数
            if method in ['spearman', 'all'] and spearman_corrs is not None and spearman_ps is not None:
                result['spearman'] = float(spearman_corrs[channel_idx])
                result['spearman_p'] = float(spearman_ps[channel_idx])
            
            # 互信息（从并行计算结果中获取）
            if method in ['mutual_info', 'all']:
                result['mutual_info'] = mi_results_dict.get(channel_idx, 0.0)
            
            # 使用绝对值作为重要性指标
            if 'pearson' in result:
                result['importance'] = abs(result['pearson'])
            elif 'spearman' in result:
                result['importance'] = abs(result['spearman'])
            elif 'mutual_info' in result:
                result['importance'] = result['mutual_info']
            else:
                result['importance'] = 0.0
            
            results[actual_channel_idx] = result
            
            # 打印结果
            info_parts = []
            if 'pearson' in result:
                info_parts.append(f"Pearson: {result['pearson']:.4f}")
            if 'spearman' in result:
                info_parts.append(f"Spearman: {result['spearman']:.4f}")
            if 'mutual_info' in result:
                info_parts.append(f"MI: {result['mutual_info']:.4f}")
            print(f"[{channel_idx+1}/{num_channels}] {category}-{name}: {', '.join(info_parts)}")
        
        return results
    
    def compute_correlation(
        self,
        input_data: torch.Tensor,
        precipitation_data: torch.Tensor,
        method: str = 'pearson'
    ) -> Dict[int, Dict]:
        """
        计算每个输入通道与降水的相关性（GPU加速批量计算）
        
        Args:
            input_data: (N, C-1, T, H, W) 输入通道数据（GPU tensor）
            precipitation_data: (N, T, H, W) 降水数据（GPU tensor）
            method: 相关性计算方法 ('pearson', 'spearman', 'mutual_info', 'all')
            
        Returns:
            Dict[channel_idx, {pearson, spearman, mi, ...}]
        """
        num_channels = input_data.shape[1]
        results = {}
        
        print(f"\n开始计算 {num_channels} 个通道的相关性...")
        print(f"方法: {method}")
        print(f"使用设备: {self.device}\n")
        
        # 展平空间和时间维度
        # input_data: (N, C-1, T, H, W) -> (N, C-1, T*H*W)
        # precipitation_data: (N, T, H, W) -> (N, T*H*W)
        N, C, T, H, W = input_data.shape
        input_flat = input_data.reshape(N, C, T * H * W)  # (N, C-1, T*H*W)
        precip_flat = precipitation_data.reshape(N, T * H * W)  # (N, T*H*W)
        
        # 进一步展平为 (N*T*H*W, C-1) 和 (N*T*H*W,)
        input_flat_all = input_flat.permute(1, 0, 2).reshape(C, -1).T  # (N*T*H*W, C-1)
        precip_flat_all = precip_flat.flatten()  # (N*T*H*W,)
        
        # 初始化结果变量
        pearson_corrs = None
        pearson_ps = None
        spearman_corrs = None
        spearman_ps = None
        
        # 批量计算Pearson和Spearman（GPU加速）
        if method in ['pearson', 'all']:
            print("批量计算Pearson相关系数...")
            pearson_corrs, pearson_ps = self._pearson_correlation_gpu(input_flat_all, precip_flat_all)
            pearson_corrs = pearson_corrs.cpu().numpy()
            pearson_ps = pearson_ps.cpu().numpy()
        
        if method in ['spearman', 'all']:
            print("批量计算Spearman秩相关系数...")
            spearman_corrs, spearman_ps = self._spearman_correlation_gpu(input_flat_all, precip_flat_all)
            spearman_corrs = spearman_corrs.cpu().numpy()
            spearman_ps = spearman_ps.cpu().numpy()
        
        # 互信息需要逐个通道计算（CPU，因为sklearn没有GPU版本）
        mi_results_dict = {}
        if method in ['mutual_info', 'all']:
            print("准备数据用于互信息计算（并行）...")
            # 将数据移到CPU用于互信息计算
            input_cpu_all = input_flat_all.cpu().numpy()
            precip_cpu_all = precip_flat_all.cpu().numpy()
            
            # 并行计算互信息
            print("并行计算互信息（使用多进程）...")
            mi_results_dict = self._compute_mutual_info_parallel(
                input_cpu_all, precip_cpu_all, num_channels
            )
        
        # 整理结果
        print("\n整理结果...")
        for channel_idx in range(num_channels):
            actual_channel_idx = channel_idx + 1  # 因为输入通道从1开始（0是降水）
            category, name = CHANNEL_MAPPING[actual_channel_idx]
            
            result = {
                'category': category,
                'name': name,
                'valid_samples': len(precip_flat_all)
            }
            
            # Pearson相关系数
            if method in ['pearson', 'all'] and pearson_corrs is not None and pearson_ps is not None:
                result['pearson'] = float(pearson_corrs[channel_idx])
                result['pearson_p'] = float(pearson_ps[channel_idx])
            
            # Spearman秩相关系数
            if method in ['spearman', 'all'] and spearman_corrs is not None and spearman_ps is not None:
                result['spearman'] = float(spearman_corrs[channel_idx])
                result['spearman_p'] = float(spearman_ps[channel_idx])
            
            # 互信息（从并行计算结果中获取）
            if method in ['mutual_info', 'all']:
                result['mutual_info'] = mi_results_dict.get(channel_idx, 0.0)
            
            # 使用绝对值作为重要性指标
            if 'pearson' in result:
                result['importance'] = abs(result['pearson'])
            elif 'spearman' in result:
                result['importance'] = abs(result['spearman'])
            elif 'mutual_info' in result:
                result['importance'] = result['mutual_info']
            else:
                result['importance'] = 0.0
            
            results[actual_channel_idx] = result
            
            # 打印结果
            info_parts = []
            if 'pearson' in result:
                info_parts.append(f"Pearson: {result['pearson']:.4f}")
            if 'spearman' in result:
                info_parts.append(f"Spearman: {result['spearman']:.4f}")
            if 'mutual_info' in result:
                info_parts.append(f"MI: {result['mutual_info']:.4f}")
            print(f"[{channel_idx+1}/{num_channels}] {category}-{name}: {', '.join(info_parts)}")
        
        return results
    
    def visualize_results(self, results: Dict[int, Dict], save_path: str, method: str = 'pearson'):
        """可视化结果"""
        # 选择用于排序的指标
        if method == 'pearson':
            sort_key = 'pearson'
            title_suffix = 'Pearson 相关系数'
        elif method == 'spearman':
            sort_key = 'spearman'
            title_suffix = 'Spearman 秩相关系数'
        elif method == 'mutual_info':
            sort_key = 'mutual_info'
            title_suffix = '互信息'
        else:
            sort_key = 'importance'
            title_suffix = '重要性'
        
        # 按相关性排序
        sorted_results = sorted(
            results.items(),
            key=lambda x: abs(x[1].get(sort_key, 0)),
            reverse=True
        )
        
        channels = [f"{v['category']}-{v['name']}" for _, v in sorted_results]
        
        # 提取相关性值
        if method == 'pearson':
            correlations = [v.get('pearson', 0) for _, v in sorted_results]
        elif method == 'spearman':
            correlations = [v.get('spearman', 0) for _, v in sorted_results]
        elif method == 'mutual_info':
            correlations = [v.get('mutual_info', 0) for _, v in sorted_results]
        else:
            correlations = [v.get('importance', 0) for _, v in sorted_results]
        
        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        
        # 1. 条形图：所有通道的相关性
        ax1 = axes[0]
        y_pos = np.arange(len(channels))
        colors = []
        for _, v in sorted_results:
            if v['category'] == 'RADAR':
                colors.append('#FF6B6B')
            elif v['category'] == 'NWP':
                colors.append('#4ECDC4')
            elif v['category'] == 'LABEL':
                colors.append('#FFE66D')
            else:  # GIS
                colors.append('#95E1D3')
        
        bars = ax1.barh(y_pos, correlations, color=colors, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(channels, fontsize=8)
        ax1.set_xlabel(f'{title_suffix}', fontsize=12)
        ax1.set_title(f'通道与降水的相关性分析 ({title_suffix})', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
        
        # 添加数值标签
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            if abs(corr) > 0.01:  # 只显示显著的值
                ax1.text(corr + 0.01 if corr >= 0 else corr - 0.01, i, 
                        f'{corr:.3f}', va='center', fontsize=7)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FFE66D', label='LABEL'),
            Patch(facecolor='#FF6B6B', label='RADAR'),
            Patch(facecolor='#4ECDC4', label='NWP'),
            Patch(facecolor='#95E1D3', label='GIS'),
        ]
        ax1.legend(handles=legend_elements, loc='lower right')
        
        # 2. 按类别分组的相关性
        ax2 = axes[1]
        category_correlations = {}
        for _, v in sorted_results:
            cat = v['category']
            if cat not in category_correlations:
                category_correlations[cat] = []
            if method == 'pearson':
                category_correlations[cat].append(abs(v.get('pearson', 0)))
            elif method == 'spearman':
                category_correlations[cat].append(abs(v.get('spearman', 0)))
            elif method == 'mutual_info':
                category_correlations[cat].append(v.get('mutual_info', 0))
            else:
                category_correlations[cat].append(v.get('importance', 0))
        
        categories = list(category_correlations.keys())
        category_means = [np.mean(category_correlations[cat]) for cat in categories]
        category_stds = [np.std(category_correlations[cat]) for cat in categories]
        
        ax2.bar(categories, category_means, yerr=category_stds,
                color=['#FFE66D', '#FF6B6B', '#4ECDC4', '#95E1D3'],
                alpha=0.7, capsize=5)
        ax2.set_ylabel(f'平均{title_suffix}', fontsize=12)
        ax2.set_title('按类别分组的平均相关性', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n可视化结果已保存到: {save_path}")
        plt.close()


def _process_single_gpu(args_tuple):
    """处理单个GPU的数据（用于多进程并行）"""
    (gpu_id, gpu_indices, full_dataset, batch_size, data_module_collate_fn, 
     method, use_streaming, num_samples) = args_tuple
    
    # 设置当前进程使用的GPU（必须在导入torch之前设置）
    # 在spawn模式下，子进程是全新的Python解释器，所以torch还没有被导入
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 导入torch（必须在设置CUDA_VISIBLE_DEVICES之后）
    import torch
    from torch.utils.data import DataLoader, SubsetRandomSampler
    
    # 验证环境变量设置
    actual_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    print(f"[进程 {os.getpid()}] 物理GPU {gpu_id} -> CUDA_VISIBLE_DEVICES={actual_cuda_visible}, 可见GPU数量={torch.cuda.device_count()}")
    
    device = torch.device(f'cuda:0')  # 因为CUDA_VISIBLE_DEVICES已设置，所以是cuda:0
    
    # 创建该GPU的数据加载器
    # 注意：在多进程环境下（ProcessPoolExecutor的子进程），必须设置num_workers=0
    # 避免嵌套多进程导致的死锁问题
    gpu_sampler = SubsetRandomSampler(gpu_indices)
    gpu_data_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=gpu_sampler,
        num_workers=0,  # 在多进程环境下必须为0，避免嵌套多进程死锁
        pin_memory=False,  # 在num_workers=0时，pin_memory应该为False
        persistent_workers=False,  # num_workers=0时，persistent_workers无效
        collate_fn=data_module_collate_fn,
    )
    
    # 创建分析器
    analyzer = ChannelCorrelationAnalyzer(
        gpu_data_loader, 
        precipitation_channel=0,
        device=device
    )
    
    # 计算该GPU的结果
    if use_streaming:
        gpu_results = analyzer.compute_correlation_streaming(
            num_samples=len(gpu_indices),
            method=method
        )
    else:
        input_data, precipitation_data = analyzer.collect_data(num_samples=len(gpu_indices))
        gpu_results = analyzer.compute_correlation(
            input_data,
            precipitation_data,
            method=method
        )
    
    return gpu_results


def _merge_multi_gpu_results(results_list: List[Dict[int, Dict]], method: str) -> Dict[int, Dict]:
    """
    合并多个GPU的结果
    
    对于Pearson和Spearman，需要合并统计量后重新计算
    对于互信息，可以简单平均
    """
    if len(results_list) == 0:
        return {}
    
    if len(results_list) == 1:
        return results_list[0]
    
    # 获取所有通道索引
    all_channels = set()
    for results in results_list:
        all_channels.update(results.keys())
    
    merged_results = {}
    
    for channel_idx in all_channels:
        # 获取该通道在所有GPU上的结果
        channel_results = []
        valid_samples_total = 0
        
        for results in results_list:
            if channel_idx in results:
                channel_results.append(results[channel_idx])
                valid_samples_total += results[channel_idx].get('valid_samples', 0)
        
        if len(channel_results) == 0:
            continue
        
        # 合并结果
        merged = {
            'category': channel_results[0]['category'],
            'name': channel_results[0]['name'],
            'valid_samples': valid_samples_total
        }
        
        # 对于Pearson和Spearman，需要加权平均（基于样本数）
        if method in ['pearson', 'all']:
            pearson_values = []
            pearson_ps = []
            weights = []
            for r in channel_results:
                if 'pearson' in r:
                    pearson_values.append(r['pearson'])
                    pearson_ps.append(r.get('pearson_p', 1.0))
                    weights.append(r.get('valid_samples', 1))
            
            if len(pearson_values) > 0:
                # 加权平均
                weights = np.array(weights)
                weights = weights / weights.sum()
                merged['pearson'] = float(np.average(pearson_values, weights=weights))
                merged['pearson_p'] = float(np.average(pearson_ps, weights=weights))
        
        if method in ['spearman', 'all']:
            spearman_values = []
            spearman_ps = []
            weights = []
            for r in channel_results:
                if 'spearman' in r:
                    spearman_values.append(r['spearman'])
                    spearman_ps.append(r.get('spearman_p', 1.0))
                    weights.append(r.get('valid_samples', 1))
            
            if len(spearman_values) > 0:
                weights = np.array(weights)
                weights = weights / weights.sum()
                merged['spearman'] = float(np.average(spearman_values, weights=weights))
                merged['spearman_p'] = float(np.average(spearman_ps, weights=weights))
        
        # 对于互信息，简单平均
        if method in ['mutual_info', 'all']:
            mi_values = []
            for r in channel_results:
                if 'mutual_info' in r:
                    mi_values.append(r['mutual_info'])
            
            if len(mi_values) > 0:
                merged['mutual_info'] = float(np.mean(mi_values))
        
        # 计算重要性
        if 'pearson' in merged:
            merged['importance'] = abs(merged['pearson'])
        elif 'spearman' in merged:
            merged['importance'] = abs(merged['spearman'])
        elif 'mutual_info' in merged:
            merged['importance'] = merged['mutual_info']
        else:
            merged['importance'] = 0.0
        
        merged_results[channel_idx] = merged
    
    return merged_results


def recommend_channels(results: Dict[int, Dict], top_k: int = 10, method: str = 'pearson') -> List[int]:
    """推荐相关性最高的top_k个通道"""
    # 选择排序指标
    if method == 'pearson':
        sort_key = 'pearson'
    elif method == 'spearman':
        sort_key = 'spearman'
    elif method == 'mutual_info':
        sort_key = 'mutual_info'
    else:
        sort_key = 'importance'
    
    sorted_results = sorted(
        results.items(),
        key=lambda x: abs(x[1].get(sort_key, 0)),
        reverse=True
    )
    
    recommended = [idx for idx, _ in sorted_results[:top_k]]
    return recommended


def main():
    parser = argparse.ArgumentParser(description='通道相关性分析（基于数据，不依赖模型）')
    parser.add_argument('--data_path', type=str, default='data/samples.jsonl',
                        help='数据路径')
    parser.add_argument('--in_shape', type=int, nargs=4, default=[20, 29, 128, 128],
                        help='输入形状: T C H W')
    parser.add_argument('--is_debug', type=lambda x: x.lower() == 'true', default=True,
                        help='调试模式')
    parser.add_argument('--task_mode', type=str, default='precipitation',
                        help='任务模式')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='用于分析的样本数量（None表示使用全部）')
    parser.add_argument('--method', type=str, default='all',
                        choices=['pearson', 'spearman', 'mutual_info', 'all'],
                        help='相关性计算方法')
    parser.add_argument('--output_dir', type=str, default='output/channel_analysis',
                        help='输出目录')
    parser.add_argument('--device', type=str, default=None,
                        help='计算设备 (None=自动选择, cuda=GPU, cpu=CPU)')
    parser.add_argument('--use_streaming', type=lambda x: x.lower() == 'true', default=None,
                        help='使用流式处理（None=自动选择：样本数>500时使用流式）')
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='使用的GPU数量（None=自动检测所有可用GPU）')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小（None=根据GPU数量自动调整）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检测GPU数量
    if torch.cuda.is_available():
        num_available_gpus = torch.cuda.device_count()
        if args.num_gpus is None:
            num_gpus = num_available_gpus
        else:
            num_gpus = min(args.num_gpus, num_available_gpus)
        print(f"检测到 {num_available_gpus} 张GPU，将使用 {num_gpus} 张GPU")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        num_gpus = 0
        print("未检测到GPU，将使用CPU")
    
    # 根据GPU数量调整batch_size
    if args.batch_size is None:
        if num_gpus > 0:
            # 每张GPU使用更大的batch_size以充分利用资源
            batch_size = 16 * num_gpus  # 每张GPU 16个样本
        else:
            batch_size = 4
    else:
        batch_size = args.batch_size
    
    print(f"使用批次大小: {batch_size}")
    
    # 加载数据
    print("="*60)
    print("步骤 1/2: 加载数据")
    print("="*60)
    data_module = ScwdsDataModule(
        data_path=args.data_path,
        resize_shape=(args.in_shape[2], args.in_shape[3]),
        batch_size=batch_size,
        num_workers=4,
        shuffle_train=True,
        is_debug=args.is_debug,
        task_mode=args.task_mode
    )
    
    # 从总样本中选取数据（而不是仅使用验证集）
    data_module.setup('fit')  # 初始化数据集
    full_dataset = data_module.dataset  # 获取完整数据集
    
    # 如果需要限制样本数，使用随机采样器
    if args.num_samples is not None and args.num_samples < len(full_dataset):
        # 随机选择指定数量的样本
        indices = torch.randperm(len(full_dataset))[:args.num_samples].tolist()
        sampler = SubsetRandomSampler(indices)
        shuffle = False  # 使用sampler时不能shuffle
    else:
        sampler = None
        shuffle = True  # 从总样本中随机采样
    
    # 创建包含所有样本的DataLoader
    data_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=4,
        pin_memory=True if num_gpus > 0 else False,
        persistent_workers=True if 4 > 0 else False,
        collate_fn=data_module._collate_fn,
    )
    
    total_samples = args.num_samples if args.num_samples is not None else len(full_dataset)
    print(f"从总样本中选取 {total_samples} 个样本进行分析（总样本数: {len(full_dataset)}）")
    
    # 如果有多张GPU，使用多GPU并行处理（真正并行）
    if num_gpus > 1:
        print(f"\n使用 {num_gpus} 张GPU并行处理（多进程）...")
        # 将数据分割到不同的GPU
        results_list = []
        
        # 为每个GPU创建数据子集（随机分配）
        dataset_size = len(full_dataset)
        
        # 如果指定了num_samples，随机选择样本索引
        if args.num_samples is not None and args.num_samples < dataset_size:
            all_indices = torch.randperm(dataset_size)[:args.num_samples].tolist()
        else:
            all_indices = list(range(dataset_size))
        
        # 将索引随机分配到各个GPU
        np.random.shuffle(all_indices)
        samples_per_gpu = len(all_indices) // num_gpus
        
        # 准备多进程参数
        gpu_tasks = []
        for gpu_id in range(num_gpus):
            start_idx = gpu_id * samples_per_gpu
            if gpu_id == num_gpus - 1:
                # 最后一个GPU处理剩余的所有数据
                gpu_indices = all_indices[start_idx:]
            else:
                gpu_indices = all_indices[start_idx:start_idx + samples_per_gpu]
            
            if len(gpu_indices) == 0:
                continue
            
            use_streaming = args.use_streaming
            if use_streaming is None:
                # 强制启用流式处理以避免OOM（即使样本数不多，数据量大时也会OOM）
                use_streaming = True
            
            gpu_tasks.append((
                gpu_id, gpu_indices, full_dataset, batch_size, 
                data_module._collate_fn, args.method, use_streaming, 
                len(gpu_indices)
            ))
        
        # 使用多进程并行处理（每个GPU一个进程）
        # 注意：如果dataset无法序列化，会回退到串行处理
        print(f"启动 {len(gpu_tasks)} 个并行进程...")
        try:
            with ProcessPoolExecutor(max_workers=len(gpu_tasks)) as executor:
                futures = {executor.submit(_process_single_gpu, task): i 
                          for i, task in enumerate(gpu_tasks)}
                
                for future in as_completed(futures):
                    gpu_id = futures[future]
                    try:
                        gpu_results = future.result()
                        results_list.append(gpu_results)
                        print(f"GPU {gpu_id} 处理完成")
                    except Exception as e:
                        print(f"GPU {gpu_id} 处理失败: {e}")
                        traceback.print_exc()
        except (pickle.PicklingError, AttributeError) as e:
            # 如果dataset无法序列化，回退到串行处理
            print(f"警告: 多进程并行失败（{e}），回退到串行处理...")
            results_list = []
            for gpu_id, gpu_indices, _, batch_size, collate_fn, method, use_streaming, _ in gpu_tasks:
                print(f"\n处理 GPU {gpu_id} 的数据...")
                gpu_sampler = SubsetRandomSampler(gpu_indices)
                gpu_data_loader = DataLoader(
                    full_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    sampler=gpu_sampler,
                    num_workers=0,  # 串行模式下也可以使用0，避免多进程问题
                    pin_memory=True,
                    persistent_workers=False,
                    collate_fn=collate_fn,
                )
                
                device = torch.device(f'cuda:{gpu_id}')
                analyzer = ChannelCorrelationAnalyzer(
                    gpu_data_loader, 
                    precipitation_channel=0,
                    device=device
                )
                
                if use_streaming:
                    gpu_results = analyzer.compute_correlation_streaming(
                        num_samples=len(gpu_indices),
                        method=method
                    )
                else:
                    input_data, precipitation_data = analyzer.collect_data(num_samples=len(gpu_indices))
                    gpu_results = analyzer.compute_correlation(
                        input_data,
                        precipitation_data,
                        method=method
                    )
                
                results_list.append(gpu_results)
        
        # 合并所有GPU的结果（对统计量进行加权平均）
        print("\n合并所有GPU的结果...")
        results = _merge_multi_gpu_results(results_list, args.method)
        
    else:
        # 单GPU或CPU处理
        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device('cuda:0' if num_gpus > 0 else 'cpu')
        analyzer = ChannelCorrelationAnalyzer(
            data_loader, 
            precipitation_channel=0,
            device=device
        )
        
        # 决定使用流式处理还是批量处理
        use_streaming = args.use_streaming
        if use_streaming is None:
            # 强制启用流式处理以避免OOM（即使样本数不多，数据量大时也会OOM）
            use_streaming = True
        
        if use_streaming:
            print("\n使用流式处理模式（支持大样本量）")
            print("="*60)
            print("步骤 2/2: 流式计算通道相关性")
            print("="*60)
            results = analyzer.compute_correlation_streaming(
                num_samples=args.num_samples,
                method=args.method
            )
        else:
            print("\n使用批量处理模式")
            # 收集数据
            input_data, precipitation_data = analyzer.collect_data(num_samples=args.num_samples)
            
            # 计算相关性
            print("\n" + "="*60)
            print("步骤 2/2: 计算通道相关性")
            print("="*60)
            results = analyzer.compute_correlation(
                input_data,
                precipitation_data,
                method=args.method
            )
    
    # 保存结果
    results_file = os.path.join(args.output_dir, 'channel_correlation.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {results_file}")
    
    # 可视化（如果有多种方法，为每种方法生成图表）
    if args.method == 'all':
        methods = ['pearson', 'spearman', 'mutual_info']
    else:
        methods = [args.method]
    
    # 创建临时分析器用于可视化（使用第一个GPU或CPU）
    if num_gpus > 0:
        vis_device = torch.device('cuda:0')
    else:
        vis_device = torch.device('cpu')
    # 创建一个简单的dummy dataloader用于可视化（实际上不会被使用）
    vis_analyzer = ChannelCorrelationAnalyzer(data_loader, precipitation_channel=0, device=vis_device)
    
    for method in methods:
        vis_path = os.path.join(args.output_dir, f'channel_correlation_{method}.png')
        vis_analyzer.visualize_results(results, vis_path, method=method)
    
    # 推荐通道
    print("\n" + "="*60)
    print("通道推荐")
    print("="*60)
    
    for method in methods:
        print(f"\n基于 {method.upper()} 的推荐:")
        for top_k in [5, 10, 20, 29]:
            recommended = recommend_channels(results, top_k=top_k, method=method)
            print(f"\nTop {top_k} 相关通道:")
            for idx in recommended:
                info = results[idx]
                corr_value = info.get(method, info.get('importance', 0))
                print(f"  [{idx:2d}] {info['category']:6s}-{info['name']:10s} "
                      f"({method}: {corr_value:.4f})")
    
    # 保存推荐通道配置（使用pearson作为主要指标）
    if 'pearson' in [m for m in methods]:
        recommended_10 = recommend_channels(results, top_k=10, method='pearson')
    else:
        recommended_10 = recommend_channels(results, top_k=10, method=methods[0])
    
    config_file = os.path.join(args.output_dir, 'recommended_channels.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump({
            'top_10_channels': recommended_10,
            'channel_names': {idx: f"{results[idx]['category']}-{results[idx]['name']}"
                             for idx in recommended_10},
            'method': methods[0] if len(methods) == 1 else 'all'
        }, f, indent=2, ensure_ascii=False)
    print(f"\n推荐通道配置已保存到: {config_file}")


if __name__ == '__main__':
    # 设置多进程启动方法为 'spawn'，以支持 CUDA 在多进程中的使用
    # 注意：'spawn' 方法会创建新的 Python 解释器进程，避免 CUDA 重新初始化问题
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # 如果已经设置过启动方法，则忽略错误
        pass
    main()
