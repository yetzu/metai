# run/test_scwds_mamba.py
import sys
import os
import glob
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
import numpy as np
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
matplotlib.use('Agg')

from metai.dataset.met_dataloader_scwds import ScwdsDataModule
# 导入 MeteoMamba 模块
from metai.model.st_mamba import MeteoMambaModule

# ==========================================
# Part 0: 辅助工具函数
# ==========================================

class TeeLogger:
    """同时输出到控制台和文件的日志类"""
    def __init__(self, log_file_path):
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.console = sys.stdout
        
    def write(self, message):
        self.console.write(message)
        self.log_file.write(message)
        self.log_file.flush()
        
    def flush(self):
        self.console.flush()
        self.log_file.flush()
        
    def close(self):
        if self.log_file:
            self.log_file.close()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

_logger = None

def set_logger(log_file_path):
    global _logger
    _logger = TeeLogger(log_file_path)
    sys.stdout = _logger
    
def restore_stdout():
    global _logger
    if _logger:
        sys.stdout = _logger.console
        _logger.close()
        _logger = None

def get_checkpoint_info(ckpt_path: str):
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        epoch = ckpt.get('epoch', None)
        global_step = ckpt.get('global_step', None)
        hparams = ckpt.get('hyper_parameters', {})
        return {
            'epoch': epoch,
            'global_step': global_step,
            'hparams': hparams,
            'ckpt_name': os.path.basename(ckpt_path)
        }
    except Exception as e:
        return {'error': str(e)}

def print_checkpoint_info(ckpt_info: dict):
    if 'error' in ckpt_info:
        print(f"[WARNING] 无法读取 checkpoint 信息: {ckpt_info['error']}")
        return
    print("=" * 80)
    print(f"[INFO] Loaded Checkpoint: {ckpt_info['ckpt_name']}")
    print(f"  Epoch: {ckpt_info.get('epoch', 'N/A')}")
    print(f"  Global Step: {ckpt_info.get('global_step', 'N/A')}")
    print("=" * 80)

# ==========================================
# Part 1: 全局评分配置 (Metric Configuration)
# ==========================================
class MetricConfig:
    MM_MAX = 30.0
    THRESHOLD_NOISE = 0.05 
    LEVEL_EDGES = np.array([0.0, 0.1, 1.0, 2.0, 5.0, 8.0, np.inf], dtype=np.float32)
    _raw_level_weights = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.3], dtype=np.float32)
    LEVEL_WEIGHTS = _raw_level_weights / _raw_level_weights.sum()
    TIME_WEIGHTS_DICT = {
        0: 0.0075, 1: 0.02, 2: 0.03, 3: 0.04, 4: 0.05,
        5: 0.06, 6: 0.07, 7: 0.08, 8: 0.09, 9: 0.1,
        10: 0.09, 11: 0.08, 12: 0.07, 13: 0.06, 14: 0.05,
        15: 0.04, 16: 0.03, 17: 0.02, 18: 0.0075, 19: 0.005
    }

    @staticmethod
    def get_time_weights(T):
        if T == 20:
            return np.array([MetricConfig.TIME_WEIGHTS_DICT[t] for t in range(T)], dtype=np.float32)
        else:
            return np.ones(T, dtype=np.float32) / T

# ==========================================
# Part 2: 核心统计计算
# ==========================================
def calc_seq_metrics(true_seq, pred_seq, verbose=True):
    """
    计算序列的降水评分指标
    """
    T, H, W = true_seq.shape
    
    pred_clean = pred_seq.copy()
    pred_clean[pred_clean < (MetricConfig.THRESHOLD_NOISE / MetricConfig.MM_MAX)] = 0.0
    
    time_weights = MetricConfig.get_time_weights(T)
    score_k_list = []
    
    tru_mm_seq = np.clip(true_seq, 0.0, None) * MetricConfig.MM_MAX
    prd_mm_seq = np.clip(pred_clean, 0.0, None) * MetricConfig.MM_MAX
    
    ts_mean_levels = np.zeros(len(MetricConfig.LEVEL_WEIGHTS))
    mae_mm_mean_levels = np.zeros(len(MetricConfig.LEVEL_WEIGHTS))
    corr_sum = 0.0

    if verbose:
        print(f"True Stats (mm): Max={np.max(tru_mm_seq):.2f}, Mean={np.mean(tru_mm_seq):.2f}")
        print(f"Pred Stats (mm): Max={np.max(prd_mm_seq):.2f}, Mean={np.mean(prd_mm_seq):.2f}")
        print("-" * 90)
        print(f"{'T':<3} | {'Corr(R)':<9} | {'TS_w_sum':<9} | {'Score_k':<9} | {'W_time':<8}")
        print("-" * 90)

    for t in range(T):
        tru_frame = tru_mm_seq[t].reshape(-1)
        prd_frame = prd_mm_seq[t].reshape(-1)
        abs_err = np.abs(prd_frame - tru_frame)

        # Corr
        mask_valid_corr = (tru_frame > 0) | (prd_frame > 0)
        if mask_valid_corr.sum() > 1:
            t_valid = tru_frame[mask_valid_corr]
            p_valid = prd_frame[mask_valid_corr]
            numerator = np.sum((t_valid - t_valid.mean()) * (p_valid - p_valid.mean()))
            denom = np.sqrt(np.sum((t_valid - t_valid.mean())**2) * np.sum((p_valid - p_valid.mean())**2))
            R_k = numerator / (denom + 1e-8)
        else:
            R_k = 0.0
        
        R_k = float(np.clip(R_k, -1.0, 1.0))
        corr_sum += R_k
        term_corr = np.sqrt(np.exp(R_k - 1.0))

        # TS & MAE
        weighted_sum_metrics = 0.0
        for i in range(len(MetricConfig.LEVEL_WEIGHTS)):
            low = MetricConfig.LEVEL_EDGES[i]
            high = MetricConfig.LEVEL_EDGES[i+1]
            w_i = MetricConfig.LEVEL_WEIGHTS[i]

            tru_bin = (tru_frame >= low) & (tru_frame < high)
            prd_bin = (prd_frame >= low) & (prd_frame < high)
            
            tp = np.logical_and(tru_bin, prd_bin).sum()
            fn = np.logical_and(tru_bin, ~prd_bin).sum()
            fp = np.logical_and(~tru_bin, prd_bin).sum()
            denom_ts = tp + fn + fp
            ts_val = (tp / denom_ts) if denom_ts > 0 else 1.0
            
            mask_eval = tru_bin | prd_bin
            mae_val = np.mean(abs_err[mask_eval]) if mask_eval.sum() > 0 else 0.0
            
            ts_mean_levels[i] += ts_val / T
            mae_mm_mean_levels[i] += mae_val / T
            
            term_mae = np.sqrt(np.exp(-mae_val / 100.0))
            weighted_sum_metrics += w_i * ts_val * term_mae

        Score_k = term_corr * weighted_sum_metrics
        score_k_list.append(Score_k)

        if verbose:
            print(f"{t:<3} | {R_k:<9.4f} | {weighted_sum_metrics:<9.4f} | {Score_k:<9.4f} | {time_weights[t]:<8.4f}")

    score_k_arr = np.array(score_k_list)
    final_score = np.sum(score_k_arr * time_weights)
    
    print("-" * 90)
    ts_str = ", ".join([f"{v:.3f}" for v in ts_mean_levels])
    mae_str = ", ".join([f"{v:.3f}" for v in mae_mm_mean_levels])
    
    print(f"[METRIC] TS_mean  (Levels): {ts_str}")
    print(f"[METRIC] MAE_mean (Levels): {mae_str}")
    print(f"[METRIC] Corr_mean: {corr_sum / T:.4f}")
    print(f"[METRIC] Final_Weighted_Score: {final_score:.6f}")
    print(f"[METRIC] Score_per_t: {', '.join([f'{s:.3f}' for s in score_k_arr])}")
    print("-" * 90)
    
    return final_score

# ==========================================
# Part 3: 绘图功能
# ==========================================
def plot_seq_visualization(obs_seq, true_seq, pred_seq, out_path, vmax=1.0):
    T = true_seq.shape[0]
    rows, cols = 4, T
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5), constrained_layout=True)
    if T == 1: axes = axes[:, np.newaxis]

    for t in range(T):
        if t < obs_seq.shape[0]:
            axes[0, t].imshow(obs_seq[t], cmap='turbo', vmin=0.0, vmax=vmax)
            axes[0, t].set_title(f'In-{t}', fontsize=6)
        else:
            axes[0, t].imshow(np.zeros_like(true_seq[0]), cmap='gray', vmin=0.0, vmax=1.0)
        axes[0, t].axis('off')
        if t == 0: axes[0, t].set_ylabel('Obs', fontsize=8)

        axes[1, t].imshow(true_seq[t], cmap='turbo', vmin=0.0, vmax=vmax)
        axes[1, t].axis('off')
        if t == 0: axes[1, t].set_ylabel('GT', fontsize=8)
        axes[1, t].set_title(f'T+{t+1}', fontsize=6)

        axes[2, t].imshow(pred_seq[t], cmap='turbo', vmin=0.0, vmax=vmax)
        axes[2, t].axis('off')
        if t == 0: axes[2, t].set_ylabel('Pred', fontsize=8)
        
        diff = true_seq[t] - pred_seq[t]
        axes[3, t].imshow(diff, cmap='bwr', vmin=-0.5, vmax=0.5)
        axes[3, t].axis('off')
        if t == 0: axes[3, t].set_ylabel('Diff', fontsize=8)

    print(f'[INFO] Saving Plot to {out_path}')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

# ==========================================
# Part 4: 主逻辑
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description='Test MeteoMamba Model')
    parser.add_argument('--data_path', type=str, default='data/samples.jsonl')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--save_dir', type=str, default='./output/meteo_mamba/vis')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--accelerator', type=str, default='cuda')
    
    # 模型参数 (用于初始化 DataModule，模型参数从 ckpt 加载)
    parser.add_argument('--in_shape', type=int, nargs=4, default=[10, 31, 256, 256])
    parser.add_argument('--aft_seq_length', type=int, default=20)
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.accelerator if torch.cuda.is_available() else 'cpu')
    
    ckpt_info = get_checkpoint_info(args.ckpt_path)
    epoch = ckpt_info.get('epoch', 'unknown')
    
    out_dir = os.path.join(args.save_dir, f'test_epoch_{epoch}')
    os.makedirs(out_dir, exist_ok=True)
    
    log_file = os.path.join(out_dir, 'test_log.txt')
    set_logger(log_file)
    
    print_checkpoint_info(ckpt_info)
    
    # Load Model
    print(f"[INFO] Loading model from {args.ckpt_path}")
    model = MeteoMambaModule.load_from_checkpoint(args.ckpt_path)
    model.eval().to(device)
    
    # Load Data
    resize_shape = (args.in_shape[2], args.in_shape[3])
    data_module = ScwdsDataModule(
        data_path=args.data_path,
        resize_shape=resize_shape,
        batch_size=1,
        num_workers=4
    )
    data_module.setup('test')
    test_loader = data_module.test_dataloader()
    
    scores = []
    
    with torch.no_grad():
        for bidx, batch in enumerate(test_loader):
            # 使用模型内置的 test_step，它已经包含了预处理和后处理
            # batch 需要手动搬运到 device
            batch_device = [
                item.to(device) if isinstance(item, torch.Tensor) else item 
                for item in batch
            ]
            
            # test_step 返回的是 {'inputs': np, 'preds': np, 'trues': np}
            outputs = model.test_step(batch_device, bidx)
            
            save_path = os.path.join(out_dir, f'sample_{bidx:03d}.png')
            
            # 调用统计和绘图
            s = calc_seq_metrics(outputs['trues'], outputs['preds'], verbose=True)
            plot_seq_visualization(outputs['inputs'], outputs['trues'], outputs['preds'], save_path)
            
            scores.append(s)
            
            if bidx >= args.num_samples - 1:
                break
                
    if len(scores) > 0:
        print(f"\n[FINAL] Average Score ({len(scores)} samples): {np.mean(scores):.6f}")
    
    restore_stdout()

if __name__ == '__main__':
    try:
        main()
    finally:
        restore_stdout()