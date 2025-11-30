# run/test_scwds_mamba.py
import sys
import os
import glob
import torch
import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np
from datetime import datetime

# 设置 matplotlib 后端
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
matplotlib.use('Agg')

from metai.dataset import ScwdsDataModule
from metai.model.met_mamba.trainer import MeteoMambaModule
from metai.model.met_mamba.metrices import MetScore, MetricTracker
from metai.utils import MetLabel

# ==========================================
# 常量配置 (与 Infer 保持一致)
# ==========================================
THRESHOLDS = [0.1, 1.0, 2.0, 5.0, 8.0]
# 物理空间最小有效降水 (mm)
MIN_VALID_RAIN_MM = 0.1
# 物理最大值 (mm)
PHYSICAL_MAX = 30.0

class TeeLogger:
    def __init__(self, log_file_path):
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.console = sys.stdout
    def write(self, message):
        self.console.write(message); self.log_file.write(message); self.log_file.flush()
    def flush(self):
        self.console.flush(); self.log_file.flush()
    def close(self):
        if self.log_file: self.log_file.close()

_logger = None
def set_logger(path): global _logger; _logger = TeeLogger(path); sys.stdout = _logger
def restore_stdout(): global _logger; sys.stdout = _logger.console if _logger else sys.stdout; _logger = None

def find_best_ckpt(save_dir: str) -> str:
    """查找最佳检查点"""
    best = os.path.join(save_dir, 'best.ckpt')
    if os.path.exists(best): return best
    last = os.path.join(save_dir, 'last.ckpt')
    if os.path.exists(last): return last
    cpts = sorted(glob.glob(os.path.join(save_dir, '**/*.ckpt'), recursive=True))
    if not cpts: raise FileNotFoundError(f'No checkpoint found in {save_dir}')
    return cpts[-1]

def denormalize_to_mm(data_norm):
    """反归一化：Log-Space -> Physical mm"""
    log_factor = np.log(PHYSICAL_MAX + 1)
    if isinstance(data_norm, torch.Tensor):
        return torch.expm1(data_norm * log_factor)
    return np.expm1(data_norm * log_factor)

def plot_seq_visualization(obs_seq, true_seq, pred_seq, out_path, vmax=30.0):
    """绘制序列对比图 (Obs, GT, Pred, Diff)"""
    T = true_seq.shape[0]
    cols = T
    fig, axes = plt.subplots(4, cols, figsize=(cols * 1.5, 6.0), constrained_layout=True)
    if T == 1: axes = axes[:, np.newaxis]
    
    cmap_rain = 'jet' 
    
    for t in range(T):
        # Row 0: Obs
        if t < obs_seq.shape[0]:
            im = obs_seq[t] if obs_seq.ndim == 3 else obs_seq[t, 0]
            axes[0, t].imshow(im, cmap='jet', vmin=0.0, vmax=1.0)
            if t == 0: axes[0, t].set_ylabel('Obs (Norm)', fontsize=8)
        else:
            axes[0, t].set_visible(False)
        axes[0, t].axis('off')
        
        # Row 1: GT
        axes[1, t].imshow(true_seq[t], cmap=cmap_rain, vmin=0.0, vmax=vmax)
        axes[1, t].axis('off')
        if t == 0: axes[1, t].set_ylabel('GT (mm)', fontsize=8)
        
        # Row 2: Pred
        axes[2, t].imshow(pred_seq[t], cmap=cmap_rain, vmin=0.0, vmax=vmax)
        axes[2, t].axis('off')
        if t == 0: axes[2, t].set_ylabel('Pred (mm)', fontsize=8)
        
        # Row 3: Diff
        diff = true_seq[t] - pred_seq[t]
        axes[3, t].imshow(diff, cmap='bwr', vmin=-10.0, vmax=10.0)
        axes[3, t].axis('off')
        if t == 0: axes[3, t].set_ylabel('Diff', fontsize=8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

def print_metrics_table(ts_levels, mae_levels, r_time, score_time, ts_time_matrix, mae_time_matrix, level_weights):
    """打印指标表格"""
    print(f"\n--- Intensity Levels (Avg) ---")
    print(f"{'Level':<10} | {'TS':<8} | {'MAE':<8}")
    for i, thr in enumerate(THRESHOLDS):
        ts = ts_levels[i] if i < len(ts_levels) else 0.0
        mae = mae_levels[i] if i < len(mae_levels) else 0.0
        print(f">={thr:<8} | {ts:<8.4f} | {mae:<8.4f}")

    print(f"\n--- Time Steps (6min) ---")
    print(f"{'Time':<6} | {'Corr':<8} | {'W-TS':<8} | {'W-MAE':<8} | {'Score':<8}")
    for t in range(len(r_time)):
        w_ts = np.sum(ts_time_matrix[t] * level_weights)
        w_mae = np.sum(mae_time_matrix[t] * level_weights)
        print(f"{(t+1)*6:<6} | {r_time[t]:<8.4f} | {w_ts:<8.4f} | {w_mae:<8.4f} | {score_time[t]:<8.4f}")

def process_batch(scorer, obs_tensor, true_tensor, pred_tensor, out_path, batch_idx, level_weights_np):
    """
    处理单个 Batch
    输入: pred_tensor, true_tensor 均为物理空间 (mm) 的数据
    """
    # 计算指标 (使用 use_log_norm=False，因为输入已经是 mm)
    with torch.no_grad():
        score_dict = scorer(pred_tensor, true_tensor)
        
    final_score = score_dict['total_score'].item()
    
    # 打印报告
    print(f"\n[{batch_idx:03d}] Score: {final_score:.6f}")
    print_metrics_table(
        score_dict['ts_levels'].cpu().numpy(), 
        score_dict['mae_levels'].cpu().numpy(), 
        score_dict['r_time'].cpu().numpy(), 
        score_dict['score_time'].cpu().numpy(), 
        score_dict['ts_time'].cpu().numpy(), 
        score_dict['mae_time'].cpu().numpy(), 
        level_weights_np
    )

    # 绘图
    # obs_tensor: 0-1 norm
    obs_np = obs_tensor[0].detach().cpu().numpy()
    true_np = true_tensor[0].detach().cpu().numpy()
    pred_np = pred_tensor[0].detach().cpu().numpy()
    
    print(f"Plotting: {os.path.basename(out_path)} | Max Pred: {pred_np.max():.2f}mm")
    plot_seq_visualization(obs_np, true_np, pred_np, out_path)
    
    return score_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='output/meteo_mamba')
    parser.add_argument('--data_path', type=str, default='data/samples.jsonl')
    parser.add_argument('--in_shape', type=int, nargs=3, default=[31, 256, 256])
    parser.add_argument('--obs_seq_len', type=int, default=10)
    parser.add_argument('--pred_seq_len', type=int, default=20)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--accelerator', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.accelerator if torch.cuda.is_available() else 'cpu')
    
    # Checkpoint
    if not args.ckpt_path:
        args.ckpt_path = find_best_ckpt(args.save_dir)
    
    # Output Dir
    out_dir = os.path.join(os.path.dirname(args.ckpt_path), f'test_vis_{datetime.now().strftime("%m%d_%H%M")}')
    os.makedirs(out_dir, exist_ok=True)
    set_logger(os.path.join(out_dir, 'test.log'))
    
    print(f"[INFO] Testing on {device}, Ckpt: {args.ckpt_path}")

    # 1. Load Model
    module = MeteoMambaModule.load_from_checkpoint(args.ckpt_path)
    module.eval().to(device)
    model = module.model
    
    # 2. Scorer (注意: use_log_norm=False)
    # 因为我们将手动反归一化并清洗数据，所以传给 Scorer 的已经是 mm 值
    scorer = MetScore(use_log_norm=False).to(device)
    tracker = MetricTracker()
    level_weights = scorer.level_weights.cpu().numpy()

    # 3. Data
    dm = ScwdsDataModule(
        data_path=args.data_path, 
        resize_shape=(args.in_shape[1], args.in_shape[2]), 
        batch_size=1, 
        num_workers=4,
        testset_name="TestSet" # 确保和 Infer 一致
    )
    dm.setup('test')
    
    # 4. Loop
    with torch.no_grad():
        for bidx, batch in enumerate(dm.test_dataloader()):
            if bidx >= args.num_samples: break

            _, x, y_log, _, _ = [t.to(device) if isinstance(t, torch.Tensor) else t for t in batch]
            
            # A. 推理 (Log Space)
            pred_log = torch.clamp(model(x), 0.0, 1.0)
            
            # B. 反归一化 (Log -> Physical mm)
            # 这一步是关键，对齐 Infer 的逻辑
            pred_mm = denormalize_to_mm(pred_log)
            true_mm = denormalize_to_mm(y_log)
            
            # C. 物理空间清洗
            pred_mm[pred_mm < MIN_VALID_RAIN_MM] = 0.0
            
            # D. 处理 (Metrics & Vis)
            save_path = os.path.join(out_dir, f'sample_{bidx:03d}.png')
            # 传入的 obs(x) 保持归一化状态用于绘图参考，pred/true 传入 mm 值
            score_dict = process_batch(scorer, x, true_mm, pred_mm, save_path, bidx, level_weights)
            tracker.update(score_dict)
            
    # Summary
    if tracker.count > 0:
        avg = tracker.compute()
        print(f"\n=== FINAL AVERAGE SCORE: {avg['total_score'].item():.6f} ===")
        
    restore_stdout()

if __name__ == '__main__':
    main()