# run/test_scwds_mamba.py
import sys
import os
import glob
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import numpy as np
import re
from datetime import datetime

# 设置 matplotlib 后端
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
matplotlib.use('Agg')

from metai.dataset import ScwdsDataModule
from metai.model.met_mamba.trainer import MeteoMambaModule
from metai.model.met_mamba.metrices import MetScore, MetricTracker
from metai.utils import MetLabel

# ==========================================
# 常量配置
# ==========================================
THRESHOLDS = [0.1, 1.0, 2.0, 5.0, 8.0]
MIN_VALID_RAIN_MM = 0.1 
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
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"Directory not found: {save_dir}")
        
    best = os.path.join(save_dir, 'best.ckpt')
    if os.path.exists(best): return best
    last = os.path.join(save_dir, 'last.ckpt')
    if os.path.exists(last): return last
    
    # 递归查找所有ckpt
    cpts = sorted(glob.glob(os.path.join(save_dir, '**/*.ckpt'), recursive=True))
    if not cpts: raise FileNotFoundError(f'No checkpoint found in {save_dir}')
    return cpts[-1]

def denormalize_to_mm(data_norm):
    """反归一化：Log-Space -> Physical mm"""
    log_factor = np.log(PHYSICAL_MAX + 1)
    if isinstance(data_norm, torch.Tensor):
        return torch.expm1(data_norm * log_factor)
    return np.expm1(data_norm * log_factor)

def get_data_stats(data, label):
    """计算数据统计指标: Min, Max, Mean, Non-Zero%"""
    if isinstance(data, torch.Tensor):
        val_min = data.min().item()
        val_max = data.max().item()
        val_mean = data.mean().item()
        nz_pct = (data > MIN_VALID_RAIN_MM).float().mean().item() * 100.0
    else:
        val_min = data.min()
        val_max = data.max()
        val_mean = data.mean()
        nz_pct = (data > MIN_VALID_RAIN_MM).mean() * 100.0
        
    return f"{label:<4} | Min:{val_min:6.2f} | Max:{val_max:6.2f} | Mean:{val_mean:6.4f} | NZ:{nz_pct:5.2f}%"

def create_precipitation_cmap():
    """
    创建自定义降水色标
    区间: 0<r<0.1, 0.1<=r<1, 1<=r<2, 2<=r<5, 5<=r<8, r>=8
    r=0 用白色表示
    """
    hex_colors = [
        '#9CF48D',  # 0 < r < 0.1 (浅绿)
        '#3CB73A',  # 0.1 <= r < 1 (中绿)
        '#63B7FF',  # 1 <= r < 2 (浅蓝)
        '#0200F9',  # 2 <= r < 5 (深蓝)
        '#EE00F0',  # 5 <= r < 8 (紫红)
        '#9F0000'   # r >= 8 (深红)
    ]
    
    cmap = mcolors.ListedColormap(hex_colors)
    cmap.set_bad('white') 
    cmap.set_under('white')
    
    bounds = [0, 0.1, 1, 2, 5, 8, 100]
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=len(hex_colors))
    
    return cmap, norm

def plot_seq_visualization(obs_seq, true_seq, pred_seq, out_path, vmax=30.0):
    """绘制序列对比图 (Obs, GT, Pred, Diff)"""
    T = true_seq.shape[0]
    obs_len = obs_seq.shape[0] 
    cols = T
    
    # [修改] 增加 figsize 高度以容纳间隙，移除 constrained_layout
    fig, axes = plt.subplots(4, cols, figsize=(cols * 1.5, 7.0))
    if T == 1: axes = axes[:, np.newaxis]
    
    precip_cmap, precip_norm = create_precipitation_cmap()
    
    for t in range(T):
        # Row 0: Obs
        if t < obs_len:
            im = obs_seq[t]
            if im.ndim == 3: im = im[0]
            im_masked = np.ma.masked_less_equal(im, 0)
            
            axes[0, t].imshow(im_masked, cmap=precip_cmap, norm=precip_norm, interpolation='nearest')
            if t == 0: axes[0, t].set_ylabel('Obs', fontsize=9, fontweight='bold') 
            
            # Label T-M
            axes[0, t].set_title(f"T - {obs_len - 1 - t}", fontsize=8)

            axes[0, t].set_xticks([])
            axes[0, t].set_yticks([])
            for spine in axes[0, t].spines.values():
                spine.set_visible(True); spine.set_linewidth(0.5)
        else:
            axes[0, t].set_visible(False)
        
        # Row 1: GT
        im_gt = true_seq[t]
        if im_gt.ndim == 3: im_gt = im_gt[0]
        im_gt_masked = np.ma.masked_less_equal(im_gt, 0)
        
        axes[1, t].imshow(im_gt_masked, cmap=precip_cmap, norm=precip_norm, interpolation='nearest')
        if t == 0: axes[1, t].set_ylabel('GT', fontsize=9, fontweight='bold')
        
        # Label T+N
        axes[1, t].set_title(f"T + {t + 1}", fontsize=8)

        axes[1, t].set_xticks([])
        axes[1, t].set_yticks([])
        for spine in axes[1, t].spines.values():
            spine.set_visible(True); spine.set_linewidth(0.5)
        
        # Row 2: Pred
        im_pred = pred_seq[t]
        if im_pred.ndim == 3: im_pred = im_pred[0]
        im_pred_masked = np.ma.masked_less_equal(im_pred, 0)
        
        axes[2, t].imshow(im_pred_masked, cmap=precip_cmap, norm=precip_norm, interpolation='nearest')
        if t == 0: axes[2, t].set_ylabel('Pred', fontsize=9, fontweight='bold')
        
        axes[2, t].set_xticks([])
        axes[2, t].set_yticks([])
        for spine in axes[2, t].spines.values():
            spine.set_visible(True); spine.set_linewidth(0.5)
        
        # Row 3: Diff
        gt_s = true_seq[t][0] if true_seq[t].ndim == 3 else true_seq[t]
        pd_s = pred_seq[t][0] if pred_seq[t].ndim == 3 else pred_seq[t]
        diff = gt_s - pd_s
        
        # [修改] Diff 范围 [-30, 30]
        axes[3, t].imshow(diff, cmap='bwr', vmin=-30.0, vmax=30.0)
        if t == 0: axes[3, t].set_ylabel('Diff', fontsize=9, fontweight='bold')

        axes[3, t].set_xticks([])
        axes[3, t].set_yticks([])
        for spine in axes[3, t].spines.values():
            spine.set_visible(True); spine.set_linewidth(0.5)

    # [修改] 调整布局
    # hspace=0.3 增加行间距，为 T+N 留出位置
    fig.subplots_adjust(bottom=0.12, top=0.95, left=0.02, right=0.98, wspace=0.1, hspace=0.3)
    
    # --- Legend 1: Precipitation (左侧) ---
    # [修改] 缩小尺寸：宽度 0.25, 高度 0.015
    cbar_ax_precip = fig.add_axes([0.20, 0.05, 0.25, 0.015])
    sm_precip = plt.cm.ScalarMappable(cmap=precip_cmap, norm=precip_norm)
    sm_precip.set_array([])
    cbar_p = fig.colorbar(sm_precip, cax=cbar_ax_precip, orientation='horizontal', spacing='uniform')
    cbar_p.set_ticks([0.1, 1, 2, 5, 8])
    cbar_p.set_ticklabels(['0.1', '1', '2', '5', '8'])
    cbar_p.set_label('Precipitation (mm)', fontsize=8)
    cbar_p.ax.tick_params(labelsize=7)
    
    # --- Legend 2: Diff (右侧) ---
    # [新增] Diff 图例，范围 -30 到 30
    cbar_ax_diff = fig.add_axes([0.55, 0.05, 0.25, 0.015])
    sm_diff = plt.cm.ScalarMappable(cmap='bwr', norm=plt.Normalize(vmin=-30, vmax=30))
    sm_diff.set_array([])
    cbar_d = fig.colorbar(sm_diff, cax=cbar_ax_diff, orientation='horizontal')
    cbar_d.set_ticks([-30, -15, 0, 15, 30])
    cbar_d.set_label('Difference (mm)', fontsize=8)
    cbar_d.ax.tick_params(labelsize=7)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

def print_metrics_table(ts_levels, mae_levels, r_time, score_time, ts_time_matrix, mae_time_matrix, level_weights, time_weights):
    """打印指标表格"""
    print(f"\n--- Intensity Levels (Avg) ---")
    print(f"{'Level':<10} | {'TS':<8} | {'MAE':<8}")
    for i, thr in enumerate(THRESHOLDS):
        ts = ts_levels[i] if i < len(ts_levels) else 0.0
        mae = mae_levels[i] if i < len(mae_levels) else 0.0
        print(f">={thr:<8} | {ts:<8.4f} | {mae:<8.4f}")

    print(f"\n-------------------- Time Steps (6min) ----------------------")
    print(f"{'Frame':<6} | {'Weight':<8} | {'Corr':<8} | {'W-TS':<8} | {'W-MAE':<8} | {'Score':<8}")
    for t in range(len(r_time)):
        w_ts = np.sum(ts_time_matrix[t] * level_weights)
        w_mae = np.sum(mae_time_matrix[t] * level_weights)
        
        tw = time_weights[t] if t < len(time_weights) else 0.0
        frame_str = f"{t+1:02d}"
        print(f"{frame_str:<6} | {tw:<8.4f} | {r_time[t]:<8.4f} | {w_ts:<8.4f} | {w_mae:<8.4f} | {score_time[t]:<8.4f}")

def process_batch(scorer, obs_tensor, true_tensor, pred_tensor, out_path, batch_idx, level_weights_np):
    """
    处理单个 Batch
    输入: pred_tensor, true_tensor 均为 Log-Norm 空间 ([0, 1]) 的数据
    """
    # 1. 计算评分
    with torch.no_grad():
        score_dict = scorer(pred_tensor, true_tensor)
    final_score = score_dict['total_score'].item()

    # 2. 计算时间权重 (用于打印)
    T = pred_tensor.shape[1]
    if hasattr(scorer, 'time_weights_default'):
        w_def = scorer.time_weights_default
        if T == len(w_def):
            time_weights = w_def
        elif T < len(w_def):
            time_weights = w_def[:T]
            time_weights = time_weights / time_weights.sum()
        else:
            time_weights = torch.ones(T, device=w_def.device) / T
        time_weights_np = time_weights.cpu().numpy()
    else:
        time_weights_np = np.ones(T) / T

    # 3. 数据准备 (反归一化到物理空间 mm)
    obs_mm = denormalize_to_mm(obs_tensor)
    true_mm = denormalize_to_mm(true_tensor)
    pred_mm = denormalize_to_mm(pred_tensor)
    
    pred_mm_clean = pred_mm.clone()
    pred_mm_clean[pred_mm_clean < MIN_VALID_RAIN_MM] = 0.0

    # 4. 打印日志信息
    print("-" * 60)
    print(f"Processing Sample [{batch_idx:03d}]: {os.path.basename(out_path)}")
    
    # 打印 Ground Truth (GT) 和 预测 (Pred) 的统计信息
    print(get_data_stats(true_mm, "GT"))   
    print(get_data_stats(pred_mm_clean, "Pred"))  
    
    # 打印指标表
    print_metrics_table(
        score_dict['ts_levels'].cpu().numpy(), 
        score_dict['mae_levels'].cpu().numpy(), 
        score_dict['r_time'].cpu().numpy(), 
        score_dict['score_time'].cpu().numpy(), 
        score_dict['ts_time'].cpu().numpy(), 
        score_dict['mae_time'].cpu().numpy(), 
        level_weights_np,
        time_weights_np
    )
    
    print(f"\n[{batch_idx:03d}] Sample Score: {final_score:.6f}")

    # 5. 绘图
    obs_np = obs_mm[0].detach().cpu().numpy()
    true_np = true_mm[0].detach().cpu().numpy()
    pred_np = pred_mm_clean[0].detach().cpu().numpy()
    
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
        if os.path.isdir(args.save_dir):
            try:
                args.ckpt_path = find_best_ckpt(args.save_dir)
            except FileNotFoundError:
                args.ckpt_path = find_best_ckpt(os.path.dirname(args.save_dir))
        else:
             args.ckpt_path = find_best_ckpt(os.path.dirname(args.save_dir))

    print(f"[INFO] Testing on {device}, Ckpt: {args.ckpt_path}")
    
    # Output Dir
    ckpt_filename = os.path.basename(args.ckpt_path)
    match = re.search(r'(epoch=\d+)', ckpt_filename)
    if match:
        dir_suffix = match.group(1)
    else:
        dir_suffix = datetime.now().strftime("%m%d_%H%M")
    
    out_dir = os.path.join(os.path.dirname(args.ckpt_path), f'test_vis_{dir_suffix}')
    os.makedirs(out_dir, exist_ok=True)
    set_logger(os.path.join(out_dir, 'test.log'))

    # 1. Load Model
    module = MeteoMambaModule.load_from_checkpoint(args.ckpt_path)
    module.eval().to(device)
    model = module.model
    
    # 2. Scorer
    scorer = MetScore(use_log_norm=True).to(device)
    tracker = MetricTracker()
    level_weights = scorer.level_weights.cpu().numpy()

    # 3. Data
    dm = ScwdsDataModule(
        data_path=args.data_path, 
        resize_shape=(args.in_shape[1], args.in_shape[2]), 
        batch_size=1, 
        num_workers=4
    )
    dm.setup('test')
    
    # 4. Loop
    with torch.no_grad():
        for bidx, batch in enumerate(dm.test_dataloader()):
            if bidx >= args.num_samples: break

            if len(batch) == 5:
                _, x, y_log, _, _ = batch
            else:
                x, y_log = batch[:2]

            x = x.to(device)
            y_log = y_log.to(device)

            # pred_log = torch.clamp(model(x), 0.0, 1.0)
            pred_log = torch.clamp(model(x, current_epoch=100), 0.0, 1.0)
            
            save_path = os.path.join(out_dir, f'sample_{bidx:03d}.png')
            
            score_dict = process_batch(scorer, x, y_log, pred_log, save_path, bidx, level_weights)
            tracker.update(score_dict)
            
    if tracker.count > 0:
        avg = tracker.compute()
        
        # [修改] 计算最终的平均时间权重，用于打印汇总表
        T_final = avg['score_time'].shape[0]
        if hasattr(scorer, 'time_weights_default'):
            w_def = scorer.time_weights_default
            if T_final == len(w_def):
                time_weights_final = w_def
            elif T_final < len(w_def):
                time_weights_final = w_def[:T_final]
                time_weights_final = time_weights_final / time_weights_final.sum()
            else:
                time_weights_final = torch.ones(T_final, device=w_def.device) / T_final
            time_weights_final_np = time_weights_final.cpu().numpy()
        else:
            time_weights_final_np = np.ones(T_final) / T_final

        print("=" * 60)
        print("   FINAL AGGREGATED METRICS (Average over all samples)")
        print("=" * 60)
        
        # [修改] 打印汇总表
        print_metrics_table(
            avg['ts_levels'].cpu().numpy(), 
            avg['mae_levels'].cpu().numpy(), 
            avg['r_time'].cpu().numpy(), 
            avg['score_time'].cpu().numpy(), 
            avg['ts_time'].cpu().numpy(), 
            avg['mae_time'].cpu().numpy(), 
            level_weights,
            time_weights_final_np
        )
        
        print("-" * 60)
        print(f"=== FINAL AVERAGE SCORE: {avg['total_score'].item():.6f} ===")
        print("-" * 60)
        
    restore_stdout()

if __name__ == '__main__':
    main()