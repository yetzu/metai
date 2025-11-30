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

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
matplotlib.use('Agg')

# 引入数据和模型模块
from metai.dataset import ScwdsDataModule
from metai.model.met_mamba.trainer import MeteoMambaModule
# 引入评分和统计工具
from metai.model.met_mamba.metrices import MetScore, MetricTracker

# ==========================================
# Part 0: 全局常量与配置
# ==========================================

# 对应 MetScore 中的分级阈值
THRESHOLDS = [0.1, 1.0, 2.0, 5.0, 8.0]

class TeeLogger:
    """双向日志记录器：同时输出到控制台和文件"""
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
    """自动查找最佳检查点文件"""
    best = os.path.join(save_dir, 'best.ckpt')
    if os.path.exists(best): return best
    last = os.path.join(save_dir, 'last.ckpt')
    if os.path.exists(last): return last
    cpts = glob.glob(os.path.join(save_dir, '*.ckpt'))
    cpts = [c for c in cpts if 'last.ckpt' not in c and 'best.ckpt' not in c]
    if len(cpts) > 0: return sorted(cpts)[-1]
    all_cpts = sorted(glob.glob(os.path.join(save_dir, '*.ckpt')))
    if len(all_cpts) == 0: raise FileNotFoundError(f'No checkpoint found in {save_dir}')
    return all_cpts[-1]

def get_checkpoint_info(ckpt_path: str):
    """解析检查点文件信息"""
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        epoch = ckpt.get('epoch', None)
        global_step = ckpt.get('global_step', None)
        hparams = ckpt.get('hyper_parameters', {})
        return {'epoch': epoch, 'global_step': global_step, 'hparams': hparams, 'ckpt_name': os.path.basename(ckpt_path)}
    except Exception as e:
        return {'error': str(e)}

def print_checkpoint_info(ckpt_info: dict):
    """打印检查点详细信息"""
    if 'error' in ckpt_info: print(f"[WARNING] {ckpt_info['error']}"); return
    print("=" * 80)
    print(f"[INFO] Loaded Checkpoint: {ckpt_info['ckpt_name']}")
    print(f"  Epoch: {ckpt_info.get('epoch', 'N/A')}")
    print(f"  Global Step: {ckpt_info.get('global_step', 'N/A')}")
    hparams = ckpt_info.get('hparams', {})
    if hparams:
        print(f"  In Shape: {hparams.get('in_shape', 'N/A')}")
        print(f"  Out Seq Length: {hparams.get('pred_seq_len', 'N/A')}")
    print("=" * 80)

def denormalize(data_norm):
    """
    反归一化：将对数空间 [0, 1] 还原为物理空间 [0, 30] mm
    """
    log_factor = np.log(30.0 + 1)
    if isinstance(data_norm, torch.Tensor):
        return torch.expm1(data_norm * log_factor)
    else:
        return np.expm1(data_norm * log_factor)

# ==========================================
# Part 1: 可视化 (Visualization)
# ==========================================
def plot_seq_visualization(obs_seq, true_seq, pred_seq, out_path, vmax=30.0):
    """
    绘制时间序列对比图：Obs, GT, Pred, Diff
    """
    T = true_seq.shape[0]
    rows, cols = 4, T
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5), constrained_layout=True)
    if T == 1: axes = axes[:, np.newaxis]
    
    cmap_rain = 'turbo' 
    
    for t in range(T):
        # Row 0: Observation
        if t < obs_seq.shape[0]:
            im = axes[0, t].imshow(obs_seq[t], cmap=cmap_rain, vmin=0.0, vmax=vmax)
            axes[0, t].set_title(f'In-{t}', fontsize=6)
        else:
            axes[0, t].imshow(np.zeros_like(true_seq[0]), cmap='gray', vmin=0, vmax=1)
        axes[0, t].axis('off')
        if t == 0: axes[0, t].set_ylabel('Obs (mm)', fontsize=8)
        
        # Row 1: Ground Truth
        axes[1, t].imshow(true_seq[t], cmap=cmap_rain, vmin=0.0, vmax=vmax)
        axes[1, t].axis('off')
        if t == 0: axes[1, t].set_ylabel('GT (mm)', fontsize=8)
        axes[1, t].set_title(f'T+{t+1}', fontsize=6)
        
        # Row 2: Prediction
        axes[2, t].imshow(pred_seq[t], cmap=cmap_rain, vmin=0.0, vmax=vmax)
        axes[2, t].axis('off')
        if t == 0: axes[2, t].set_ylabel('Pred (mm)', fontsize=8)
        
        # Row 3: Difference
        diff = true_seq[t] - pred_seq[t]
        axes[3, t].imshow(diff, cmap='bwr', vmin=-15.0, vmax=15.0)
        axes[3, t].axis('off')
        if t == 0: axes[3, t].set_ylabel('Diff', fontsize=8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

# ==========================================
# Part 2: 主处理逻辑 (Main Processing)
# ==========================================
def print_metrics_table(ts_levels, mae_levels, r_time, score_time, ts_time_matrix, mae_time_matrix, level_weights, prefix=""):
    """
    通用函数：打印格式化的指标表格，包含时次权重累积
    """
    # 1. 打印分级指标 (Level-wise Average)
    print(f"\n{prefix}--- Performance by Intensity Level (Avg over Time) ---")
    print(f"{'Level (mm)':<15} | {'TS Score':<12} | {'MAE':<12}")
    print("-" * 46)
    for i, thr in enumerate(THRESHOLDS):
        ts_val = ts_levels[i] if i < len(ts_levels) else 0.0
        mae_val = mae_levels[i] if i < len(mae_levels) else 0.0
        label = f">= {thr}"
        print(f"{label:<15} | {ts_val:<12.4f} | {mae_val:<12.4f}")

    # 2. 打印时效指标 (Time-wise)
    print(f"\n{prefix}--- Performance by Time Step (6min interval) ---")
    # Updated Order: Step | Time | Corr (R) | W-TS | W-MAE | Score_k
    print(f"{'Step':<6} | {'Time':<8} | {'Corr (R)':<10} | {'W-TS':<10} | {'W-MAE':<10} | {'Score_k':<10}")
    print("-" * 72)
    
    for t in range(len(r_time)):
        # 计算该时次的权重累积值
        # ts_time_matrix[t] 是一个向量 [Level1, Level2, ...]
        w_ts = np.sum(ts_time_matrix[t] * level_weights)
        w_mae = np.sum(mae_time_matrix[t] * level_weights)
        
        time_min = (t + 1) * 6
        print(f"T+{t+1:<3} | {time_min:<6}m | {r_time[t]:<10.4f} | {w_ts:<10.4f} | {w_mae:<10.4f} | {score_time[t]:<10.4f}")
    print("-" * 72)

def process_batch(scorer, obs_tensor, true_tensor, pred_tensor, out_path, batch_idx, level_weights_np):
    """
    处理单个 Batch：计算指标、打印报告、绘图
    """
    
    # 1. 计算指标 (Score)
    with torch.no_grad():
        score_dict = scorer(pred_tensor, true_tensor)
        
    final_score = score_dict['total_score'].item()
    
    # 提取 Tensor 数据并转为 Numpy
    ts_levels = score_dict['ts_levels'].cpu().numpy()
    mae_levels = score_dict['mae_levels'].cpu().numpy()
    r_time = score_dict['r_time'].cpu().numpy()
    score_time = score_dict['score_time'].cpu().numpy()
    ts_time_matrix = score_dict['ts_time'].cpu().numpy()
    mae_time_matrix = score_dict['mae_time'].cpu().numpy()

    # 2. 打印单样本报告
    print(f"\n{'='*25} Sample {batch_idx:03d} Report {'='*25}")
    
    print_metrics_table(
        ts_levels, mae_levels, r_time, score_time, 
        ts_time_matrix, mae_time_matrix, level_weights_np
    )

    print(f"[METRIC] Total Weighted Score: {final_score:.6f}")
    

    # 3. 准备可视化数据 (转换为物理值 mm)
    def to_numpy_ch_mm(x, ch=0):
        if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
        if x.ndim == 5: x = x[0] 
        if x.ndim == 4: x = x[:, ch] 
        return denormalize(x)

    obs_mm = to_numpy_ch_mm(obs_tensor) 
    true_mm = to_numpy_ch_mm(true_tensor)
    pred_mm = to_numpy_ch_mm(pred_tensor)
    
    # 计算非零值（>0.1mm）比例，衡量雨区面积占比
    rain_threshold = 0.1
    gt_rain_ratio = (true_mm > rain_threshold).mean() * 100
    pred_rain_ratio = (pred_mm > rain_threshold).mean() * 100

    print(f"Plotting (Physical mm): {os.path.basename(out_path)}")
    print(f"  Max Rain: GT={true_mm.max():.2f}mm, Pred={pred_mm.max():.2f}mm")
    print(f"  Rain Ratio (>{rain_threshold}mm): GT={gt_rain_ratio:.2f}%, Pred={pred_rain_ratio:.2f}%")
    
    plot_seq_visualization(obs_mm, true_mm, pred_mm, out_path, vmax=30.0)
    
    return score_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='output/meteo_mamba')
    parser.add_argument('--data_path', type=str, default='data/samples.jsonl')
    parser.add_argument('--in_shape', type=int, nargs=3, default=[31, 256, 256], 
                        help='Input shape (C, H, W) without batch or time dim')
    parser.add_argument('--obs_seq_len', type=int, default=10, help='Observation sequence length')
    parser.add_argument('--pred_seq_len', type=int, default=20, help='Prediction sequence length')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--accelerator', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.accelerator if torch.cuda.is_available() else 'cpu')
    
    # 自动寻找 Checkpoint
    if args.ckpt_path is None or not os.path.exists(args.ckpt_path):
        try:
            args.ckpt_path = find_best_ckpt(args.save_dir)
        except FileNotFoundError:
            print(f"[ERROR] No checkpoint found in {args.save_dir}")
            return

    ckpt_info = get_checkpoint_info(args.ckpt_path)
    epoch = ckpt_info.get('epoch', 0)
    
    # 确定输出目录
    ckpt_dir_abs = os.path.dirname(os.path.abspath(args.ckpt_path))
    if os.path.basename(ckpt_dir_abs) == 'checkpoints':
        version_dir = os.path.dirname(ckpt_dir_abs)
    else:
        version_dir = ckpt_dir_abs
    
    out_dir = os.path.join(version_dir, f'vis_epoch_{epoch:02d}')
    
    os.makedirs(out_dir, exist_ok=True)
    set_logger(os.path.join(out_dir, 'test_report.log'))
    
    print(f"[INFO] Starting Test on {device}")
    print_checkpoint_info(ckpt_info)
    print(f"[INFO] Input Shape (C, H, W): {args.in_shape}")
    print(f"[INFO] Seq Lengths: Obs={args.obs_seq_len}, Pred={args.pred_seq_len}")
    print(f"[INFO] Results will be saved to: {out_dir}")
    
    # 1. 加载模型
    try:
        module = MeteoMambaModule.load_from_checkpoint(args.ckpt_path)
        module.eval()
        module.to(device)
        model = module.model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # 2. 初始化评分器 & 指标追踪器
    scorer = MetScore(use_log_norm=True).to(device)
    tracker = MetricTracker()
    
    # 获取 Level Weights (用于计算加权 TS/MAE)
    level_weights_np = scorer.level_weights.cpu().numpy()
    
    print(f"[INFO] MetScore & MetricTracker initialized")
    print(f"[INFO] Level Weights: {level_weights_np}")

    # 3. 加载数据
    resize_shape = (args.in_shape[1], args.in_shape[2])
    dm = ScwdsDataModule(
        data_path=args.data_path, 
        resize_shape=resize_shape, 
        batch_size=1, 
        num_workers=4,
        aft_seq_length=args.pred_seq_len
    )
    dm.setup('test')
    
    # 4. 循环测试
    with torch.no_grad():
        for bidx, batch in enumerate(dm.test_dataloader()):
            if bidx >= args.num_samples:
                print(f"\n[INFO] Reached sample limit ({args.num_samples}). Stopping.")
                break

            batch_device = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            _, x, y, _, t_mask = batch_device
            
            # 推理
            pred_raw = model(x)
            pred = torch.clamp(pred_raw, 0.0, 1.0)
            
            # 单样本处理
            save_path = os.path.join(out_dir, f'sample_{bidx:03d}.png')
            score_dict = process_batch(scorer, x, y, pred, save_path, bidx, level_weights_np)
            
            # 累积指标
            tracker.update(score_dict)
            
    # 5. 最终汇总报告
    if tracker.count > 0:
        avg_metrics = tracker.compute()
        
        print(f"\n\n{'='*30} FINAL TEST SUMMARY ({tracker.count} samples) {'='*30}")
        # 提取平均指标并打印
        ts_levels_avg = avg_metrics['ts_levels'].cpu().numpy()
        mae_levels_avg = avg_metrics['mae_levels'].cpu().numpy()
        r_time_avg = avg_metrics['r_time'].cpu().numpy()
        score_time_avg = avg_metrics['score_time'].cpu().numpy()
        ts_time_avg = avg_metrics['ts_time'].cpu().numpy()
        mae_time_avg = avg_metrics['mae_time'].cpu().numpy()
        
        print_metrics_table(
            ts_levels_avg, mae_levels_avg, r_time_avg, score_time_avg,
            ts_time_avg, mae_time_avg, level_weights_np,
            prefix="[AVG] "
        )

        print(f"[AVERAGE TOTAL SCORE]: {avg_metrics['total_score'].item():.6f}")
        
        print("="*88)
        print(f"Log saved to: {os.path.join(out_dir, 'test_report.log')}")
        
    restore_stdout()

if __name__ == '__main__':
    try:
        main()
    finally:
        restore_stdout()