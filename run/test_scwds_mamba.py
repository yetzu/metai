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

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
matplotlib.use('Agg')

from metai.dataset.met_dataloader_scwds import ScwdsDataModule
# [关键] 确保引用正确的模块名
from metai.model.met_mamba.trainer import MeteoMambaModule

# ==========================================
# Part 0: 辅助工具函数
# ==========================================

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
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        epoch = ckpt.get('epoch', None)
        global_step = ckpt.get('global_step', None)
        hparams = ckpt.get('hyper_parameters', {})
        return {'epoch': epoch, 'global_step': global_step, 'hparams': hparams, 'ckpt_name': os.path.basename(ckpt_path)}
    except Exception as e:
        return {'error': str(e)}

def print_checkpoint_info(ckpt_info: dict):
    if 'error' in ckpt_info: print(f"[WARNING] {ckpt_info['error']}"); return
    print("=" * 80)
    print(f"[INFO] Loaded Checkpoint: {ckpt_info['ckpt_name']}")
    print(f"  Epoch: {ckpt_info.get('epoch', 'N/A')}")
    print(f"  Global Step: {ckpt_info.get('global_step', 'N/A')}")
    hparams = ckpt_info.get('hparams', {})
    if hparams:
        print(f"  In Shape: {hparams.get('in_shape', 'N/A')}")
        print(f"  Out Seq Length: {hparams.get('aft_seq_length', 'N/A')}")
    print("=" * 80)

# ==========================================
# Part 1: Metric Config
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
        if T == 20: return np.array([MetricConfig.TIME_WEIGHTS_DICT[t] for t in range(T)], dtype=np.float32)
        else: return np.ones(T, dtype=np.float32) / T

# ==========================================
# Part 2: Metrics Calculation
# ==========================================
def calc_seq_metrics(true_seq, pred_seq, verbose=True):
    T, H, W = true_seq.shape
    pred_clean = pred_seq.copy()
    pred_clean[pred_clean < (MetricConfig.THRESHOLD_NOISE / MetricConfig.MM_MAX)] = 0.0
    
    tru_mm = np.clip(true_seq, 0, None) * MetricConfig.MM_MAX
    prd_mm = np.clip(pred_clean, 0, None) * MetricConfig.MM_MAX
    time_weights = MetricConfig.get_time_weights(T)
    
    score_list = []
    corr_sum = 0.0
    ts_mean_levels = np.zeros(len(MetricConfig.LEVEL_WEIGHTS))
    mae_mm_mean_levels = np.zeros(len(MetricConfig.LEVEL_WEIGHTS))
    
    if verbose:
        print(f"True Stats (mm): Max={np.max(tru_mm):.2f}, Mean={np.mean(tru_mm):.2f}")
        print(f"Pred Stats (mm): Max={np.max(prd_mm):.2f}, Mean={np.mean(prd_mm):.2f}")
        print("-" * 90)
        print(f"{'T':<3} | {'Corr(R)':<9} | {'TS_w_sum':<9} | {'Score_k':<9} | {'W_time':<8}")
        print("-" * 90)

    for t in range(T):
        tf, pf = tru_mm[t].flatten(), prd_mm[t].flatten()
        abs_err = np.abs(pf - tf)
        mask = (tf>0)|(pf>0)
        R_k = np.corrcoef(tf[mask], pf[mask])[0,1] if mask.sum()>1 else 0.0
        if np.isnan(R_k): R_k=0.0
        R_k = np.clip(R_k, -1, 1)
        corr_sum += R_k
        term_corr = np.sqrt(np.exp(R_k - 1.0))
        
        w_sum_metrics = 0.0
        for i in range(len(MetricConfig.LEVEL_WEIGHTS)):
            l, h = MetricConfig.LEVEL_EDGES[i], MetricConfig.LEVEL_EDGES[i+1]
            w_i = MetricConfig.LEVEL_WEIGHTS[i]
            t_bin, p_bin = (tf>=l)&(tf<h), (pf>=l)&(pf<h)
            tp, fn, fp = (t_bin&p_bin).sum(), (t_bin&~p_bin).sum(), (~t_bin&p_bin).sum()
            ts = tp / (tp+fn+fp+1e-8) if (tp+fn+fp)>0 else 1.0
            mask_eval = t_bin | p_bin
            mae = np.mean(abs_err[mask_eval]) if mask_eval.sum()>0 else 0.0
            ts_mean_levels[i] += ts / T
            mae_mm_mean_levels[i] += mae / T
            w_sum_metrics += w_i * ts * np.sqrt(np.exp(-mae/100.0))
            
        Score_k = term_corr * w_sum_metrics
        score_list.append(Score_k)
        if verbose: print(f"{t:<3} | {R_k:<9.4f} | {w_sum_metrics:<9.4f} | {Score_k:<9.4f} | {time_weights[t]:<8.4f}")
        
    final = (np.array(score_list) * time_weights).sum()
    print("-" * 90)
    print(f"[METRIC] TS_mean  (Levels): {', '.join([f'{v:.3f}' for v in ts_mean_levels])}")
    print(f"[METRIC] MAE_mean (Levels): {', '.join([f'{v:.3f}' for v in mae_mm_mean_levels])}")
    print(f"[METRIC] Corr_mean: {corr_sum / T:.4f}")
    print(f"[METRIC] Final_Weighted_Score: {final:.6f}")
    print("-" * 90)
    return {"final_score": final, "pred_clean": pred_clean}

# ==========================================
# Part 3: Visualization
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
            axes[0, t].imshow(np.zeros_like(true_seq[0]), cmap='gray', vmin=0, vmax=1)
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

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

# ==========================================
# Part 4: Main
# ==========================================
def render(obs_seq, true_seq, pred_seq, out_path):
    # [关键修复] 统一处理维度
    def to_numpy_ch(x, ch=0):
        if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
        
        # 1. 移除 Batch 维度 (B=1) -> (T, C, H, W)
        if x.ndim == 5: x = x[0]
            
        # 2. 提取单通道 -> (T, H, W)
        if x.ndim == 4: x = x[:, ch]
            
        return x

    obs = to_numpy_ch(obs_seq) 
    tru = to_numpy_ch(true_seq)
    prd = to_numpy_ch(pred_seq)
    
    print(f"Processing: {os.path.basename(out_path)}")
    metrics = calc_seq_metrics(tru, prd, verbose=True)
    plot_seq_visualization(obs, tru, metrics['pred_clean'], out_path)
    return metrics['final_score']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='output/meteo_mamba')
    parser.add_argument('--data_path', type=str, default='data/samples.jsonl')
    parser.add_argument('--in_shape', type=int, nargs=4, default=[10, 31, 256, 256])
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--accelerator', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.accelerator if torch.cuda.is_available() else 'cpu')
    
    if args.ckpt_path is None or not os.path.exists(args.ckpt_path):
        try:
            args.ckpt_path = find_best_ckpt(args.save_dir)
        except FileNotFoundError:
            print(f"[ERROR] No checkpoint found in {args.save_dir}")
            return

    ckpt_info = get_checkpoint_info(args.ckpt_path)
    epoch = ckpt_info.get('epoch', 0)
    
    out_dir = os.path.join(args.save_dir, f'vis_{epoch:02d}')
    os.makedirs(out_dir, exist_ok=True)
    set_logger(os.path.join(out_dir, 'log.txt'))
    
    print(f"[INFO] Starting Test on {device}")
    print_checkpoint_info(ckpt_info)
    print(f"[INFO] Input Shape: {args.in_shape}")
    print(f"[INFO] Metric MM_MAX: {MetricConfig.MM_MAX}")
    print(f"[INFO] 可视化结果将保存到: {out_dir}")
    
    try:
        model = MeteoMambaModule.load_from_checkpoint(args.ckpt_path).to(device).eval()
    except Exception as e:
        print(f"[ERROR] 模型加载失败: {e}")
        return
    
    resize_shape = (args.in_shape[2], args.in_shape[3])
    dm = ScwdsDataModule(data_path=args.data_path, resize_shape=resize_shape, batch_size=1, num_workers=4)
    dm.setup('test')
    
    scores = []
    with torch.no_grad():
        for bidx, batch in enumerate(dm.test_dataloader()):
            if bidx >= args.num_samples:
                print(f"[INFO] Reached limit ({args.num_samples}). Stopping.")
                break

            batch_device = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            
            if hasattr(model, 'test_step') and callable(getattr(model, 'test_step')):
                 outputs = model.test_step(batch_device, bidx)
            else:
                 _, x, y, _, _ = batch_device
                 logits = model.model(x)
                 pred = torch.clamp(torch.sigmoid(logits), 0.0, 1.0)
                 outputs = {'inputs': x, 'trues': y, 'preds': pred}

            save_path = os.path.join(out_dir, f'sample_{bidx:03d}.png')
            s = render(outputs['inputs'], outputs['trues'], outputs['preds'], save_path)
            scores.append(s)
            
    if len(scores) > 0:
        print(f"\n[FINAL] Average Score ({len(scores)} samples): {np.mean(scores):.6f}")
    restore_stdout()

if __name__ == '__main__':
    try:
        main()
    finally:
        restore_stdout()