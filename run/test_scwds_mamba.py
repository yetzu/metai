# run/test_scwds_mamba.py
import sys
import os
import torch
import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
matplotlib.use('Agg')

from metai.dataset.met_dataloader_scwds import ScwdsDataModule
from metai.model.st_mamba import MeteoMambaModule

# ==========================================
# Logger
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

def set_logger(path): global _logger; _logger = TeeLogger(path); sys.stdout = _logger
def restore_stdout(): global _logger; sys.stdout = _logger.console if _logger else sys.stdout; _logger = None

# ==========================================
# Global Metric Config (SimVP Style)
# ==========================================
class MetricConfig:
    MM_MAX = 30.0
    THRESHOLD_NOISE = 0.05 
    LEVEL_EDGES = np.array([0.1, 1.0, 2.0, 5.0, 8.0, np.inf], dtype=np.float32)
    _raw_level_weights = np.array([0.1, 0.1, 0.2, 0.25, 0.35], dtype=np.float32)
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
# Metrics Calculation
# ==========================================
def calc_seq_metrics(true_seq, pred_seq, verbose=True):
    """计算序列评分 (SimVP 风格)"""
    T, H, W = true_seq.shape
    pred_clean = pred_seq.copy()
    pred_clean[pred_clean < (MetricConfig.THRESHOLD_NOISE / MetricConfig.MM_MAX)] = 0.0
    
    tru_mm = np.clip(true_seq, 0, None) * MetricConfig.MM_MAX
    prd_mm = np.clip(pred_clean, 0, None) * MetricConfig.MM_MAX
    time_weights = MetricConfig.get_time_weights(T)
    
    score_list = []
    corr_sum = 0.0
    
    if verbose:
        print("-" * 80)
        print(f"{'T':<3}|{'Corr':<8}|{'Score':<8}")
        
    for t in range(T):
        tf, pf = tru_mm[t].flatten(), prd_mm[t].flatten()
        
        # Correlation
        mask = (tf>0)|(pf>0)
        R_k = np.corrcoef(tf[mask], pf[mask])[0,1] if mask.sum()>1 else 0.0
        if np.isnan(R_k): R_k=0.0
        R_k = np.clip(R_k, -1, 1)
        corr_sum += R_k
        
        # TS & MAE Weighted Sum
        w_sum = 0.0
        abs_err = np.abs(pf - tf)
        
        for i in range(len(MetricConfig.LEVEL_WEIGHTS)):
            l, h = MetricConfig.LEVEL_EDGES[i], MetricConfig.LEVEL_EDGES[i+1]
            t_bin, p_bin = (tf>=l)&(tf<h), (pf>=l)&(pf<h)
            tp, fn, fp = (t_bin&p_bin).sum(), (t_bin&~p_bin).sum(), (~t_bin&p_bin).sum()
            ts = tp / (tp+fn+fp+1e-8)
            
            mask_eval = t_bin | p_bin
            mae = np.mean(abs_err[mask_eval]) if mask_eval.sum()>0 else 0.0
            w_sum += MetricConfig.LEVEL_WEIGHTS[i] * ts * np.sqrt(np.exp(-mae/100.0))
            
        score = np.sqrt(np.exp(R_k-1.0)) * w_sum
        score_list.append(score)
        
        if verbose: print(f"{t:<3}|{R_k:<8.4f}|{score:<8.4f}")
        
    final = (np.array(score_list) * time_weights).sum()
    print("-" * 80)
    print(f"Final Weighted Score: {final:.6f}")
    print("-" * 80)
    
    return {
        "final_score": final,
        "pred_clean": pred_clean
    }

# ==========================================
# Visualization (SimVP Style with Diff)
# ==========================================
def plot_seq_visualization(obs_seq, true_seq, pred_seq, out_path, vmax=1.0):
    """绘制 Obs, GT, Pred, Diff"""
    T = true_seq.shape[0]
    rows, cols = 4, T
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5), constrained_layout=True)
    if T == 1: axes = axes[:, np.newaxis]

    for t in range(T):
        # 1. Obs
        if t < obs_seq.shape[0]:
            axes[0, t].imshow(obs_seq[t], cmap='turbo', vmin=0.0, vmax=vmax)
            axes[0, t].set_title(f'In-{t}', fontsize=6)
        else:
            axes[0, t].imshow(np.zeros_like(true_seq[0]), cmap='gray', vmin=0, vmax=1)
        axes[0, t].axis('off')
        if t == 0: axes[0, t].set_ylabel('Obs', fontsize=8)

        # 2. GT
        axes[1, t].imshow(true_seq[t], cmap='turbo', vmin=0.0, vmax=vmax)
        axes[1, t].axis('off')
        if t == 0: axes[1, t].set_ylabel('GT', fontsize=8)
        axes[1, t].set_title(f'T+{t+1}', fontsize=6)

        # 3. Pred
        axes[2, t].imshow(pred_seq[t], cmap='turbo', vmin=0.0, vmax=vmax)
        axes[2, t].axis('off')
        if t == 0: axes[2, t].set_ylabel('Pred', fontsize=8)
        
        # 4. Diff
        diff = true_seq[t] - pred_seq[t]
        axes[3, t].imshow(diff, cmap='bwr', vmin=-0.5, vmax=0.5)
        axes[3, t].axis('off')
        if t == 0: axes[3, t].set_ylabel('Diff', fontsize=8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

# ==========================================
# Main
# ==========================================
def render(obs_seq, true_seq, pred_seq, out_path):
    # Convert to numpy and select channel 0 (Radar/Precip)
    def to_numpy(x):
        if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
        if x.ndim == 4: x = x[:, 0] # (T, C, H, W) -> (T, H, W)
        return x

    obs, tru, prd = to_numpy(obs_seq), to_numpy(true_seq), to_numpy(pred_seq)
    
    print(f"Processing: {os.path.basename(out_path)}")
    metrics = calc_seq_metrics(tru, prd, verbose=True)
    plot_seq_visualization(obs, tru, metrics['pred_clean'], out_path)
    return metrics['final_score']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='output/meteo_mamba/vis')
    parser.add_argument('--in_shape', type=int, nargs=4, default=[10, 31, 256, 256])
    # [关键参数] 默认绘制前 10 个
    parser.add_argument('--num_samples', type=int, default=10) 
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"[INFO] Loading: {args.ckpt_path}")
    model = MeteoMambaModule.load_from_checkpoint(args.ckpt_path).to(device).eval()
    
    resize_shape = (args.in_shape[2], args.in_shape[3])
    dm = ScwdsDataModule(resize_shape=resize_shape, batch_size=1, num_workers=4)
    dm.setup('test')
    
    os.makedirs(args.save_dir, exist_ok=True)
    set_logger(os.path.join(args.save_dir, 'log.txt'))
    
    scores = []
    with torch.no_grad():
        for i, batch in enumerate(dm.test_dataloader()):
            # [关键逻辑] 只处理前 N 个样本
            if i >= args.num_samples: 
                print(f"[INFO] Reached num_samples limit ({args.num_samples}). Stopping.")
                break
                
            # Move batch to device (metadata is list, so skip it)
            batch_device = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            
            # Call updated test_step
            out = model.test_step(batch_device, i)
            
            # Render and Plot
            save_path = os.path.join(args.save_dir, f'sample_{i:03d}.png')
            s = render(out['inputs'], out['trues'], out['preds'], save_path)
            scores.append(s)
            
    if scores:
        print(f"\n[FINAL] Average Score ({len(scores)} samples): {np.mean(scores):.6f}")
    restore_stdout()

if __name__ == '__main__': main()