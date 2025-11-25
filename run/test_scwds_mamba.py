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

# --- 工具函数 (复用 simvp 测试脚本逻辑) ---
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

class MetricConfig:
    MM_MAX = 30.0
    THRESHOLD_NOISE = 0.05
    LEVEL_EDGES = np.array([0.0, 0.1, 1.0, 2.0, 5.0, 8.0, np.inf], dtype=np.float32)
    LEVEL_WEIGHTS = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.3], dtype=np.float32)
    LEVEL_WEIGHTS /= LEVEL_WEIGHTS.sum()
    TIME_WEIGHTS_DICT = {i: w for i, w in enumerate([0.0075, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.0075, 0.005])}
    @staticmethod
    def get_time_weights(T): return np.array([MetricConfig.TIME_WEIGHTS_DICT[t] for t in range(T)], dtype=np.float32) if T==20 else np.ones(T)/T

def calc_seq_metrics(true_seq, pred_seq, verbose=True):
    T, H, W = true_seq.shape
    pred_clean = pred_seq.copy()
    pred_clean[pred_clean < (MetricConfig.THRESHOLD_NOISE / MetricConfig.MM_MAX)] = 0.0
    tru_mm, prd_mm = true_seq * MetricConfig.MM_MAX, pred_clean * MetricConfig.MM_MAX
    time_weights = MetricConfig.get_time_weights(T)
    
    score_list, ts_levels, mae_levels = [], np.zeros(len(MetricConfig.LEVEL_WEIGHTS)), np.zeros(len(MetricConfig.LEVEL_WEIGHTS))
    corr_sum = 0.0
    
    if verbose: print(f"{'T':<3}|{'Corr':<8}|{'Score':<8}")
    for t in range(T):
        tf, pf = tru_mm[t].flatten(), prd_mm[t].flatten()
        # Corr
        mask = (tf>0)|(pf>0)
        if mask.sum()>1:
            t_v, p_v = tf[mask], pf[mask]
            R_k = np.corrcoef(t_v, p_v)[0,1]
            if np.isnan(R_k): R_k=0.0
        else: R_k = 0.0
        R_k = np.clip(R_k, -1, 1)
        corr_sum += R_k
        
        # Levels
        w_sum = 0.0
        for i in range(len(MetricConfig.LEVEL_WEIGHTS)):
            l, h = MetricConfig.LEVEL_EDGES[i], MetricConfig.LEVEL_EDGES[i+1]
            t_bin, p_bin = (tf>=l)&(tf<h), (pf>=l)&(pf<h)
            tp, fn, fp = (t_bin&p_bin).sum(), (t_bin&~p_bin).sum(), (~t_bin&p_bin).sum()
            ts = tp / (tp+fn+fp+1e-8)
            mask_eval = t_bin | p_bin
            mae = np.abs(pf[mask_eval]-tf[mask_eval]).mean() if mask_eval.sum()>0 else 0.0
            
            ts_levels[i]+=ts/T; mae_levels[i]+=mae/T
            w_sum += MetricConfig.LEVEL_WEIGHTS[i] * ts * np.sqrt(np.exp(-mae/100.0))
            
        score = np.sqrt(np.exp(R_k-1.0)) * w_sum
        score_list.append(score)
        if verbose: print(f"{t:<3}|{R_k:<8.4f}|{score:<8.4f}")
        
    final = (np.array(score_list) * time_weights).sum()
    print(f"Final: {final:.6f}")
    return final

def plot_seq(obs, tru, pred, path):
    T = tru.shape[0]
    fig, ax = plt.subplots(3, T, figsize=(T*1.5, 4.5))
    for t in range(T):
        if t < obs.shape[0]: ax[0,t].imshow(obs[t], cmap='turbo', vmin=0, vmax=1); ax[0,t].axis('off')
        else: ax[0,t].axis('off')
        ax[1,t].imshow(tru[t], cmap='turbo', vmin=0, vmax=1); ax[1,t].axis('off')
        ax[2,t].imshow(pred[t], cmap='turbo', vmin=0, vmax=1); ax[2,t].axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='output/meteo_mamba/vis')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MeteoMambaModule.load_from_checkpoint(args.ckpt_path).to(device).eval()
    dm = ScwdsDataModule(batch_size=1, num_workers=4); dm.setup('test')
    
    os.makedirs(args.save_dir, exist_ok=True)
    set_logger(os.path.join(args.save_dir, 'log.txt'))
    
    scores = []
    with torch.no_grad():
        for i, batch in enumerate(dm.test_dataloader()):
            if i >= 10: break
            batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            out = model.test_step(batch, i)
            
            print(f"Sample {i}")
            s = calc_seq_metrics(out['trues'], out['preds'])
            plot_seq(out['inputs'], out['trues'], out['preds'], os.path.join(args.save_dir, f'{i}.png'))
            scores.append(s)
            
    print(f"Avg Score: {np.mean(scores):.6f}")
    restore_stdout()

if __name__ == '__main__': main()