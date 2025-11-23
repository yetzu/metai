# run/gan_test_scwds_simvp.py
import sys
import os
import glob
import torch
import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np
from datetime import datetime
from timm.layers.drop import DropPath 

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
matplotlib.use('Agg')

from metai.dataset.met_dataloader_scwds import ScwdsDataModule
from metai.model.simvp.simvp_gan import SimVP_GAN 

# ==========================================
# Part 1: 全局配置
# ==========================================
class MetricConfig:
    MM_MAX = 30.0
    THRESHOLD_NOISE = 0.1 / 30.0
    LEVEL_EDGES = np.array([0.01, 0.1, 1.0, 2.0, 5.0, 8.0], dtype=np.float32)
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

def find_best_ckpt(save_dir: str) -> str:
    ckpt_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(ckpt_dir): ckpt_dir = save_dir 
    
    last = os.path.join(ckpt_dir, 'last.ckpt')
    if os.path.exists(last):
        print(f"[INFO] Found last checkpoint: {last}")
        return last
    
    cpts = sorted(glob.glob(os.path.join(ckpt_dir, '*.ckpt')), key=os.path.getmtime)
    if len(cpts) > 0:
        print(f"[INFO] Found checkpoints in {ckpt_dir}: {[os.path.basename(c) for c in cpts]}")
        return cpts[-1]

    if len(cpts) == 0:
        raise FileNotFoundError(f'No checkpoint found in {save_dir} or {ckpt_dir}')
    return cpts[-1]

def get_checkpoint_info(ckpt_path: str):
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        epoch = ckpt.get('epoch', None)
        return {'epoch': epoch}
    except Exception:
        return {'epoch': 0}

def enable_dropout(m):
    if type(m) in [torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d, DropPath]:
        m.train()

# ==========================================
# Part 2: 统计指标计算函数
# ==========================================
def calc_seq_metrics(true_seq, pred_seq, verbose=True):
    T, H, W = true_seq.shape
    pred_clean = pred_seq.copy()
    pred_clean[pred_clean < MetricConfig.THRESHOLD_NOISE] = 0.0
    
    time_weights = MetricConfig.get_time_weights(T)
    score_k_list = []
    
    ts_mean_levels = np.zeros(len(MetricConfig.LEVEL_EDGES))
    mae_mm_mean_levels = np.zeros(len(MetricConfig.LEVEL_EDGES))
    corr_mean_sum = 0.0

    tru_clipped = np.clip(true_seq, 0.0, None)
    prd_clipped = np.clip(pred_clean, 0.0, None)

    true_nz_ratio = np.mean(tru_clipped > MetricConfig.THRESHOLD_NOISE) * 100
    pred_nz_ratio = np.mean(prd_clipped > MetricConfig.THRESHOLD_NOISE) * 100

    print(f"True Stats: Max={np.max(tru_clipped)*MetricConfig.MM_MAX:.2f}, Mean={np.mean(tru_clipped)*MetricConfig.MM_MAX:.2f}, RainRatio={true_nz_ratio:.2f}%")
    print(f"Pred Stats: Max={np.max(prd_clipped)*MetricConfig.MM_MAX:.2f}, Mean={np.mean(prd_clipped)*MetricConfig.MM_MAX:.2f}, RainRatio={pred_nz_ratio:.2f}%")

    if verbose:
        print("-" * 90)
        print(f"{'T':<3} | {'Corr(R)':<9} | {'TS_w_sum':<9} | {'Score_k':<9} | {'W_time':<6}")
        print("-" * 90)

    for t in range(T):
        tru_frame = tru_clipped[t].reshape(-1)
        prd_frame = prd_clipped[t].reshape(-1)
        tru_mm = tru_frame * MetricConfig.MM_MAX
        prd_mm = prd_frame * MetricConfig.MM_MAX
        abs_err = np.abs(prd_mm - tru_mm)

        tru_mean = tru_frame.mean()
        prd_mean = prd_frame.mean()
        tru_center = tru_frame - tru_mean
        prd_center = prd_frame - prd_mean
        numerator = np.dot(tru_center, prd_center)
        sum_tru_sq = np.sum(tru_center ** 2)
        sum_prd_sq = np.sum(prd_center ** 2)
        denom = np.sqrt(sum_tru_sq * sum_prd_sq) + 1e-8
        
        if sum_tru_sq == 0 and sum_prd_sq == 0: corr_t = 1.0
        elif sum_tru_sq == 0 or sum_prd_sq == 0: corr_t = 0.0
        else: corr_t = numerator / denom
        corr_t = float(np.clip(corr_t, -1.0, 1.0))
        corr_mean_sum += corr_t

        ts_levels = []
        mae_term_levels = [] 
        for i, threshold in enumerate(MetricConfig.LEVEL_EDGES):
            tru_bin = tru_mm >= threshold
            prd_bin = prd_mm >= threshold
            tp = np.logical_and(tru_bin, prd_bin).sum()
            fp = np.logical_and(~tru_bin, prd_bin).sum()
            fn = np.logical_and(tru_bin, ~prd_bin).sum()
            denom_ts = tp + fp + fn
            ts_val = (tp / denom_ts) if denom_ts > 0 else 1.0
            ts_levels.append(ts_val)

            mask_valid = (tru_mm >= threshold) | (prd_mm >= threshold)
            current_mae = abs_err[mask_valid].mean() if mask_valid.sum() > 0 else 0.0
            mae_term = np.sqrt(np.exp(-current_mae / 100.0))
            mae_term_levels.append(mae_term)
            
            ts_mean_levels[i] += ts_val / T
            mae_mm_mean_levels[i] += current_mae / T 

        ts_levels = np.array(ts_levels)
        mae_term_levels = np.array(mae_term_levels)

        term_corr = np.sqrt(np.exp(corr_t - 1.0))
        term_weighted = np.sum(MetricConfig.LEVEL_WEIGHTS * ts_levels * mae_term_levels)
        score_k = term_corr * term_weighted
        score_k_list.append(score_k)

        if verbose:
            ts_show = np.sum(ts_levels * MetricConfig.LEVEL_WEIGHTS)
            print(f"{t:<3} | {corr_t:<9.4f} | {ts_show:<9.4f} | {score_k:<9.4f} | {time_weights[t]:<6.4f}")

    score_k_arr = np.array(score_k_list)
    final_score = np.sum(score_k_arr * time_weights)
    corr_mean = corr_mean_sum / T

    print("-" * 90)
    print(f"[METRIC] TS_mean  (Levels): {', '.join([f'{v:.3f}' for v in ts_mean_levels])}")
    print(f"[METRIC] MAE_mean (Levels): {', '.join([f'{v:.3f}' for v in mae_mm_mean_levels])}")
    print(f"[METRIC] Corr_mean: {corr_mean:.4f}")
    print(f"[METRIC] Final_Weighted_Score: {final_score:.6f}")
    print(f"[METRIC] Score_per_t: {', '.join([f'{s:.3f}' for s in score_k_arr])}")
    print("-" * 90)
    return final_score

# ==========================================
# Part 3: 绘图功能
# ==========================================
def plot_seq_visualization(obs_seq, true_seq, pred_seq, scores, out_path, vmax=1.0):
    T = true_seq.shape[0]
    rows = 4
    cols = T
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5), constrained_layout=True)
    if T == 1: axes = axes[:, np.newaxis]

    for t in range(T):
        if t < obs_seq.shape[0]:
            axes[0, t].imshow(obs_seq[t], cmap='turbo', vmin=0.0, vmax=vmax)
        else: axes[0, t].imshow(np.zeros_like(true_seq[0]), cmap='gray')
        axes[0, t].axis('off')
        if t == 0: axes[0, t].set_title('Obs', fontsize=8)

        axes[1, t].imshow(true_seq[t], cmap='turbo', vmin=0.0, vmax=vmax)
        axes[1, t].axis('off')
        if t == 0: axes[1, t].set_title('GT', fontsize=8)

        axes[2, t].imshow(pred_seq[t], cmap='turbo', vmin=0.0, vmax=vmax)
        axes[2, t].axis('off')
        if t == 0: axes[2, t].set_title('Pred (GAN)', fontsize=8)
        
        diff = true_seq[t] - pred_seq[t]
        axes[3, t].imshow(diff, cmap='bwr', vmin=-0.5, vmax=0.5)
        axes[3, t].axis('off')
        if t == 0: axes[3, t].set_title('Diff', fontsize=8)

    print(f'[INFO] Saving Plot to {out_path}')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

# ==========================================
# Part 4: 主流程
# ==========================================
def render(obs_seq, true_seq, pred_seq, out_path: str, vmax: float = 1.0):
    def to_numpy_ch(x, ch=0):
        if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
        if x.ndim == 4: x = x[:, ch]
        return x

    obs = to_numpy_ch(obs_seq)
    tru = to_numpy_ch(true_seq)
    prd = to_numpy_ch(pred_seq)
    
    print(f"Processing: {os.path.basename(out_path)}")
    final_score = calc_seq_metrics(tru, prd, verbose=True)
    plot_seq_visualization(obs, tru, prd, np.zeros(tru.shape[0]), out_path, vmax=vmax)
    return final_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/samples.jsonl')
    parser.add_argument('--save_dir', type=str, default='./output/simvp_gan')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--backbone_ckpt_path', type=str, default='./output/simvp/last.ckpt')
    parser.add_argument('--tta', type=int, default=1, help='TTA Steps')
    parser.add_argument('--gan_ckpt_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--accelerator', type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    device_str = args.device
    if args.accelerator == 'cuda': device_str = 'cuda'
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print(f"[INFO] Starting GAN Testing")
    print(f" - Save Dir: {args.save_dir}")
    print(f" - TTA Steps: {args.tta}")
    print("="*80)

    try:
        if args.gan_ckpt_path and os.path.exists(args.gan_ckpt_path):
            ckpt_path = args.gan_ckpt_path
        else:
            ckpt_path = find_best_ckpt(args.save_dir)
            
        print(f"[INFO] Loading GAN checkpoint: {ckpt_path}")
        
        if not os.path.exists(args.backbone_ckpt_path):
             potential = os.path.join(os.path.dirname(args.save_dir), 'simvp', 'last.ckpt')
             if os.path.exists(potential): args.backbone_ckpt_path = potential

        model = SimVP_GAN.load_from_checkpoint(
            ckpt_path, 
            map_location=device,
            backbone_ckpt_path=args.backbone_ckpt_path
        )
        model.eval().to(device)
        
        if args.tta > 1:
            print(f"[INFO] TTA Enabled ({args.tta}x). Enabling Dropout.")
            model.apply(enable_dropout)
        
        resize_shape = model.backbone.resize_shape
        print(f"[INFO] Resize Shape: {resize_shape}")
        
    except Exception as e:
        print(f"[ERROR] Load failed: {e}")
        return

    data_module = ScwdsDataModule(
        data_path=args.data_path,
        resize_shape=resize_shape, 
        batch_size=1, 
        num_workers=1,
        test_split=0.2, 
    )
    data_module.setup(stage='test')
    test_loader = data_module.test_dataloader()

    ckpt_info = get_checkpoint_info(ckpt_path)
    epoch = ckpt_info.get('epoch', 0)
    
    folder_name = f'vis_{epoch}_tta{args.tta}'
    out_dir = os.path.join(args.save_dir, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    
    log_file = os.path.join(out_dir, 'log.txt')
    
    score_mean_list = []
    processed_count = 0
    
    with open(log_file, 'w') as f:
        class Tee(object):
            def __init__(self, *files): self.files = files
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush() 
            def flush(self):
                for f in self.files: f.flush()
        
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, f)

        try:
            with torch.no_grad():
                for bidx, batch in enumerate(test_loader):
                    metadata_batch, x_raw, y_raw, _, _ = batch 
                    x_raw = x_raw.to(device)
                    y_raw = y_raw.to(device)
                    x = model.backbone._interpolate_batch_gpu(x_raw, mode='max_pool')
                    y = model.backbone._interpolate_batch_gpu(y_raw, mode='max_pool')
                    
                    if args.tta > 1:
                        preds = [model(x) for _ in range(args.tta)]
                        y_pred = torch.stack(preds).mean(dim=0)
                    else:
                        y_pred = model(x)
                    
                    obs_np = x[0].cpu().float()
                    tru_np = y[0].cpu().float()
                    prd_np = y_pred[0].cpu().float()
                    
                    # 🚨 关键修改：使用 sample_id 命名文件
                    sample_id = metadata_batch[0].get('sample_id', f'sample_{bidx:03d}')
                    print(f"Processing {bidx+1}/{args.num_samples}: {sample_id}")
                    
                    out_path = os.path.join(out_dir, f'sample_{bidx:03d}.png')
                    score = render(obs_np, tru_np, prd_np, out_path)
                    score_mean_list.append(score)
                    
                    processed_count += 1
                    if processed_count >= args.num_samples:
                        break

            if len(score_mean_list) > 0:
                print(f"\n[FINAL] Average Score: {np.mean(score_mean_list):.6f}")
                print(f"[INFO] Visualizations saved to: {out_dir}")
                
        finally:
            sys.stdout = original_stdout

if __name__ == '__main__':
    main()