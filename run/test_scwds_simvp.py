# run/test_scwds_simvp.py
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
from metai.model.simvp import SimVPConfig, SimVP

# ==========================================
# Part 0: 辅助工具函数
# ==========================================

def create_diff_cmap():
    """创建中间色为 #30123B 的差异图 colormap"""
    middle_color = (0.188, 0.071, 0.231, 1.0)
    colors = [
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 0.3, 0.8, 1.0),
        middle_color,
        (0.9, 0.2, 0.0, 1.0),
        (1.0, 0.0, 0.0, 1.0),
    ]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('diff_turbo', colors, N=n_bins)
    return cmap

def find_best_ckpt(save_dir: str) -> str:
    # 优先查找 best.ckpt
    best = os.path.join(save_dir, 'best.ckpt')
    if os.path.exists(best): return best
    
    # 其次查找 last.ckpt
    last = os.path.join(save_dir, 'last.ckpt')
    if os.path.exists(last): return last
    
    # 最后查找所有 checkpoint 文件，返回最新的
    cpts = sorted(glob.glob(os.path.join(save_dir, '*.ckpt')))
    if len(cpts) == 0:
        raise FileNotFoundError(f'No checkpoint found in {save_dir}')
    return cpts[-1]

def print_checkpoint_info(ckpt_path: str):
    """从 checkpoint 文件中提取并打印训练关键信息"""
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        print("=" * 80)
        print(f"[INFO] Loaded Checkpoint: {os.path.basename(ckpt_path)}")
        print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
        print(f"  Global Step: {ckpt.get('global_step', 'N/A')}")
        
        hparams = ckpt.get('hyper_parameters', {})
        if hparams:
            print(f"  Model Type: {hparams.get('model_type', 'N/A')}")
            print(f"  Hidden Dim (T): {hparams.get('hid_T', 'N/A')}")
        print("=" * 80)
    except Exception as e:
        print(f"[WARNING] 无法读取 checkpoint 信息: {e}")

# ==========================================
# Part 1: 全局评分配置 (Metric Configuration)
# ==========================================
class MetricConfig:
    # 反归一化参数 (假设数据归一化时除以了30)
    MM_MAX = 30.0
    
    # 噪音阈值 (小于此值的预测视为0，用于拿到 0mm 档的 0.1 分)
    # 0.05mm 是一个经验值，既能过滤浮点噪声，又不会误杀有效小雨
    THRESHOLD_NOISE = 0.05 
    
    # 阈值边缘 (mm) - 对应表2
    # 区间: [0, 0.1), [0.1, 1.0), [1.0, 2.0), [2.0, 5.0), [5.0, 8.0), [8.0, inf)
    LEVEL_EDGES = np.array([0.0, 0.1, 1.0, 2.0, 5.0, 8.0, np.inf], dtype=np.float32)
    
    # 降水等级权重 (表2)
    _raw_level_weights = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.3], dtype=np.float32)
    # 归一化权重 (虽然原始和已为1，但保险起见)
    LEVEL_WEIGHTS = _raw_level_weights / _raw_level_weights.sum()

    # 时效权重 (表1) - 对应索引 0-19 (6min - 120min)
    TIME_WEIGHTS_DICT = {
        0: 0.0075, 1: 0.02, 2: 0.03, 3: 0.04, 4: 0.05,
        5: 0.06, 6: 0.07, 7: 0.08, 8: 0.09, 9: 0.1,
        10: 0.09, 11: 0.08, 12: 0.07, 13: 0.06, 14: 0.05,
        15: 0.04, 16: 0.03, 17: 0.02, 18: 0.0075, 19: 0.005
    }

    @staticmethod
    def get_time_weights(T):
        """根据序列长度T获取时效权重"""
        if T == 20:
            return np.array([MetricConfig.TIME_WEIGHTS_DICT[t] for t in range(T)], dtype=np.float32)
        else:
            # 如果 T 不是 20，使用均匀权重兜底
            print(f"[WARN] T={T}, expected 20. Using uniform time weights.")
            return np.ones(T, dtype=np.float32) / T

# ==========================================
# Part 2: 核心统计计算 (Core Metrics)
# ==========================================
def calc_seq_metrics(true_seq, pred_seq, verbose=True):
    """
    计算序列的降水评分指标 (Strict implementation of Contest Rules)
    """
    T, H, W = true_seq.shape
    
    # 1. 预处理：噪音过滤 (Post-Processing)
    # 这一步对于获得 0mm 档的 0.1 分至关重要
    pred_clean = pred_seq.copy()
    # 将归一化前的数值小于阈值的置为0
    # 注意：pred_seq 是归一化后的 [0,1]，THRESHOLD_NOISE 是 mm
    pred_clean[pred_clean < (MetricConfig.THRESHOLD_NOISE / MetricConfig.MM_MAX)] = 0.0
    
    time_weights = MetricConfig.get_time_weights(T)
    score_k_list = []
    
    # 反归一化到 mm 并截断负值
    tru_mm_seq = np.clip(true_seq, 0.0, None) * MetricConfig.MM_MAX
    prd_mm_seq = np.clip(pred_clean, 0.0, None) * MetricConfig.MM_MAX
    
    # 用于日志输出的累积变量
    ts_mean_levels = np.zeros(len(MetricConfig.LEVEL_WEIGHTS))
    mae_mm_mean_levels = np.zeros(len(MetricConfig.LEVEL_WEIGHTS))
    corr_sum = 0.0

    if verbose:
        print(f"True Stats: Max={np.max(tru_mm_seq):.2f}, Mean={np.mean(tru_mm_seq):.2f}")
        print(f"Pred Stats: Max={np.max(prd_mm_seq):.2f}, Mean={np.mean(prd_mm_seq):.2f}")
        print("-" * 80)
        print(f"{'T':<3} | {'Corr(R)':<9} | {'TS_w_sum':<9} | {'Score_k':<9}")
        print("-" * 80)

    for t in range(T):
        tru_frame = tru_mm_seq[t].reshape(-1)
        prd_frame = prd_mm_seq[t].reshape(-1)
        abs_err = np.abs(prd_frame - tru_frame)

        # --- A. 计算相关系数 R_k (公式 6) ---
        # 规则：观测值和预测值同时为0的格点，不参与计算
        mask_valid_corr = (tru_frame > 0) | (prd_frame > 0)
        
        if mask_valid_corr.sum() > 1:
            t_valid = tru_frame[mask_valid_corr]
            p_valid = prd_frame[mask_valid_corr]
            
            numerator = np.sum((t_valid - t_valid.mean()) * (p_valid - p_valid.mean()))
            denom = np.sqrt(np.sum((t_valid - t_valid.mean())**2) * np.sum((p_valid - p_valid.mean())**2))
            
            R_k = numerator / (denom + 1e-8)
        else:
            # 如果全图双零，或只有一个点非零，无法计算相关性
            # 视情况给 0.0 (保守)
            R_k = 0.0
            
        R_k = float(np.clip(R_k, -1.0, 1.0))
        corr_sum += R_k

        # 计算 Score 公式第一项: sqrt(exp(R - 1))
        term_corr = np.sqrt(np.exp(R_k - 1.0))

        # --- B. 逐等级计算 TS 和 MAE ---
        weighted_sum_metrics = 0.0
        
        for i in range(len(MetricConfig.LEVEL_WEIGHTS)):
            low = MetricConfig.LEVEL_EDGES[i]
            high = MetricConfig.LEVEL_EDGES[i+1]
            w_i = MetricConfig.LEVEL_WEIGHTS[i]

            # 生成 Mask: [low, high)
            # i=0 时对应 [0, 0.1)，即 0mm 档
            tru_bin = (tru_frame >= low) & (tru_frame < high)
            prd_bin = (prd_frame >= low) & (prd_frame < high)
            
            # TS 计算
            tp = np.logical_and(tru_bin, prd_bin).sum()
            fn = np.logical_and(tru_bin, ~prd_bin).sum() # 漏报
            fp = np.logical_and(~tru_bin, prd_bin).sum() # 空报 (虚警)
            
            denom_ts = tp + fn + fp
            # 特殊处理：分母为0说明该等级既没发生也没预报，视为正确(1.0)
            ts_val = (tp / denom_ts) if denom_ts > 0 else 1.0
            
            # MAE 计算 (公式 5)
            # 计算范围：命中该等级区域的并集 (只要真值或预报有一个落在该等级区间，就算误差)
            mask_eval = tru_bin | prd_bin
            if mask_eval.sum() > 0:
                mae_val = np.mean(abs_err[mask_eval])
            else:
                mae_val = 0.0
            
            # 统计累加 (用于最终打印平均值)
            ts_mean_levels[i] += ts_val / T
            mae_mm_mean_levels[i] += mae_val / T
            
            # Score 公式第二项: sqrt(exp(-MAE/100))
            term_mae = np.sqrt(np.exp(-mae_val / 100.0))
            
            # 加权累加
            weighted_sum_metrics += w_i * ts_val * term_mae

        # --- C. 计算当前时刻 Score_k ---
        Score_k = term_corr * weighted_sum_metrics
        score_k_list.append(Score_k)

        if verbose:
            # 仅打印部分信息防止刷屏
            print(f"{t:<3} | {R_k:<9.4f} | {weighted_sum_metrics:<9.4f} | {Score_k:<9.4f}")

    # ---- 4. 计算最终加权总分 ----
    score_k_arr = np.array(score_k_list)
    final_score = np.sum(score_k_arr * time_weights)
    
    # 打印汇总日志
    print("-" * 80)
    labels = ["0mm", "0.1-1", "1-2", "2-5", "5-8", ">=8"]
    
    ts_str = ", ".join([f"{v:.3f}" for v in ts_mean_levels])
    mae_str = ", ".join([f"{v:.3f}" for v in mae_mm_mean_levels])
    
    print(f"[METRIC] TS_mean  (Levels): {ts_str}")
    print(f"[METRIC] MAE_mean (Levels): {mae_str}")
    print(f"[METRIC] Corr_mean: {corr_sum / T:.4f}")
    print(f"[METRIC] Final_Weighted_Score: {final_score:.6f}")
    print("-" * 80)
    
    return {
        "final_score": final_score,
        "score_per_frame": score_k_arr,
        "pred_clean": pred_clean # 返回给绘图用
    }

# ==========================================
# Part 3: 绘图功能 (Visualization)
# ==========================================
def plot_seq_visualization(obs_seq, true_seq, pred_seq, scores, out_path, vmax=1.0):
    """绘制 Obs, GT, Pred, Diff 对比图"""
    T = true_seq.shape[0]
    rows, cols = 4, T
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5), constrained_layout=True)
    if T == 1: axes = axes[:, np.newaxis]

    for t in range(T):
        # 1. Obs
        if t < obs_seq.shape[0]:
            axes[0, t].imshow(obs_seq[t], cmap='turbo', vmin=0.0, vmax=vmax)
        else:
            axes[0, t].imshow(np.zeros_like(true_seq[0]), cmap='gray')
        axes[0, t].axis('off')
        if t == 0: axes[0, t].set_title('Obs', fontsize=8)

        # 2. GT
        axes[1, t].imshow(true_seq[t], cmap='turbo', vmin=0.0, vmax=vmax)
        axes[1, t].axis('off')
        if t == 0: axes[1, t].set_title('GT', fontsize=8)

        # 3. Pred
        axes[2, t].imshow(pred_seq[t], cmap='turbo', vmin=0.0, vmax=vmax)
        axes[2, t].axis('off')
        if t == 0: axes[2, t].set_title('Pred', fontsize=8)
        
        # 4. Diff (GT - Pred)
        diff = true_seq[t] - pred_seq[t]
        axes[3, t].imshow(diff, cmap='bwr', vmin=-0.5, vmax=0.5)
        axes[3, t].axis('off')
        if t == 0: axes[3, t].set_title('Diff', fontsize=8)

    print(f'[INFO] Saving Plot to {out_path}')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

# ==========================================
# Part 4: 主入口函数 (Wrapper)
# ==========================================
def render(obs_seq, true_seq, pred_seq, out_path: str, vmax: float = 1.0):
    # 1. 数据格式统一 (转 Numpy & 提取通道)
    def to_numpy_ch(x, ch=0):
        if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
        if x.ndim == 4: x = x[:, ch] # (T, C, H, W) -> (T, H, W)
        return x

    obs = to_numpy_ch(obs_seq)
    tru = to_numpy_ch(true_seq)
    prd = to_numpy_ch(pred_seq)
    
    print(f"Processing: {os.path.basename(out_path)}")
    
    # 2. 调用统计模块
    metrics_res = calc_seq_metrics(tru, prd, verbose=True)
    
    final_score = metrics_res['final_score']
    
    # 3. 调用绘图模块
    plot_seq_visualization(obs, tru, metrics_res['pred_clean'], metrics_res['score_per_frame'], out_path, vmax=vmax)
    
    return final_score

def parse_args():
    parser = argparse.ArgumentParser(description='Test SCWDS SimVP Model')
    parser.add_argument('--data_path', type=str, default='data/samples.jsonl')
    parser.add_argument('--in_shape', type=int, nargs=4, default=[20, 28, 128, 128])
    parser.add_argument('--save_dir', type=str, default='./output/simvp')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--accelerator', type=str, default='cuda')
    parser.add_argument('--device', type=str, default='cuda') 
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.accelerator if torch.cuda.is_available() else 'cpu')
    
    print(f"[INFO] Starting Test on {device}")
    
    # 1. Config & Model
    config = SimVPConfig(
        data_path=args.data_path,
        in_shape=tuple(args.in_shape),
        save_dir=args.save_dir
    )
    resize_shape = (config.in_shape[2], config.in_shape[3])
    
    ckpt_path = find_best_ckpt(config.save_dir)
    print_checkpoint_info(ckpt_path)
    
    # 加载模型
    model = SimVP.load_from_checkpoint(ckpt_path)
    model.eval().to(device)
    
    # 2. Data
    data_module = ScwdsDataModule(
        data_path=config.data_path,
        resize_shape=resize_shape,
        batch_size=1,
        num_workers=4
    )
    data_module.setup('test')
    test_loader = data_module.test_dataloader()
    
    # 3. Loop
    out_dir = os.path.join(config.save_dir, 'vis_final_v2')
    os.makedirs(out_dir, exist_ok=True)
    
    scores = []
    
    with torch.no_grad():
        for bidx, batch in enumerate(test_loader):
            metadata_batch, batch_x, batch_y, target_mask, input_mask = batch
            
            # Inference
            outputs = model.test_step(
                (metadata_batch, batch_x.to(device), batch_y.to(device), target_mask.to(device), input_mask.to(device)), 
                bidx
            )
            
            # Render
            save_path = os.path.join(out_dir, f'sample_{bidx:03d}.png')
            s = render(outputs['inputs'], outputs['trues'], outputs['preds'], save_path)
            scores.append(s)
            
            if bidx >= args.num_samples - 1:
                break
    
    if len(scores) > 0:
        print(f"\n[FINAL] Average Score ({len(scores)} samples): {np.mean(scores):.6f}")

if __name__ == '__main__':
    main()