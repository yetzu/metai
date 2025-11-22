# run/gan_test_scwds_simvp.py
import sys
import os
import glob
import torch
import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
matplotlib.use('Agg')

from metai.dataset.met_dataloader_scwds import ScwdsDataModule
from metai.model.simvp import SimVP_GAN  # <--- 注意这里导入的是 GAN 模型

def find_best_ckpt(save_dir: str) -> str:
    """查找 GAN 训练生成的最佳 checkpoint"""
    # 优先找 best.ckpt
    best = os.path.join(save_dir, 'checkpoints', 'last.ckpt') # 优先看 last，因为 GAN 训练波动大
    if os.path.exists(best):
        return best
    
    # 否则找最新的
    ckpt_dir = os.path.join(save_dir, 'checkpoints')
    cpts = sorted(glob.glob(os.path.join(ckpt_dir, '*.ckpt')), key=os.path.getmtime)
    if len(cpts) == 0:
        # 尝试在上级目录找
        cpts = sorted(glob.glob(os.path.join(save_dir, '*.ckpt')), key=os.path.getmtime)
        if len(cpts) == 0:
            raise FileNotFoundError(f'No checkpoint found in {save_dir}')
    
    print(f"[INFO] Found checkpoints: {[os.path.basename(c) for c in cpts]}")
    return cpts[-1]

import numpy as np
import matplotlib.pyplot as plt
import os

def render(obs_seq, true_seq, pred_seq, out_path: str, vmax: float = 1.0):
    # obs_seq/true_seq/pred_seq: (T, C, H, W) or (T, 1, H, W) with same C
    T, C, H, W = obs_seq.shape
    ch = 0
    obs = obs_seq[:, ch]
    tru = true_seq[:, ch]
    prd = pred_seq[:, ch]

    print(f"tru: {np.max(tru)}, {np.min(tru)}, {np.mean(tru)}")
    print(f"prd: {np.max(prd)}, {np.min(prd)}, {np.mean(prd)}")

    # ---- 1. 定义时效权重 (表1) ----
    # 对应 1-20 个时效 (6min - 120min)
    time_weights_dict = {
        0: 0.0075, 1: 0.02, 2: 0.03, 3: 0.04, 4: 0.05,
        5: 0.06, 6: 0.07, 7: 0.08, 8: 0.09, 9: 0.1,
        10: 0.09, 11: 0.08, 12: 0.07, 13: 0.06, 14: 0.05,
        15: 0.04, 16: 0.03, 17: 0.02, 18: 0.0075, 19: 0.005
    }
    # 确保 T=20，如果 T 不同需做相应截取或处理
    if T == 20:
        time_weights = np.array([time_weights_dict[t] for t in range(T)], dtype=np.float32)
    else:
        # 如果 T 不是 20，暂时使用均值权重或报错，这里做归一化处理兜底
        print(f"Warning: T={T}, expected 20 for standard weights. Using uniform weights.")
        time_weights = np.ones(T, dtype=np.float32) / T

    # ---- 2. 定义要素分级及权重 (表2) ----
    # 表格中有 6 行：0mm, 0.1..., 1.0..., 2.0..., 5.0..., >=8mm
    # TS评分通常基于阈值（>=Threshold）。我们将"0mm"视为全场降水/无降水判断(>=0.01近似)
    # 阈值边缘：
    level_edges_mm = np.array([0.01, 0.1, 1.0, 2.0, 5.0, 8.0,], dtype=np.float32)
    # 对应权重：
    level_weights = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.3], dtype=np.float32)
    
    # 确保权重归一化 (虽然表里加起来已经是1.0)
    level_weights = level_weights / level_weights.sum()
    
    mm_max = 30.0  # 数据反归一化参数

    ts_collection = []
    mae_collection = []
    corr_collection = []
    score_k_collection = [] # 存储每个时刻的 Score_k

    tru_clipped = np.clip(tru, 0.0, None)
    prd_clipped = np.clip(prd, 0.0, None)

    for t in range(T):
        tru_frame = tru_clipped[t].reshape(-1)
        prd_frame = prd_clipped[t].reshape(-1)

        tru_mm = tru_frame * mm_max
        prd_mm = prd_frame * mm_max
        abs_err = np.abs(prd_mm - tru_mm)
        
        # ------------------------------------------------
        # 计算 R_k (公式 6)
        # ------------------------------------------------
        tru_mean = tru_frame.mean()
        prd_mean = prd_frame.mean()
        tru_center = tru_frame - tru_mean
        prd_center = prd_frame - prd_mean
        numerator = float(np.dot(tru_center, prd_center))
        sum_tru_sq = float(np.sum(tru_center ** 2))
        sum_prd_sq = float(np.sum(prd_center ** 2))
        sigma = 1e-6 
        denom = float(np.sqrt(sum_tru_sq * sum_prd_sq + sigma))
        
        if sum_tru_sq == 0 and sum_prd_sq == 0:
            corr_t = 1.0
        else:
            corr_t = numerator / denom if denom > 0 else 1e-5 # 若分母极小，相关性视为0
        corr_t = float(np.clip(corr_t, -1.0, 1.0))
        
        # ------------------------------------------------
        # 计算 TS_ik 和 MAE_ik (按等级循环)
        # ------------------------------------------------
        ts_t = []
        mae_t = []
        
        for idx, threshold in enumerate(level_edges_mm):
            # TS 计算: 基于阈值判别 (Thresholding)
            tru_bin = tru_mm >= threshold
            prd_bin = prd_mm >= threshold
            
            tp = np.logical_and(tru_bin, prd_bin).sum()
            fp = np.logical_and(~tru_bin, prd_bin).sum()
            fn = np.logical_and(tru_bin, ~prd_bin).sum()
            denom_ts = tp + fp + fn
            
            # 处理分母为0的情况 (无事件)
            current_ts = float(tp / denom_ts) if denom_ts > 0 else 1.0 # 若完全无事件，TS记为1(完全正确)
            ts_t.append(current_ts)

            # MAE 计算 (公式 5)
            # N_B: 样本中观测值或预报值大于0的格点数。
            # 在分级计算中，通常指该等级范围内的有效点，或者大于该阈值的点。
            # 结合TS语境，这里取大于该阈值的点作为有效点计算MAE
            mask_valid = (tru_mm >= threshold) | (prd_mm >= threshold)
            
            if np.any(mask_valid):
                current_mae = float(abs_err[mask_valid].mean())
            else:
                current_mae = 0.0 # 无有效点，MAE记为0
            mae_t.append(current_mae)

        ts_t = np.array(ts_t, dtype=np.float32)
        mae_t = np.array(mae_t, dtype=np.float32)

        # ------------------------------------------------
        # 计算 Score_k (公式 3)
        # Score_k = sqrt(exp(R_k - 1)) * Σ [ W_i * TS_ik * sqrt(exp(-MAE_ik / 100)) ]
        # ------------------------------------------------
        
        # 1. 相关性项
        term_corr = np.sqrt(np.exp(corr_t - 1.0))
        
        # 2. MAE 项: sqrt(exp(-MAE / 100))
        # 注意：公式中是 MAE_ik / 100，直接使用绝对误差值
        term_mae = np.sqrt(np.exp(-mae_t / 100.0))
        
        # 3. 加权求和
        # Sum [ W_i * TS_ik * MAE_term ]
        weighted_sum = np.sum(level_weights * ts_t * term_mae)
        
        score_t = float(term_corr * weighted_sum)

        ts_collection.append(ts_t)
        mae_collection.append(mae_t)
        corr_collection.append(corr_t)
        score_k_collection.append(score_t)

    ts_collection = np.stack(ts_collection, axis=0)
    mae_collection = np.stack(mae_collection, axis=0)
    corr_collection = np.array(corr_collection)
    score_k_collection = np.array(score_k_collection)

    # ---- 3. 计算最终加权总分 ----
    # Score = Σ (Score_k * Time_Weight_k)
    # 假设 time_weights 和 score_k_collection 长度一致
    final_score = float(np.sum(score_k_collection * time_weights))

    # 为了日志展示方便，保留均值计算
    ts_mean = ts_collection.mean(axis=0)
    mae_mean = mae_collection.mean(axis=0)
    corr_mean = float(corr_collection.mean())

    level_labels = [f">={edge}" for edge in level_edges_mm]

    print("[METRIC] TS_mean@" + "/".join(level_labels) + ": "
          + ", ".join(f"{v:.3f}" for v in ts_mean))
    print("[METRIC] MAE_mm_mean@" + "/".join(level_labels) + ": "
          + ", ".join(f"{v:.3f}" for v in mae_mean))
    print(f"[METRIC] Corr_mean: {corr_mean:.3f}")
    print(f"[METRIC] Final_Weighted_Score: {final_score:.4f}")
    
    # 每个时效得分输出
    print("[METRIC] Score_per_t: " + ", ".join(f"{v:.3f}" for v in score_k_collection))

    # ---- 绘图部分 (保持不变) ----
    rows, cols = 4, T
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), constrained_layout=True)
    for t in range(T):
        axes[0, t].imshow(obs[t], cmap='turbo', vmin=0.0, vmax=vmax)
        axes[0, t].axis('off')
        if t == 0: axes[0, t].set_title('Obs', fontsize=8)

        axes[1, t].imshow(tru[t], cmap='turbo', vmin=0.0, vmax=vmax)
        axes[1, t].axis('off')
        if t == 0: axes[1, t].set_title('GT', fontsize=8)

        axes[2, t].imshow(prd[t], cmap='turbo', vmin=0.0, vmax=vmax)
        axes[2, t].axis('off')
        if t == 0: axes[2, t].set_title('Pred', fontsize=8)
        
        diff = tru[t] - prd[t]
        axes[3, t].imshow( diff, cmap='bwr', vmin=-1, vmax=1) # 修改为bwr更能体现差异正负
        axes[3, t].axis('off')
        if t == 0: axes[3, t].set_title('Diff', fontsize=8)
            
    print(f'[INFO] Plot done, Saving to {out_path}')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    
    return final_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/samples.jsonl')
    parser.add_argument('--save_dir', type=str, default='./output/simvp_gan', help='Path to GAN checkpoint dir')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--accelerator', type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    device_str = args.device
    if args.accelerator == 'cuda': device_str = 'cuda'
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    
    print("="*50)
    print(f"[INFO] Starting GAN Testing")
    print(f" - Save Dir: {args.save_dir}")
    print("="*50)

    # 1. 查找并加载模型
    try:
        ckpt_path = find_best_ckpt(args.save_dir)
        print(f"[INFO] Loading GAN checkpoint: {ckpt_path}")
        
        # 加载 GAN 模型
        model = SimVP_GAN.load_from_checkpoint(ckpt_path, map_location=device)
        model.eval().to(device)
        
        # 获取 resize shape (从 backbone 中继承)
        resize_shape = model.backbone.resize_shape
        print(f"[INFO] Resize Shape from model: {resize_shape}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # 2. 数据加载 (batch_size=1)
    data_module = ScwdsDataModule(
        data_path=args.data_path,
        resize_shape=resize_shape, 
        batch_size=1, 
        num_workers=1,
        test_split=0.2,
    )
    data_module.setup(stage='test')
    test_loader = data_module.test_dataloader()

    # 3. 推理循环
    out_dir = os.path.join(args.save_dir, 'vis_results')
    os.makedirs(out_dir, exist_ok=True)
    
    score_mean_list = []
    processed_count = 0
    
    with torch.no_grad():
        for bidx, batch in enumerate(test_loader):
            metadata_batch, x_raw, y_raw, _ = batch
            x_raw = x_raw.to(device)
            y_raw = y_raw.to(device)
            
            # 同样需要手动插值，因为 GAN 的 training_step 里手动做了
            x = model.backbone._interpolate_batch_gpu(x_raw, mode='max_pool')
            y = model.backbone._interpolate_batch_gpu(y_raw, mode='max_pool')
            
            # 前向传播 (GAN forward 会自动调用 backbone + refiner)
            y_pred = model(x)
            
            obs_np = x[0].cpu().float().numpy()
            tru_np = y[0].cpu().float().numpy()
            prd_np = y_pred[0].cpu().float().numpy()
            
            sample_id = metadata_batch[0].get('sample_id', f'sample_{bidx}')
            out_path = os.path.join(out_dir, f'{sample_id}.png')
            
            print(f"Processing {bidx+1}/{args.num_samples}: {sample_id}")
            score = render(obs_np, tru_np, prd_np, out_path)
            score_mean_list.append(score)
            
            processed_count += 1
            if processed_count >= args.num_samples:
                break

    if len(score_mean_list) > 0:
        print(f"\n[FINAL] Average Score: {np.mean(score_mean_list):.6f}")

if __name__ == '__main__':
    main()