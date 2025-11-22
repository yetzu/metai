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

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
matplotlib.use('Agg')

from metai.dataset.met_dataloader_scwds import ScwdsDataModule
from metai.model.simvp import SimVPConfig
from metai.model.simvp import SimVP

# 创建中间色为 #30123B 的差异图 colormap
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


# ==========================================
# Part 1: 全局配置 (Configuration)
# ==========================================
class MetricConfig:
    # 反归一化参数
    MM_MAX = 30.0
    # 噪音阈值
    THRESHOLD_NOISE = 0.1 / 30.0
    
    # 阈值边缘 (mm)
    LEVEL_EDGES = np.array([0.01, 0.1, 1.0, 2.0, 5.0, 8.0], dtype=np.float32)
    
    # 降水等级权重 (需归一化)
    _raw_level_weights = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.3], dtype=np.float32)
    LEVEL_WEIGHTS = _raw_level_weights / _raw_level_weights.sum()

    # 时效权重 (0-19)
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
            return np.ones(T, dtype=np.float32) / T

# ==========================================
# Part 2: 统计指标计算函数 (Metrics)
# ==========================================
def calc_seq_metrics(true_seq, pred_seq, verbose=True):
    """
    计算序列的降水评分指标
    ... (内部逻辑保持不变，假设TS, MAE, Corr计算正确)
    """
    T, H, W = true_seq.shape
    
    # 预处理：噪音过滤
    pred_clean = pred_seq.copy()
    pred_clean[pred_clean < MetricConfig.THRESHOLD_NOISE] = 0.0
    
    time_weights = MetricConfig.get_time_weights(T)
    score_k_list = []
    
    # 用于日志输出的累积变量 (为简洁省略其初始化，但应在代码中存在)
    ts_mean_levels = np.zeros(len(MetricConfig.LEVEL_EDGES))
    mae_mm_mean_levels = np.zeros(len(MetricConfig.LEVEL_EDGES))
    corr_mean_sum = 0.0

    # 截断负值
    tru_clipped = np.clip(true_seq, 0.0, None)
    prd_clipped = np.clip(pred_clean, 0.0, None)

    print(f"tru: {np.max(tru_clipped)}, {np.min(tru_clipped)}, {np.mean(tru_clipped)}")
    print(f"prd: {np.max(prd_clipped)}, {np.min(prd_clipped)}, {np.mean(prd_clipped)}")

    if verbose:
        print("-" * 80)
        print(f"Metric Calculation | T={T} | True Max: {true_seq.max()*MetricConfig.MM_MAX:.2f}mm")
        print(f"{'T':<3} | {'Corr(R_k)':<9} | {'TS_w_sum':<9} | {'MAE_w_sum':<9} | {'Score_k':<9} | {'W_time':<6}")
        print("-" * 80)

    for t in range(T):
        # 1. 准备数据 (转为mm)
        tru_frame = tru_clipped[t].reshape(-1)
        prd_frame = prd_clipped[t].reshape(-1)
        tru_mm = tru_frame * MetricConfig.MM_MAX
        prd_mm = prd_frame * MetricConfig.MM_MAX
        abs_err = np.abs(prd_mm - tru_mm)

        # 2. 计算 Corr (R_k) (公式 6)
        tru_mean = tru_frame.mean()
        prd_mean = prd_frame.mean()
        tru_center = tru_frame - tru_mean
        prd_center = prd_frame - prd_mean
        
        numerator = np.dot(tru_center, prd_center)
        sum_tru_sq = np.sum(tru_center ** 2)
        sum_prd_sq = np.sum(prd_center ** 2)
        denom = np.sqrt(sum_tru_sq * sum_prd_sq) + 1e-8
        
        if sum_tru_sq == 0 and sum_prd_sq == 0:
            corr_t = 1.0
        elif sum_tru_sq == 0 or sum_prd_sq == 0:
            corr_t = 0.0
        else:
            corr_t = numerator / denom
        corr_t = float(np.clip(corr_t, -1.0, 1.0))
        corr_mean_sum += corr_t # 累加相关系数

        # 3. 逐等级计算 TS 和 MAE
        ts_levels = []
        mae_term_levels = [] 

        for i, threshold in enumerate(MetricConfig.LEVEL_EDGES):
            tru_bin = tru_mm >= threshold
            prd_bin = prd_mm >= threshold
            
            # TS 计算
            tp = np.logical_and(tru_bin, prd_bin).sum()
            fp = np.logical_and(~tru_bin, prd_bin).sum()
            fn = np.logical_and(tru_bin, ~prd_bin).sum()
            denom_ts = tp + fp + fn
            ts_val = (tp / denom_ts) if denom_ts > 0 else 1.0
            ts_levels.append(ts_val)

            # MAE 计算 (有效区域)
            mask_valid = (tru_mm >= threshold) | (prd_mm >= threshold)
            current_mae = abs_err[mask_valid].mean() if mask_valid.sum() > 0 else 0.0
            
            # MAE Term (公式 3 右侧 exp 项)
            mae_term = np.sqrt(np.exp(-current_mae / 100.0))
            mae_term_levels.append(mae_term)
            
            # 累加用于最终平均值的统计 (匹配旧日志格式)
            ts_mean_levels[i] += ts_val / T
            mae_mm_mean_levels[i] += current_mae / T 


        ts_levels = np.array(ts_levels)
        mae_term_levels = np.array(mae_term_levels)

        # 4. 组合 Score_k (公式 3)
        term_corr = np.sqrt(np.exp(corr_t)) # 假设 Ra_k = corr_t - 1.0 + 1.0 = corr_t
        term_weighted = np.sum(MetricConfig.LEVEL_WEIGHTS * ts_levels * mae_term_levels)
        score_k = term_corr * term_weighted
        score_k_list.append(score_k)

        # 5. 打印单帧日志 (保持原样)
        if verbose:
            ts_show = np.sum(ts_levels * MetricConfig.LEVEL_WEIGHTS)
            mae_show = np.sum(mae_term_levels * MetricConfig.LEVEL_WEIGHTS)
            print(f"{t:<3} | {corr_t:<9.4f} | {ts_show:<9.4f} | {mae_show:<9.4f} | {score_k:<9.4f} | {time_weights[t]:<6.4f}")

    score_k_arr = np.array(score_k_list)
    final_score = np.sum(score_k_arr * time_weights)

    # 打印匹配旧日志格式的统计信息
    corr_mean = corr_mean_sum / T
    print("-" * 80)
    # 打印格式与旧日志保持一致
    print(f"[METRIC] TS_mean@>=0.01/...: {ts_mean_levels[0]:.3f}, {ts_mean_levels[1]:.3f}, {ts_mean_levels[2]:.3f}, {ts_mean_levels[3]:.3f}, {ts_mean_levels[4]:.3f}, {ts_mean_levels[5]:.3f}")
    print(f"[METRIC] MAE_mm_mean@>=0.01/...: {mae_mm_mean_levels[0]:.3f}, {mae_mm_mean_levels[1]:.3f}, {mae_mm_mean_levels[2]:.3f}, {mae_mm_mean_levels[3]:.3f}, {mae_mm_mean_levels[4]:.3f}, {mae_mm_mean_levels[5]:.3f}")
    print(f"[METRIC] Corr_mean: {corr_mean:.3f}")
    print(f"[METRIC] Final_Weighted_Score: {final_score:.4f}")
    print(f"[METRIC] Score_per_t: {', '.join([f'{s:.3f}' for s in score_k_arr])}")
    print("-" * 80)
    
    return {
        "final_score": final_score,
        "score_per_frame": score_k_arr,
        "pred_clean": pred_clean # 返回处理过噪音的预测值供绘图使用
    }

# ==========================================
# Part 3: 绘图功能函数 (Visualization)
# ==========================================
def plot_seq_visualization(obs_seq, true_seq, pred_seq, scores, out_path, vmax=1.0):
    """
    绘制 Obs, GT, Pred, Diff 对比图
    ... (逻辑保持不变)
    """
    T = true_seq.shape[0]
    rows = 4
    cols = T
    
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
        
        # 标注分数
        if scores is not None:
            axes[3, t].text(0.5, -0.15, f"S:{scores[t]:.2f}", 
                            transform=axes[3, t].transAxes, ha='center', fontsize=7, color='black')

    print(f'[INFO] Saving Plot to {out_path}')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

# ==========================================
# Part 4: 主入口函数 (Wrapper)
# ==========================================
def render(obs_seq, true_seq, pred_seq, out_path: str, vmax: float = 1.0):
    """
    整合计算与绘图的主函数
    ... (逻辑保持不变)
    """
    
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
    scores_arr = metrics_res['score_per_frame']
    pred_clean = metrics_res['pred_clean'] # 获取去噪后的预测值用于画图
    
    # 3. 调用绘图模块
    plot_seq_visualization(obs, tru, pred_clean, scores_arr, out_path, vmax=vmax)
    
    return final_score


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Test SCWDS SimVP Model')
    
    # 基础参数
    parser.add_argument('--data_path', type=str, default='data/samples.jsonl', help='Path to test data')
    parser.add_argument('--in_shape', type=int, nargs=4, default=[20, 28, 128, 128], help='Input shape: T C H W')
    parser.add_argument('--save_dir', type=str, default='./output/simvp', help='Directory where model checkpoints are saved')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--accelerator', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Accelerator to run inference on (e.g., cuda, cpu, cuda:0)')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 构建配置（与训练保持一致）
    config_kwargs = {
        'data_path': args.data_path,
        'in_shape': tuple(args.in_shape),
        'save_dir': args.save_dir,
    }
    
    config = SimVPConfig(**config_kwargs)
    
    # 从配置中获取参数
    resize_shape = (config.in_shape[2], config.in_shape[3])
    device = torch.device(args.accelerator)
    
    print(f"[INFO] 测试配置:")
    # ... (省略日志打印)
    
    # 数据集
    data_module = ScwdsDataModule(
        data_path=config.data_path,
        resize_shape=resize_shape,
        batch_size=1, # 推理/测试阶段单批次通常设为 1
        num_workers=1
    )
    # 强制 setup test 阶段
    data_module.setup('test')
    test_loader = data_module.test_dataloader()
    assert test_loader is not None
    
    # 加载模型
    ckpt_path = find_best_ckpt(config.save_dir)
    print(f"[INFO] 加载检查点: {ckpt_path}")
    # 注意: SimVP.load_from_checkpoint 必须能找到 SimVPConfig 中的所有参数
    model: SimVP = SimVP.load_from_checkpoint(ckpt_path)
    model.eval().to(device)

    # 简单可视化若干batch（仅展示每个batch的第一个样本）
    out_dir = os.path.join(config.save_dir, 'vis')
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] 可视化结果将保存到: {out_dir}")

    # 收集所有样本的 Score_mean
    score_mean_list = []

    with torch.no_grad():
        # 修正: test_loader 迭代时接收 5 个元素
        for bidx, batch in enumerate(test_loader):
            # 解包 5 个元素: metadata, x, y, target_mask, input_mask
            metadata_batch, batch_x, batch_y, target_mask, input_mask = batch
            
            # 移到设备上
            batch_x = batch_x.to(device) 
            batch_y = batch_y.to(device) 
            target_mask = target_mask.to(device)
            input_mask = input_mask.to(device)
            
            # test_step 期望接收 5 元素 (metadata, x, y, target_mask, input_mask)
            test_batch_tuple = (metadata_batch, batch_x, batch_y, target_mask, input_mask)
            
            # 调用 test_step
            outputs = model.test_step(test_batch_tuple, bidx)
            
            # 从outputs中提取数据
            obs_seq = outputs['inputs'] 
            true_seq = outputs['trues'] 
            pred_seq = outputs['preds'] 
            
            print(f'[INFO] Batch {bidx}: obs_seq.shape = {obs_seq.shape}, true_seq.shape = {true_seq.shape}, pred_seq.shape = {pred_seq.shape}')
            
            # 如果数据是 (T, H, W) 形状（缺少通道维度），添加通道维度 (适配 render 函数)
            if len(obs_seq.shape) == 3: obs_seq = obs_seq[:, np.newaxis, :, :]
            if len(true_seq.shape) == 3: true_seq = true_seq[:, np.newaxis, :, :]
            if len(pred_seq.shape) == 3: pred_seq = pred_seq[:, np.newaxis, :, :]

            out_path = os.path.join(out_dir, f'sample_{bidx:04d}.png')
            score_mean = render(obs_seq, true_seq, pred_seq, out_path, vmax=1.0)
            score_mean_list.append(score_mean)
            
            # 仅示例输出指定数量的样本
            if bidx >= args.num_samples - 1:
                break
    
    # 计算所有样本的平均 Score
    if len(score_mean_list) > 0:
        overall_score = np.mean(score_mean_list)
        print(f"\n[FINAL] 所有样本的平均 Score: {overall_score:.6f}")
        print(f"[FINAL] 共处理 {len(score_mean_list)} 个样本")
    
    print(f"[INFO] 测试完成！共生成 {len(score_mean_list)} 个可视化样本")


if __name__ == '__main__':
    main()