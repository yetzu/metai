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

class TeeLogger:
    """同时输出到控制台和文件的日志类"""
    def __init__(self, log_file_path):
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.console = sys.stdout
        
    def write(self, message):
        """写入消息到控制台和文件"""
        self.console.write(message)
        self.log_file.write(message)
        self.log_file.flush()
        
    def flush(self):
        """刷新缓冲区"""
        self.console.flush()
        self.log_file.flush()
        
    def close(self):
        """关闭文件"""
        if self.log_file:
            self.log_file.close()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# 全局日志对象，初始为 None
_logger = None

def set_logger(log_file_path):
    """设置全局日志对象"""
    global _logger
    _logger = TeeLogger(log_file_path)
    sys.stdout = _logger
    
def restore_stdout():
    """恢复标准输出"""
    global _logger
    if _logger:
        sys.stdout = _logger.console
        _logger.close()
        _logger = None

def find_best_ckpt(save_dir: str) -> str:
    # 优先查找 best.ckpt
    best = os.path.join(save_dir, 'best.ckpt')
    if os.path.exists(best): return best
    
    # 其次查找 last.ckpt
    last = os.path.join(save_dir, 'last.ckpt')
    if os.path.exists(last): return last
    
    # 最后查找所有 checkpoint 文件，返回最新的
    # 排除 'last.ckpt' 以免干扰排序 (如果 glob 包含它)
    cpts = glob.glob(os.path.join(save_dir, '*.ckpt'))
    cpts = [c for c in cpts if 'last.ckpt' not in c and 'best.ckpt' not in c]
    
    if len(cpts) > 0:
        # 按修改时间或文件名排序，这里按文件名排序通常能取到 epoch 最大的
        cpts = sorted(cpts)
        return cpts[-1]
        
    # 如果只有 best/last 但前面没捕获到（理论上不可能），再兜底
    all_cpts = sorted(glob.glob(os.path.join(save_dir, '*.ckpt')))
    if len(all_cpts) == 0:
        raise FileNotFoundError(f'No checkpoint found in {save_dir}')
    return all_cpts[-1]

def get_checkpoint_info(ckpt_path: str):
    """从 checkpoint 文件中提取训练关键信息（不打印）"""
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        epoch = ckpt.get('epoch', None)
        global_step = ckpt.get('global_step', None)
        hparams = ckpt.get('hyper_parameters', {})
        return {
            'epoch': epoch,
            'global_step': global_step,
            'hparams': hparams,
            'ckpt_name': os.path.basename(ckpt_path)
        }
    except Exception as e:
        return {'error': str(e)}

def print_checkpoint_info(ckpt_info: dict):
    """打印 checkpoint 信息"""
    if 'error' in ckpt_info:
        print(f"[WARNING] 无法读取 checkpoint 信息: {ckpt_info['error']}")
        return
    
    print("=" * 80)
    print(f"[INFO] Loaded Checkpoint: {ckpt_info['ckpt_name']}")
    print(f"  Epoch: {ckpt_info.get('epoch', 'N/A')}")
    print(f"  Global Step: {ckpt_info.get('global_step', 'N/A')}")
    
    hparams = ckpt_info.get('hparams', {})
    if hparams:
        print(f"  Model Type: {hparams.get('model_type', 'N/A')}")
        print(f"  In Shape: {hparams.get('in_shape', 'N/A')}")
        print(f"  Out Seq Length: {hparams.get('aft_seq_length', 'N/A')}")
    print("=" * 80)

# ==========================================
# Part 1: 全局评分配置 (Metric Configuration)
# ==========================================
class MetricConfig:
    # [关键确认] MM_MAX = 30.0
    # 物理含义：数据中的最大整数值 300 对应 30.0 mm 降水
    # 模型输出 [0, 1] -> 映射到 [0, 30.0] mm
    MM_MAX = 30.0
    
    # 噪音阈值 (mm)
    THRESHOLD_NOISE = 0.05 
    
    # 阈值边缘 (mm)
    LEVEL_EDGES = np.array([0.0, 0.1, 1.0, 2.0, 5.0, 8.0, np.inf], dtype=np.float32)
    
    # 降水等级权重
    _raw_level_weights = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.3], dtype=np.float32)
    LEVEL_WEIGHTS = _raw_level_weights / _raw_level_weights.sum()

    # 时效权重 (0-19 帧，对应 2 小时)
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
    计算序列的降水评分指标
    true_seq: (T, H, W) 归一化值 [0, 1]
    pred_seq: (T, H, W) 归一化值 [0, 1]
    """
    T, H, W = true_seq.shape
    
    # 1. 预处理：噪音过滤
    # 转换阈值到归一化空间： 0.05 / 30.0
    pred_clean = pred_seq.copy()
    pred_clean[pred_clean < (MetricConfig.THRESHOLD_NOISE / MetricConfig.MM_MAX)] = 0.0
    
    time_weights = MetricConfig.get_time_weights(T)
    score_k_list = []
    
    # 反归一化到 mm 并截断负值
    tru_mm_seq = np.clip(true_seq, 0.0, None) * MetricConfig.MM_MAX
    prd_mm_seq = np.clip(pred_clean, 0.0, None) * MetricConfig.MM_MAX
    
    ts_mean_levels = np.zeros(len(MetricConfig.LEVEL_WEIGHTS))
    mae_mm_mean_levels = np.zeros(len(MetricConfig.LEVEL_WEIGHTS))
    corr_sum = 0.0

    if verbose:
        print(f"True Stats (mm): Max={np.max(tru_mm_seq):.2f}, Mean={np.mean(tru_mm_seq):.2f}")
        print(f"Pred Stats (mm): Max={np.max(prd_mm_seq):.2f}, Mean={np.mean(prd_mm_seq):.2f}")
        print("-" * 90)
        print(f"{'T':<3} | {'Corr(R)':<9} | {'TS_w_sum':<9} | {'Score_k':<9} | {'W_time':<8}")
        print("-" * 90)

    for t in range(T):
        tru_frame = tru_mm_seq[t].reshape(-1)
        prd_frame = prd_mm_seq[t].reshape(-1)
        abs_err = np.abs(prd_frame - tru_frame)

        # --- A. 计算相关系数 R_k ---
        mask_valid_corr = (tru_frame > 0) | (prd_frame > 0)
        if mask_valid_corr.sum() > 1:
            t_valid = tru_frame[mask_valid_corr]
            p_valid = prd_frame[mask_valid_corr]
            numerator = np.sum((t_valid - t_valid.mean()) * (p_valid - p_valid.mean()))
            denom = np.sqrt(np.sum((t_valid - t_valid.mean())**2) * np.sum((p_valid - p_valid.mean())**2))
            R_k = numerator / (denom + 1e-8)
        else:
            R_k = 0.0
        
        R_k = float(np.clip(R_k, -1.0, 1.0))
        corr_sum += R_k
        term_corr = np.sqrt(np.exp(R_k - 1.0))

        # --- B. 逐等级计算 TS 和 MAE ---
        weighted_sum_metrics = 0.0
        for i in range(len(MetricConfig.LEVEL_WEIGHTS)):
            low = MetricConfig.LEVEL_EDGES[i]
            high = MetricConfig.LEVEL_EDGES[i+1]
            w_i = MetricConfig.LEVEL_WEIGHTS[i]

            tru_bin = (tru_frame >= low) & (tru_frame < high)
            prd_bin = (prd_frame >= low) & (prd_frame < high)
            
            tp = np.logical_and(tru_bin, prd_bin).sum()
            fn = np.logical_and(tru_bin, ~prd_bin).sum()
            fp = np.logical_and(~tru_bin, prd_bin).sum()
            denom_ts = tp + fn + fp
            ts_val = (tp / denom_ts) if denom_ts > 0 else 1.0
            
            mask_eval = tru_bin | prd_bin
            mae_val = np.mean(abs_err[mask_eval]) if mask_eval.sum() > 0 else 0.0
            
            ts_mean_levels[i] += ts_val / T
            mae_mm_mean_levels[i] += mae_val / T
            
            term_mae = np.sqrt(np.exp(-mae_val / 100.0))
            weighted_sum_metrics += w_i * ts_val * term_mae

        # --- C. 计算 Score_k ---
        Score_k = term_corr * weighted_sum_metrics
        score_k_list.append(Score_k)

        if verbose:
            print(f"{t:<3} | {R_k:<9.4f} | {weighted_sum_metrics:<9.4f} | {Score_k:<9.4f} | {time_weights[t]:<8.4f}")

    score_k_arr = np.array(score_k_list)
    final_score = np.sum(score_k_arr * time_weights)
    
    print("-" * 90)
    ts_str = ", ".join([f"{v:.3f}" for v in ts_mean_levels])
    mae_str = ", ".join([f"{v:.3f}" for v in mae_mm_mean_levels])
    
    print(f"[METRIC] TS_mean  (Levels): {ts_str}")
    print(f"[METRIC] MAE_mean (Levels): {mae_str}")
    print(f"[METRIC] Corr_mean: {corr_sum / T:.4f}")
    print(f"[METRIC] Final_Weighted_Score: {final_score:.6f}")
    print(f"[METRIC] Score_per_t: {', '.join([f'{s:.3f}' for s in score_k_arr])}")
    print("-" * 90)
    
    return {
        "final_score": final_score,
        "score_per_frame": score_k_arr,
        "pred_clean": pred_clean
    }

# ==========================================
# Part 3: 绘图功能 (Visualization)
# ==========================================
def plot_seq_visualization(obs_seq, true_seq, pred_seq, scores, out_path, vmax=1.0):
    """
    绘制 Obs, GT, Pred, Diff 对比图
    obs_seq: (T_in, H, W)
    true_seq: (T_out, H, W)
    pred_seq: (T_out, H, W)
    """
    T = true_seq.shape[0] # T_out = 20
    rows, cols = 4, T
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5), constrained_layout=True)
    if T == 1: axes = axes[:, np.newaxis]

    for t in range(T):
        # 1. Obs (Input) - 注意处理 T_in < T_out 的情况
        if t < obs_seq.shape[0]:
            axes[0, t].imshow(obs_seq[t], cmap='turbo', vmin=0.0, vmax=vmax)
            axes[0, t].set_title(f'In-{t}', fontsize=6)
        else:
            # 超过输入长度的部分显示为空白或灰色
            axes[0, t].imshow(np.zeros_like(true_seq[0]), cmap='gray', vmin=0.0, vmax=1.0)
        axes[0, t].axis('off')
        if t == 0: axes[0, t].set_ylabel('Obs', fontsize=8)

        # 2. GT (Target)
        axes[1, t].imshow(true_seq[t], cmap='turbo', vmin=0.0, vmax=vmax)
        axes[1, t].axis('off')
        if t == 0: axes[1, t].set_ylabel('GT', fontsize=8)
        axes[1, t].set_title(f'T+{t+1}', fontsize=6)

        # 3. Pred
        axes[2, t].imshow(pred_seq[t], cmap='turbo', vmin=0.0, vmax=vmax)
        axes[2, t].axis('off')
        if t == 0: axes[2, t].set_ylabel('Pred', fontsize=8)
        
        # 4. Diff (GT - Pred)
        diff = true_seq[t] - pred_seq[t]
        axes[3, t].imshow(diff, cmap='bwr', vmin=-0.5, vmax=0.5)
        axes[3, t].axis('off')
        if t == 0: axes[3, t].set_ylabel('Diff', fontsize=8)

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
        # 输入维度 (T, C, H, W), 通道 0 是雷达/降水标签
        if x.ndim == 4: x = x[:, ch] 
        return x

    obs = to_numpy_ch(obs_seq) # 取 ch=0 (Radar/Label)
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
    
    # [关键修改] 更新默认输入形状: 10帧, 54通道 (30基础+24未来NWP)
    parser.add_argument('--in_shape', type=int, nargs=4, default=[10, 54, 256, 256],
                        help='Input shape (T, C, H, W). Default: [10, 54, 256, 256]')
    
    # [关键新增] 增加 aft_seq_length，默认20
    parser.add_argument('--aft_seq_length', type=int, default=20,
                        help='Output sequence length. Default: 20')
                        
    parser.add_argument('--save_dir', type=str, default='./output/simvp')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--accelerator', type=str, default='cuda')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.accelerator if torch.cuda.is_available() else 'cpu')
    
    # 1. Config & Model
    # 初始化 Config 对象，主要用于获取 resize_shape
    # 注意: aft_seq_length 会通过 hparams 从 checkpoint 加载，这里 Config 仅作辅助
    config = SimVPConfig(
        data_path=args.data_path,
        in_shape=tuple(args.in_shape),
        save_dir=args.save_dir,
        aft_seq_length=args.aft_seq_length
    )
    resize_shape = (config.in_shape[2], config.in_shape[3])
    
    # 查找并加载 checkpoint
    ckpt_path = find_best_ckpt(config.save_dir)
    ckpt_info = get_checkpoint_info(ckpt_path)
    epoch = ckpt_info.get('epoch', None)
    
    if epoch is None:
        epoch = 0
    
    # 创建输出目录
    out_dir = os.path.join(config.save_dir, f'vis_{epoch:02d}')
    os.makedirs(out_dir, exist_ok=True)
    
    # 设置日志
    log_file_path = os.path.join(out_dir, 'log.txt')
    set_logger(log_file_path)
    
    print(f"[INFO] Starting Test on {device}")
    print_checkpoint_info(ckpt_info)
    print(f"[INFO] Input Shape: {args.in_shape}")
    print(f"[INFO] Output Length: {args.aft_seq_length}")
    print(f"[INFO] Metric MM_MAX: {MetricConfig.MM_MAX}")
    print(f"[INFO] 可视化结果将保存到: {out_dir}")
    
    # 加载模型
    model = SimVP.load_from_checkpoint(ckpt_path)
    model.eval().to(device)
    
    # 2. Data
    # 使用 ScwdsDataModule，它会自动调用更新后的 MetSample
    data_module = ScwdsDataModule(
        data_path=config.data_path,
        resize_shape=resize_shape,
        batch_size=1,
        num_workers=4
    )
    # setup('test') 会准备 test_dataloader
    data_module.setup('test')
    test_loader = data_module.test_dataloader()
    
    scores = []
    
    with torch.no_grad():
        for bidx, batch in enumerate(test_loader):
            metadata_batch, batch_x, batch_y, target_mask, input_mask = batch
            
            # Inference
            # batch_x: [1, 10, 54, 256, 256]
            # batch_y: [1, 20, 1, 256, 256]
            outputs = model.test_step(
                (metadata_batch, batch_x.to(device), batch_y.to(device), target_mask.to(device), input_mask.to(device)), 
                bidx
            )
            
            # Render
            save_path = os.path.join(out_dir, f'sample_{bidx:03d}.png')
            
            # outputs['inputs']: [10, 54, H, W] -> render 取 ch=0 -> Past Radar
            # outputs['trues']: [20, 1, H, W]
            # outputs['preds']: [20, 1, H, W]
            s = render(outputs['inputs'], outputs['trues'], outputs['preds'], save_path)
            scores.append(s)
            
            if bidx >= args.num_samples - 1:
                break
    
    if len(scores) > 0:
        print(f"\n[FINAL] Average Score ({len(scores)} samples): {np.mean(scores):.6f}")
    
    restore_stdout()

if __name__ == '__main__':
    try:
        main()
    finally:
        restore_stdout()