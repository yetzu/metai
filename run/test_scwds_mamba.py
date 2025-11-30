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

# [修改] 使用更简洁的包级导入
from metai.dataset import ScwdsDataModule
from metai.model.met_mamba.trainer import MeteoMambaModule
# 引入标准评价指标模块
from metai.model.met_mamba.metrices import MetScore

# ==========================================
# Part 0: 辅助工具函数 (Helper Functions)
# ==========================================

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
    公式: mm = exp(norm * log(31)) - 1
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
    注意：输入数据应为物理值 (mm)，vmax 默认为 30.0 mm
    """
    T = true_seq.shape[0]
    rows, cols = 4, T
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5), constrained_layout=True)
    if T == 1: axes = axes[:, np.newaxis]
    
    # 统一颜色映射单位 (mm)
    cmap_rain = 'turbo' 
    
    for t in range(T):
        # 第0行: 观测 (Observation / Context)
        if t < obs_seq.shape[0]:
            im = axes[0, t].imshow(obs_seq[t], cmap=cmap_rain, vmin=0.0, vmax=vmax)
            axes[0, t].set_title(f'In-{t}', fontsize=6)
        else:
            axes[0, t].imshow(np.zeros_like(true_seq[0]), cmap='gray', vmin=0, vmax=1)
        axes[0, t].axis('off')
        if t == 0: axes[0, t].set_ylabel('Obs (mm)', fontsize=8)
        
        # 第1行: 真值 (Ground Truth)
        axes[1, t].imshow(true_seq[t], cmap=cmap_rain, vmin=0.0, vmax=vmax)
        axes[1, t].axis('off')
        if t == 0: axes[1, t].set_ylabel('GT (mm)', fontsize=8)
        axes[1, t].set_title(f'T+{t+1}', fontsize=6)
        
        # 第2行: 预测 (Prediction)
        axes[2, t].imshow(pred_seq[t], cmap=cmap_rain, vmin=0.0, vmax=vmax)
        axes[2, t].axis('off')
        if t == 0: axes[2, t].set_ylabel('Pred (mm)', fontsize=8)
        
        # 第3行: 差异 (Difference) (范围: -15mm 到 +15mm)
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
def process_batch(scorer, obs_tensor, true_tensor, pred_tensor, out_path):
    """
    计算评价指标并绘制图像 (使用物理降水值)
    """
    
    # 1. 计算指标 (Score)
    # MetScore 内部会自动处理 Log 反归一化，所以传入 normalized tensor 即可
    with torch.no_grad():
        score_dict = scorer(pred_tensor, true_tensor)
        
    final_score = score_dict['total_score'].item()
    
    # 打印详细评价指标
    r_time = score_dict['r_time'].cpu().numpy()
    score_time = score_dict['score_time'].cpu().numpy()
    
    print("-" * 60)
    print(f"{'T':<3} | {'Corr(R)':<10} | {'Score_k':<10}")
    print("-" * 60)
    for t in range(len(r_time)):
        print(f"{t:<3} | {r_time[t]:<10.4f} | {score_time[t]:<10.4f}")
    print("-" * 60)
    print(f"[METRIC] Final Weighted Score: {final_score:.6f}")
    print("-" * 60)

    # 2. 准备可视化数据 (转换为物理值 mm)
    def to_numpy_ch_mm(x, ch=0):
        # 取出数据并转换为 Numpy
        if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
        if x.ndim == 5: x = x[0] # (1, T, C, H, W) -> (T, C, H, W)
        if x.ndim == 4: x = x[:, ch] # (T, C, H, W) -> (T, H, W)
        
        # [关键] 反归一化：从对数归一化转换回物理 mm
        return denormalize(x)

    # 提取第0通道 (RA)，并还原为 mm
    obs_mm = to_numpy_ch_mm(obs_tensor) 
    true_mm = to_numpy_ch_mm(true_tensor)
    pred_mm = to_numpy_ch_mm(pred_tensor)
    
    print(f"Plotting (Physical mm): {os.path.basename(out_path)}")
    print(f"  Max Rain: GT={true_mm.max():.2f}mm, Pred={pred_mm.max():.2f}mm")
    
    # 绘图时 vmax 设为 30.0 mm (或根据数据动态调整)
    plot_seq_visualization(obs_mm, true_mm, pred_mm, out_path, vmax=30.0)
    
    return final_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='output/meteo_mamba')
    parser.add_argument('--data_path', type=str, default='data/samples.jsonl')
    
    # in_shape 改为 (C, H, W) 三个整数，与 config.py 保持一致
    parser.add_argument('--in_shape', type=int, nargs=3, default=[31, 256, 256], 
                        help='Input shape (C, H, W) without batch or time dim')
    
    # 显式添加序列长度参数
    parser.add_argument('--obs_seq_len', type=int, default=10, help='Observation sequence length')
    parser.add_argument('--pred_seq_len', type=int, default=20, help='Prediction sequence length')
    
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
    
    # [修改] 自动推导输出目录：使其位于 version_XX 文件夹下
    # 逻辑：获取 ckpt 的绝对路径，如果是 .../version_X/checkpoints/file.ckpt，则输出到 .../version_X/vis_00
    ckpt_dir_abs = os.path.dirname(os.path.abspath(args.ckpt_path))
    if os.path.basename(ckpt_dir_abs) == 'checkpoints':
        version_dir = os.path.dirname(ckpt_dir_abs)
    else:
        version_dir = ckpt_dir_abs
    
    out_dir = os.path.join(version_dir, f'vis_{epoch:02d}')
    
    os.makedirs(out_dir, exist_ok=True)
    set_logger(os.path.join(out_dir, 'log.txt'))
    
    print(f"[INFO] Starting Test on {device}")
    print_checkpoint_info(ckpt_info)
    print(f"[INFO] Input Shape (C, H, W): {args.in_shape}")
    print(f"[INFO] Seq Lengths: Obs={args.obs_seq_len}, Pred={args.pred_seq_len}")
    print(f"[INFO] Visualization results will be saved to: {out_dir}")
    
    # 1. 加载模型
    try:
        module = MeteoMambaModule.load_from_checkpoint(args.ckpt_path)
        module.eval()
        module.to(device)
        model = module.model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # 2. 初始化评分器 (use_log_norm=True)
    scorer = MetScore(use_log_norm=True).to(device)
    print(f"[INFO] MetScore initialized (use_log_norm=True)")

    # 3. 加载数据
    # resize_shape 取 in_shape 的后两维 (H, W)，与 trainer.py 逻辑一致
    resize_shape = (args.in_shape[1], args.in_shape[2])
    
    dm = ScwdsDataModule(
        data_path=args.data_path, 
        resize_shape=resize_shape, 
        batch_size=1, 
        num_workers=4,
        aft_seq_length=args.pred_seq_len  # 确保 DataModule 知道预测长度
    )
    dm.setup('test')
    
    scores = []
    with torch.no_grad():
        for bidx, batch in enumerate(dm.test_dataloader()):
            if bidx >= args.num_samples:
                print(f"[INFO] Reached limit ({args.num_samples}). Stopping.")
                break

            batch_device = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            _, x, y, _, t_mask = batch_device
            
            pred_raw = model(x)
            pred = torch.clamp(pred_raw, 0.0, 1.0)
            
            save_path = os.path.join(out_dir, f'sample_{bidx:03d}.png')
            s = process_batch(scorer, x, y, pred, save_path)
            scores.append(s)
            
    if len(scores) > 0:
        print(f"\n[FINAL] Average Score ({len(scores)} samples): {np.mean(scores):.6f}")
    restore_stdout()

if __name__ == '__main__':
    try:
        main()
    finally:
        restore_stdout()