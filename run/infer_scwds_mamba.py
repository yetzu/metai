# run/infer_scwds_mamba.py
import sys
import os
import glob
from typing import Any
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 设置 matplotlib 后端
matplotlib.use('Agg')

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.utils import MetLabel, MLOGI, MLOGE
from metai.dataset.met_dataset_scwds import ScwdsDataModule
from metai.model.met_mamba.trainer import MeteoMambaModule
from metai.utils.met_config import get_config

# ==========================================
# 常量配置
# ==========================================
USER_ID = "CP2025000081"
TRACK_ID = "track1"
TIME_STEP_MINUTES = 6

# 物理空间最小有效降水 (mm)，小于此值视为噪声置零
MIN_VALID_RAIN_MM = 0.1
# 物理最大值 (mm)，对应训练时归一化的分母
PHYSICAL_MAX = 30.0

def find_latest_ckpt(save_dir: str) -> str:
    """
    递归查找指定目录下最新的检查点文件 (.ckpt)。
    优先查找 best.ckpt 和 last.ckpt，否则按文件名排序取最后一个。
    """
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {save_dir}")

    best = os.path.join(save_dir, 'best.ckpt')
    if os.path.exists(best): return best

    last = os.path.join(save_dir, 'last.ckpt')
    if os.path.exists(last): return last

    cpts = sorted(glob.glob(os.path.join(save_dir, '**/*.ckpt'), recursive=True))
    if len(cpts) == 0:
        raise FileNotFoundError(f'No checkpoint found in {save_dir}')

    return cpts[-1]

def denormalize_to_mm(data_norm):
    """
    反归一化：将对数空间 [0, 1] 还原为物理空间 [0, 30] mm。
    公式: x_mm = expm1(x_norm * log(30 + 1))
    """
    log_factor = np.log(PHYSICAL_MAX + 1)
    if isinstance(data_norm, torch.Tensor):
        return torch.expm1(data_norm * log_factor)
    return np.expm1(data_norm * log_factor)

def calc_stats(data_tensor):
    """
    计算统计指标：最小值、最大值、均值、非零占比（降水面积比）。
    """
    if isinstance(data_tensor, np.ndarray):
        data_tensor = torch.from_numpy(data_tensor)

    _min = data_tensor.min().item()
    _max = data_tensor.max().item()
    _mean = data_tensor.mean().item()
    _nz = (data_tensor > MIN_VALID_RAIN_MM).float().mean().item()

    return f"Min:{_min:.2f} | Max:{_max:.2f} | Mean:{_mean:.4f} | NZ:{_nz:.2%}"

def plot_inference(obs_seq, pred_seq, save_path):
    """
    绘制 Obs (历史观测) 与 Pred (未来预测) 的对比图并保存。
    obs_seq, pred_seq 均应为线性归一化到 [0, 1] 的数据。
    """
    T_in = obs_seq.shape[0]
    T_out = pred_seq.shape[0]
    cols = max(T_in, T_out)

    fig, axes = plt.subplots(2, cols, figsize=(cols * 1.5, 4.0), constrained_layout=True)
    vmax = 1.0

    for t in range(cols):
        # 第一行: 输入观测 (Obs)
        ax = axes[0, t]
        if t < T_in:
            ax.imshow(obs_seq[t], cmap='turbo', vmin=0, vmax=vmax)
            if t == 0: ax.set_title('Obs', fontsize=10)
        else:
            ax.set_visible(False)
        ax.axis('off')

        # 第二行: 未来预测 (Pred)
        ax = axes[1, t]
        if t < T_out:
            ax.imshow(pred_seq[t], cmap='turbo', vmin=0, vmax=vmax)
            if t == 0: ax.set_title('Pred', fontsize=10)
        else:
            ax.set_visible(False)
        ax.axis('off')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(description='Infer SCWDS Mamba Model')
    parser.add_argument('--data_path', type=str, default='data/samples.testset.jsonl', help='Path to testset jsonl')
    parser.add_argument('--resize_shape', type=int, nargs=2, default=[256, 256], help='Model input resolution')
    parser.add_argument('--ckpt_dir', type=str, default='./output/meteo_mamba', help='Directory containing checkpoints')
    parser.add_argument('--save_dir', type=str, default='./submit/output', help='Directory to save submission files')
    parser.add_argument('--accelerator', type=str, default='cuda', help='Device accelerator')
    parser.add_argument('--vis', action='store_true', help='Enable visualization')
    parser.add_argument('--vis_output', type=str, default='./submit/vis_infer', help='Visualization output directory')
    return parser.parse_args()

def main():
    args = parse_args()
    met_config = get_config()
    FMT = met_config.file_date_format
    device = torch.device(args.accelerator if torch.cuda.is_available() else 'cpu')

    MLOGI(f"[INFO] Start Inference on {device}")
    MLOGI(f"[INFO] Checkpoint Dir: {args.ckpt_dir}")
    MLOGI(f"[INFO] Output Dir: {args.save_dir}")

    # 1. 加载数据模块
    # 使用 "TestSet" 名称以匹配 TestSet.lmdb 文件
    data_module = ScwdsDataModule(
        data_path=args.data_path,
        resize_shape=tuple(args.resize_shape),
        batch_size=1,
        num_workers=2,
        testset_name="TestSet"
    )
    data_module.setup('infer')
    infer_loader = data_module.infer_dataloader()

    # 2. 加载模型
    try:
        ckpt_path = find_latest_ckpt(args.ckpt_dir)
        MLOGI(f"Loading checkpoint: {ckpt_path}")
        model = MeteoMambaModule.load_from_checkpoint(ckpt_path, map_location=device)
        model.eval().to(device)
    except Exception as e:
        MLOGE(f"Model load failed: {e}")
        return

    MLOGI(f"Starting inference loop (Total: {len(infer_loader)} samples)...")

    # 3. 推理循环
    with torch.no_grad():
        for bidx, (metadata_list, batch_x, input_mask) in enumerate(infer_loader):
            try:
                batch_x = batch_x.to(device)

                # 模型推理
                # 输入: [B, T_in, C, H, W], 输出: [B, T_out, C, H, W]
                # batch_y_raw = model(batch_x)
                batch_y_raw = model(batch_x, current_epoch=100)
                
                # 范围截断 [0, 1]
                batch_y = torch.clamp(batch_y_raw, 0.0, 1.0)
                
                # 可选：在 Log 空间预先去除极小的浮点噪声 (< 1e-4) 以减少插值时的晕染
                batch_y[batch_y < 1e-4] = 0.0

                # 解析单样本 (Batch=1)
                x_sample = batch_x[0]
                y_sample = batch_y[0, :, 0, :, :] # 去除 Channel 维 -> [T, H, W]

                metadata = metadata_list[0]
                sample_id = metadata['sample_id']
                timestamps = metadata.get('timestamps', [])

                if not timestamps:
                    MLOGE(f"Skipping {sample_id}: No timestamps found.")
                    continue

                # 解析元数据 ID: Task_Region_Time_Station_Radar_Batch
                parts = sample_id.split('_')
                task_id = metadata.get('task_id', parts[0])
                region_id = metadata.get('region_id', parts[1])
                time_id = parts[2]
                station_id = metadata.get('station_id', parts[3])
                case_id = metadata.get('case_id', '_'.join(parts[:4]))

                # 计算起始时间
                last_obs_idx = batch_x.shape[1] - 1
                if last_obs_idx >= len(timestamps):
                    last_obs_idx = -1
                last_obs_dt = datetime.strptime(timestamps[last_obs_idx], FMT)

                # 后处理与保存
                pred_vis_list = []
                final_frames_mm = []

                for idx, y_frame in enumerate(y_sample):
                    # 1. 插值: 256x256 -> 301x301 (竞赛要求)
                    y_interp = F.interpolate(
                        y_frame.view(1, 1, *y_frame.shape),
                        size=(301, 301),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()

                    # 2. 反归一化: Log-Space -> Physical Space (mm)
                    y_mm = denormalize_to_mm(y_interp)

                    # 3. 物理空间清洗: 过滤小于 0.1mm 的微量降水
                    y_mm[y_mm < MIN_VALID_RAIN_MM] = 0.0
                    
                    # 收集用于统计
                    final_frames_mm.append(y_mm)

                    # 4. 单位转换: mm -> 0.1mm (INT16)
                    # 例如 1.5mm -> 15
                    y_save_uint16 = (y_mm * 10.0).cpu().numpy().astype(np.uint16)

                    # 5. 构造文件名并保存
                    forecast_dt = last_obs_dt + timedelta(minutes=TIME_STEP_MINUTES * (idx + 1))
                    fname = f"{task_id}_{region_id}_{time_id}_{station_id}_Forcast_{forecast_dt.strftime(FMT)}.npy"
                    
                    save_path = os.path.join(args.save_dir, USER_ID, TRACK_ID, case_id)
                    os.makedirs(save_path, exist_ok=True)
                    np.save(os.path.join(save_path, fname), y_save_uint16)

                    # 收集用于可视化 (线性归一化到 0-1)
                    if args.vis:
                        pred_vis_list.append(y_mm.cpu().numpy() / PHYSICAL_MAX)

                # 打印统计信息
                # Obs: 假设第 0 通道为降水/雷达，先反归一化
                obs_mm = denormalize_to_mm(x_sample[:, 0, :, :])
                obs_stats = calc_stats(obs_mm)
                
                # Pred: 使用最终处理过的物理值
                pred_mm_stack = torch.stack(final_frames_mm)
                pred_stats = calc_stats(pred_mm_stack)

                MLOGI(f"[{bidx}] {sample_id}")
                MLOGI(f"   Obs Stats : {obs_stats}")
                MLOGI(f"   Pred Stats: {pred_stats}")

                # 6. 可视化
                if args.vis:
                    obs_vis = (obs_mm.cpu().numpy() / PHYSICAL_MAX)
                    pred_vis = np.array(pred_vis_list)
                    vis_path = os.path.join(args.vis_output, f"{sample_id}.png")
                    plot_inference(obs_vis, pred_vis, vis_path)

            except Exception as e:
                MLOGE(f"Inference failed for batch {bidx}: {e}")
                import traceback
                traceback.print_exc()
                continue

    MLOGI(f"Completed. Output saved to: {args.save_dir}")

if __name__ == '__main__':
    main()