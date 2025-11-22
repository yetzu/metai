# run/infer_scwds_simvp.py
import sys
import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 设置 matplotlib 后端，防止无 GUI 环境报错
matplotlib.use('Agg')

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入实际依赖
from metai.utils import MetLabel, MLOGI, MLOGE
from metai.dataset.met_dataloader_scwds import ScwdsDataModule
from metai.model.simvp import SimVPConfig, SimVP
from metai.utils.met_config import get_config

# 竞赛常量
USER_ID = "CP2025000081"  # 请确保这是正确的选手ID
TRACK_ID = "SimVP"
TIME_STEP_MINUTES = 6 

def find_latest_ckpt(save_dir: str) -> str:
    """优先找 best.ckpt (如果存在), 否则找 last.ckpt"""
    best = os.path.join(save_dir, 'best.ckpt')
    if os.path.exists(best): return best
    
    last = os.path.join(save_dir, 'last.ckpt')
    if os.path.exists(last): return last
    
    cpts = sorted(glob.glob(os.path.join(save_dir, '*.ckpt')))
    if len(cpts) == 0:
        raise FileNotFoundError(f'No checkpoint found in {save_dir}')
    return cpts[-1]

def plot_inference(obs_seq, pred_seq, save_path):
    """
    绘制推理结果对比图
    Args:
        obs_seq: 输入序列 (T, H, W) [0, 1]
        pred_seq: 预测序列 (T, H, W) [0, 1]
        save_path: 图片保存路径
    """
    T_in = obs_seq.shape[0]
    T_out = pred_seq.shape[0]
    
    # 设置绘图布局：2行 (Obs, Pred)，列数为 T
    cols = max(T_in, T_out)
    fig, axes = plt.subplots(2, cols, figsize=(cols * 1.5, 3.5), constrained_layout=True)
    
    # 统一色标范围 (0-1 代表 0-30mm)
    vmax = 1.0 
    
    # 1. Plot Input (Obs)
    for t in range(cols):
        ax = axes[0, t]
        if t < T_in:
            ax.imshow(obs_seq[t], cmap='turbo', vmin=0.0, vmax=vmax)
            if t == 0: ax.set_title('Input (Past 2h)', fontsize=10)
        else:
            ax.axis('off') # 如果输入比输出短
        ax.axis('off')

    # 2. Plot Prediction
    for t in range(cols):
        ax = axes[1, t]
        if t < T_out:
            ax.imshow(pred_seq[t], cmap='turbo', vmin=0.0, vmax=vmax)
            if t == 0: ax.set_title('Pred (Future 2h)', fontsize=10)
        else:
            ax.axis('off')
        ax.axis('off')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(description='Infer SCWDS SimVP Model')
    parser.add_argument('--data_path', type=str, default='data/samples.jsonl')
    parser.add_argument('--in_shape', type=int, nargs=4, default=[20, 28, 256, 256]) 
    parser.add_argument('--save_dir', type=str, default='./output/simvp')
    parser.add_argument('--accelerator', type=str, default='cuda')
    # 可视化参数
    parser.add_argument('--vis', action='store_true', help='Enable visualization plotting')
    parser.add_argument('--vis_output', type=str, default='./output/simvp/vis_infer', help='Directory to save visualization plots')
    return parser.parse_args()

def main():
    args = parse_args()
    met_config = get_config() 
    
    # 1. Config
    config = SimVPConfig(
        data_path=args.data_path,
        in_shape=tuple(args.in_shape),
        save_dir=args.save_dir,
    )
    resize_shape = (config.in_shape[2], config.in_shape[3])
    
    # 设备选择
    device = torch.device(args.accelerator if torch.cuda.is_available() else 'cpu')
    
    RA_MAX = MetLabel.RA.max 
    FMT = met_config.file_date_format
    
    print("=" * 60)
    print(f"[INFO] Inference Config:")
    print(f"  Model Dir: {args.save_dir}")
    print(f"  User ID:   {USER_ID}")
    print(f"  Device:    {device}")
    print("=" * 60)
    
    # 2. Data
    data_module = ScwdsDataModule(
        data_path=config.data_path,
        resize_shape=resize_shape,
        batch_size=1,
        num_workers=1
    )
    data_module.setup('infer')
    infer_loader = data_module.infer_dataloader()
    
    # 3. Model
    try:
        ckpt_path = find_latest_ckpt(config.save_dir)
        MLOGI(f"加载检查点: {ckpt_path}")
        model = SimVP.load_from_checkpoint(ckpt_path, map_location=device)
        model.eval().to(device)
    except Exception as e:
        MLOGE(f"模型加载失败: {e}")
        return

    # 4. Inference Loop
    with torch.no_grad():
        for bidx, batch in enumerate(infer_loader):
            try:
                metadata_list, batch_x, input_mask = batch
                batch_x = batch_x.to(device)
                
                # Inference
                # batch_y shape: [1, 20, 1, H, W] (normalized 0-1)
                batch_y = model.infer_step((metadata_list, batch_x, input_mask), batch_idx=bidx)
                batch_y = batch_y.squeeze() # [20, H, W]
                
                metadata = metadata_list[0]
                sample_id = metadata['sample_id']
                
                # Parse Metadata
                sample_id_parts = sample_id.split('_')
                task_id = metadata.get('task_id') or sample_id_parts[0]
                region_id = metadata.get('region_id') or sample_id_parts[1]
                time_id = sample_id_parts[2] 
                station_id = metadata.get('station_id') or sample_id_parts[3]
                case_id = metadata.get('case_id') or '_'.join(sample_id_parts[:4])
                timestamps = metadata.get('timestamps')
                
                if not timestamps: continue
                last_obs_time_str = timestamps[-1]
                last_obs_dt = datetime.strptime(last_obs_time_str, FMT)
                
                # 用于统计该样本的整体预测情况
                seq_max_val = 0.0
                seq_mean_val = 0.0
                seq_zero_count = 0  # 0值计数
                seq_total_count = 0 # 总像素计数

                # 收集数据用于绘图
                pred_frames_vis = []
                
                # Save Results
                for idx, y in enumerate(batch_y):
                    # 1. Upsample to target resolution (301x301)
                    y_interp = F.interpolate(
                        y.unsqueeze(0).unsqueeze(0), 
                        size=(301, 301),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze() 

                    # 2. 反归一化到【物理数值】(mm)
                    # 训练时归一化是除以300。300对应30.0mm。
                    PHYSICAL_MAX = 30.0 
                    y_phys = y_interp * PHYSICAL_MAX
                    
                    # 3. 物理阈值去噪 (关键提分点)
                    THRESHOLD_NOISE = 0.05 # 0.05mm
                    y_phys[y_phys < THRESHOLD_NOISE] = 0.0
                    
                    # 4. 转换为【存储格式】(放大 10 倍)
                    y_stored = y_phys * 10.0
                    
                    # 5. 保存为 Float32 (保留精度)
                    y_final_np = y_stored.cpu().numpy().astype(np.float32)

                    # 6. Save File
                    forecast_dt = last_obs_dt + timedelta(minutes=TIME_STEP_MINUTES * (idx + 1))
                    forecast_time_str = forecast_dt.strftime(FMT)
                    
                    npy_dir_final = os.path.join(
                        'submit', 'output', USER_ID, TRACK_ID, case_id
                    )
                    os.makedirs(npy_dir_final, exist_ok=True)
                    
                    npy_path = os.path.join(
                        npy_dir_final,
                        f"{task_id}_{region_id}_{time_id}_{station_id}_Forcast_{forecast_time_str}.npy"
                    )
                    
                    np.save(npy_path, y_final_np)

                    # 统计
                    seq_max_val = max(seq_max_val, float(y_final_np.max()) / 10.0)
                    seq_mean_val += float(y_final_np.mean()) / 10.0
                    seq_zero_count += np.sum(y_final_np == 0)
                    seq_total_count += y_final_np.size
                    
                    # 收集归一化预测值用于绘图 (恢复到 0-1)
                    if args.vis:
                        # y_final_np 是 0-300，RA_MAX 是 300
                        pred_frames_vis.append(y_final_np / RA_MAX)
                
                # 打印关键统计信息
                seq_mean_val /= len(batch_y)
                zero_ratio = seq_zero_count / seq_total_count if seq_total_count > 0 else 0.0
                MLOGI(f"No.{bidx} {sample_id} | Max: {seq_max_val:.2f}mm | Mean: {seq_mean_val:.4f}mm | Zero: {zero_ratio:.2%}")
                
                # 7. 执行可视化
                if args.vis:
                    # 获取输入序列 (Rain Channel is 0)
                    # batch_x shape: [1, T, C, H, W] -> [T, H, W]
                    obs_frames = batch_x[0, :, 0, :, :].cpu().numpy() 
                    pred_frames = np.array(pred_frames_vis)
                    print(f"obs_frames.shape = {obs_frames.shape}, pred_frames.shape = {pred_frames.shape}")
                    
                    vis_path = os.path.join(args.vis_output, f"{sample_id}.png")
                    plot_inference(obs_frames, pred_frames, vis_path)
                

            except Exception as e:
                MLOGE(f"样本 {bidx} 推理失败: {e}")
                continue
            
    MLOGI("✅ 推理完成！")

if __name__ == '__main__':
    main()