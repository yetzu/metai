# run/gan_infer_scwds_simvp.py
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

# 设置 matplotlib 后端
matplotlib.use('Agg')

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入依赖
from metai.utils import MetLabel, MLOGI, MLOGE
from metai.dataset.met_dataloader_scwds import ScwdsDataModule
from metai.model.simvp import SimVP_GAN
from metai.utils.met_config import get_config

# 竞赛常量
USER_ID = "CP2025000081" 
TRACK_ID = "GAN"
TIME_STEP_MINUTES = 6 

def find_best_gan_ckpt(save_dir: str) -> str:
    """查找 GAN 训练的最佳 checkpoint"""
    # 优先找 best.ckpt
    # 注意：GAN 训练脚本中 ModelCheckpoint 保存路径通常在 checkpoints 子目录
    ckpt_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(ckpt_dir):
        ckpt_dir = save_dir # 兼容旧结构
        
    # 优先找 last.ckpt (GAN 训练波动大，last 往往具有最新状态)
    last = os.path.join(ckpt_dir, 'last.ckpt')
    if os.path.exists(last): return last
    
    # 否则按时间排序找最新的
    cpts = sorted(glob.glob(os.path.join(ckpt_dir, '*.ckpt')), key=os.path.getmtime)
    if len(cpts) == 0:
        raise FileNotFoundError(f'No checkpoint found in {save_dir}')
    return cpts[-1]

def plot_inference(obs_seq, pred_seq, save_path):
    """
    绘制推理结果对比图 (Obs vs Pred)
    """
    T_in = obs_seq.shape[0]
    T_out = pred_seq.shape[0]
    
    cols = max(T_in, T_out)
    fig, axes = plt.subplots(2, cols, figsize=(cols * 1.5, 3.5), constrained_layout=True)
    
    vmax = 1.0 
    
    # 1. Plot Input (Obs)
    for t in range(cols):
        ax = axes[0, t]
        if t < T_in:
            ax.imshow(obs_seq[t], cmap='turbo', vmin=0.0, vmax=vmax)
            if t == 0: ax.set_title('Input (Past)', fontsize=10)
        else:
            ax.axis('off')
        ax.axis('off')

    # 2. Plot Prediction
    for t in range(cols):
        ax = axes[1, t]
        if t < T_out:
            ax.imshow(pred_seq[t], cmap='turbo', vmin=0.0, vmax=vmax)
            if t == 0: ax.set_title('Pred (GAN)', fontsize=10)
        else:
            ax.axis('off')
        ax.axis('off')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(description='Infer SCWDS SimVP-GAN Model')
    parser.add_argument('--data_path', type=str, default='data/samples.testset.jsonl') # 默认改为测试集
    # 注意：GAN 模型的输入尺寸由 Backbone 决定，这里只是为了兼容接口，实际会读取模型配置
    parser.add_argument('--in_shape', type=int, nargs=4, default=[20, 28, 256, 256]) 
    parser.add_argument('--save_dir', type=str, default='./output/simvp_gan', help='GAN output dir')
    parser.add_argument('--accelerator', type=str, default='cuda')
    parser.add_argument('--vis', action='store_true', help='Enable visualization')
    parser.add_argument('--vis_output', type=str, default='./output/simvp_gan/vis_infer')
    return parser.parse_args()

def main():
    args = parse_args()
    met_config = get_config() 
    FMT = met_config.file_date_format
    RA_MAX = MetLabel.RA.max 
    
    device = torch.device(args.accelerator if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print(f"[INFO] GAN Inference Config:")
    print(f"  Model Dir: {args.save_dir}")
    print(f"  Device:    {device}")
    print("=" * 60)

    # 1. 加载模型 (SimVP_GAN)
    try:
        ckpt_path = find_best_gan_ckpt(args.save_dir)
        MLOGI(f"加载 GAN 检查点: {ckpt_path}")
        
        model = SimVP_GAN.load_from_checkpoint(ckpt_path, map_location=device)
        model.eval().to(device)
        
        # 从模型中获取正确的 resize_shape (256, 256)
        resize_shape = model.backbone.resize_shape
        MLOGI(f"模型输入尺寸: {resize_shape}")
        
    except Exception as e:
        MLOGE(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. 数据模块
    data_module = ScwdsDataModule(
        data_path=args.data_path,
        resize_shape=resize_shape, # 必须与模型一致
        batch_size=1,
        num_workers=1
    )
    # 使用 'infer' stage (或者 test，取决于 dataset 实现)
    # 这里假设 infer_dataloader 可用
    data_module.setup('infer') 
    infer_loader = data_module.infer_dataloader()
    
    # 3. 推理循环
    with torch.no_grad():
        for bidx, batch in enumerate(infer_loader):
            try:
                metadata_list, batch_x, input_mask = batch
                
                # 数据预处理 (与 GAN training_step 保持一致)
                batch_x = batch_x.to(device)
                # 显式调用 Backbone 的插值函数，确保分辨率对齐
                x = model.backbone._interpolate_batch_gpu(batch_x, mode='max_pool')
                
                # Inference
                # GAN 的 forward 已经包含了 backbone + refiner + clamp
                # Output: [1, T, C, H, W] (normalized 0-1)
                batch_y = model(x)
                batch_y = batch_y.squeeze() # [20, H, W]
                
                metadata = metadata_list[0]
                sample_id = metadata['sample_id']
                
                # 解析元数据
                sample_id_parts = sample_id.split('_')
                # 兼容不同的 ID 格式
                if len(sample_id_parts) >= 4:
                    task_id = metadata.get('task_id') or sample_id_parts[0]
                    region_id = metadata.get('region_id') or sample_id_parts[1]
                    time_id = sample_id_parts[2] 
                    station_id = metadata.get('station_id') or sample_id_parts[3]
                    case_id = metadata.get('case_id') or '_'.join(sample_id_parts[:4])
                else:
                    # Fallback for simple IDs
                    task_id, region_id, time_id, station_id, case_id = "T0", "R0", "Time", "St", sample_id

                timestamps = metadata.get('timestamps')
                if not timestamps: 
                    MLOGE(f"Skipping {sample_id}: No timestamps found")
                    continue
                    
                last_obs_time_str = timestamps[-1]
                last_obs_dt = datetime.strptime(last_obs_time_str, FMT)
                
                # 统计变量
                seq_max_val = 0.0
                seq_mean_val = 0.0
                pred_frames_vis = []
                
                # 后处理与保存
                for idx, y in enumerate(batch_y):
                    # 1. Upsample to target resolution (301x301)
                    # 这里的 y 是 256x256
                    y_interp = F.interpolate(
                        y.unsqueeze(0).unsqueeze(0), 
                        size=(301, 301),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze() 

                    # 2. 反归一化 -> 物理数值 (mm)
                    PHYSICAL_MAX = 30.0 
                    y_phys = y_interp * PHYSICAL_MAX
                    
                    # 3. 物理阈值去噪
                    THRESHOLD_NOISE = 0.05 
                    y_phys[y_phys < THRESHOLD_NOISE] = 0.0
                    
                    # 4. 转换为存储格式 (放大 10 倍)
                    y_stored = y_phys * 10.0
                    y_final_np = y_stored.cpu().numpy().astype(np.float32)

                    # 5. 构造文件名并保存
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
                    
                    if args.vis:
                        pred_frames_vis.append(y_final_np / RA_MAX) # 归一化用于绘图
                
                seq_mean_val /= len(batch_y)
                MLOGI(f"No.{bidx} {sample_id} | Max: {seq_max_val:.2f}mm | Mean: {seq_mean_val:.4f}mm")
                
                # 6. 可视化
                if args.vis:
                    # obs shape: [1, T, C, H, W] -> [T, H, W]
                    obs_frames = batch_x[0, :, 0, :, :].cpu().numpy() 
                    pred_frames = np.array(pred_frames_vis)
                    
                    vis_path = os.path.join(args.vis_output, f"{sample_id}.png")
                    plot_inference(obs_frames, pred_frames, vis_path)

            except Exception as e:
                MLOGE(f"样本 {bidx} 推理失败: {e}")
                import traceback
                traceback.print_exc()
                continue
            
    MLOGI("✅ GAN 推理完成！")

if __name__ == '__main__':
    main()