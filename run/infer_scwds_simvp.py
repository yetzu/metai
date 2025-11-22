# run/infer_scwds_simvp.py
import sys
import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Any, Optional, Union

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入实际依赖 (假设它们存在于 metai 库中)
from metai.utils import MetLabel, MLOGI
from metai.dataset.met_dataloader_scwds import ScwdsDataModule
from metai.model.simvp import SimVPConfig, SimVP
from metai.utils.met_config import get_config

# 假设这些常量应从配置文件或环境变量获取
# 实际代码中，这些值应由 met_config 或配置对象提供
USER_ID = "YourUserID" 
TRACK_ID = "TestSet"
TIME_STEP_MINUTES = 6 


def find_latest_ckpt(save_dir: str) -> str:
    """查找最新的或 last.ckpt 文件"""
    last = os.path.join(save_dir, 'last.ckpt')
    if os.path.exists(last):
        return last
    cpts = sorted(glob.glob(os.path.join(save_dir, '*.ckpt')))
    if len(cpts) == 0:
        raise FileNotFoundError(f'No checkpoint found in {save_dir}')
    return cpts[-1]


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Infer SCWDS SimVP Model')
    parser.add_argument('--data_path', type=str, default='data/samples.jsonl',
                         help='Path to test data')
    # in_shape 必须匹配训练时的配置
    parser.add_argument('--in_shape', type=int, nargs=4, default=[20, 29, 128, 128], 
                         help='Input shape: T C H W (must match training config)') 
    parser.add_argument('--save_dir', type=str, default='./output/simvp',
                         help='Directory where model checkpoints are saved')
    parser.add_argument('--accelerator', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                         help='Accelerator to run inference on (e.g., cuda, cpu, cuda:0)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 获取配置对象
    # ⚠️ NOTE: 确保 get_config() 返回 MetConfig 实例
    met_config = get_config() 
    
    # 2. 构建 SimVPConfig
    config = SimVPConfig(
        data_path=args.data_path,
        in_shape=tuple(args.in_shape),
        save_dir=args.save_dir,
    )
    
    # 3. 设置设备和常数
    resize_shape = (config.in_shape[2], config.in_shape[3])
    device = torch.device(args.accelerator)
    
    # 获取 MetLabel.RA.max 和文件日期格式
    RA_MAX = MetLabel.RA.max 
    FMT = met_config.file_date_format
    
    # 4. 数据模块
    data_module = ScwdsDataModule(
        data_path=config.data_path,
        resize_shape=resize_shape,
        batch_size=1,
        num_workers=1
    )
    data_module.setup('infer') # 确保设置 infer dataset
    infer_loader = data_module.infer_dataloader()
    assert infer_loader is not None
    
    MLOGI(f"推理数据加载器已创建，样本数量: {len(data_module.infer_dataset)}")
    
    # 5. 加载模型
    ckpt_path = find_latest_ckpt(config.save_dir)
    MLOGI(f"加载检查点: {ckpt_path}")
    
    # 使用 map_location=device 确保模型加载到正确的设备
    model = SimVP.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval().to(device)

    with torch.no_grad():
        for bidx, batch in enumerate(infer_loader):
            # 6. 处理 Batch (3 元素: metadata, input_data, input_mask)
            metadata_list, batch_x, input_mask = batch
            
            batch_x = batch_x.to(device) 
            
            # 7. 调用 model.infer_step
            # model.infer_step 内部会进行 GPU 插值和前向传播，返回 [1, T, C=1, H, W]
            # 注意: infer_step 签名只接收 2 个位置参数: (batch, batch_idx)
            batch_y = model.infer_step(batch, batch_idx=bidx) 
            
            # 去掉 batch 和 channel 维度
            batch_y = batch_y.squeeze() # [20, H, W]
            
            metadata = metadata_list[0]
            MLOGI(f"[INFO] No.{bidx} sample_id = {metadata['sample_id']}")

            sample_id = metadata['sample_id']
            sample_id_parts = sample_id.split('_')
            
            # 8. 元数据解析
            task_id = metadata.get('task_id') or sample_id_parts[0]
            region_id = metadata.get('region_id') or sample_id_parts[1]
            time_id = sample_id_parts[2] 
            station_id = metadata.get('station_id') or sample_id_parts[3]
            case_id = metadata.get('case_id') or '_'.join(sample_id_parts[:4])
            timestamps = metadata.get('timestamps')
            
            if timestamps is None or not timestamps:
                 MLOGI(f"[WARN] timestamps missing for {sample_id}, skipping.")
                 continue

            last_obs_time_str = timestamps[-1]
            try:
                last_obs_dt = datetime.strptime(last_obs_time_str, FMT)
            except ValueError:
                 MLOGI(f"[ERROR] Date format error for {last_obs_time_str}. Skipping.")
                 continue
            
            # 9. 结果处理和保存
            for idx, y in enumerate(batch_y):
                # 上采样到 301x301 (Bilinear)
                y_interp = F.interpolate(
                    y.unsqueeze(0).unsqueeze(0), 
                    size=(301, 301),
                    mode='bilinear',
                    align_corners=False
                ).squeeze() 

                # 反归一化到 [0, RA_MAX] mm/h 范围
                y_mm = y_interp * RA_MAX
                
                # 确保最低预测值大于竞赛要求的最低阈值 (0.003 * 30 = 0.09 mm/h)
                y_mm = torch.clamp(y_mm, min=0.003) 

                y_final_np = y_mm.cpu().numpy().astype(np.float32) # 保持 Float32 精度

                # 计算预报时间
                forecast_dt = last_obs_dt + timedelta(minutes=TIME_STEP_MINUTES * (idx + 1))
                forecast_time_str = forecast_dt.strftime(FMT)
                
                # 构建最终文件路径
                npy_dir_final = os.path.join(
                    'submit',
                    'output',
                    USER_ID, 
                    TRACK_ID, 
                    case_id
                )
                os.makedirs(npy_dir_final, exist_ok=True)
                
                npy_path = os.path.join(
                    npy_dir_final,
                    f"{task_id}_{region_id}_{time_id}_{station_id}_Forcast_{forecast_time_str}.npy"
                )
                
                # 保存 numpy 文件
                np.save(npy_path, y_final_np)
                MLOGI(f'[INFO] 已保存 npy 文件: {os.path.basename(npy_path)}')
            
            # 如果需要限制样本数量，在此处 break
            # if bidx >= max_samples_to_process: break
            
    MLOGI("[INFO] 推理完成！所有文件已保存到提交目录。")


if __name__ == '__main__':
    # ⚠️ NOTE: 在运行之前，请确保 USER_ID 和 TRACK_ID 变量在实际环境中是可访问的或已设置。
    main()