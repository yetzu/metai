import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.dataset import MetCase
from metai.utils import MLOGE, MLOGI, get_config

def plot_sample_row(sample_id: str, frames: List[np.ndarray], save_dir: str):
    """
    将30帧数据绘制成一行并保存
    """
    length = len(frames)
    # 创建一个 1行 x length列 的画布，宽度设置大一点以保证清晰度
    fig, axes = plt.subplots(1, length, figsize=(length * 2, 3))
    
    # 标题
    fig.suptitle(f"Sample: {sample_id} (RA Sequence)", fontsize=16)
    
    for i, ax in enumerate(axes):
        data = frames[i]
        
        # 简单的掩码处理：将0值设为白色，非0值显示颜色
        masked_data = np.ma.masked_where(data <= 0, data)
        
        # 绘图：使用 jet 颜色映射，vmax 设为 20 (或者根据数据动态调整) 以便观察强回波
        # origin='lower' 确保坐标原点在左下角 (符合气象习惯)
        im = ax.imshow(masked_data, cmap='jet', vmin=0, vmax=20, origin='lower')
        
        ax.axis('off') # 关闭坐标轴
        ax.set_title(f"T{i}", fontsize=8)

    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{sample_id}.png")
    plt.savefig(save_path, dpi=100)
    plt.close(fig) # 关闭画布释放内存

def find_file_by_timestamp(directory: str, timestamp: str) -> Optional[str]:
    """
    在目录下查找以 {timestamp}.npy 结尾的文件
    比硬拼文件名更安全，因为不同数据集文件前缀可能不同
    """
    suffix = f"_{timestamp}.npy"
    for f in os.listdir(directory):
        if f.endswith(suffix):
            return os.path.join(directory, f)
    return None

def verify_and_visualize(sample_obj: dict, config, version: str, save_dir: str) -> bool:
    """
    验证单个样本并可视化
    """
    sample_id = sample_obj['sample_id']
    timestamps = sample_obj['timestamps']
    
    # ========================================================
    # [修复] 从右边只切一次，确保获取完整的 case_id
    # 示例: CP_HB_..._Z9270_000 -> CP_HB_..._Z9270
    # ========================================================
    case_id = sample_id.rsplit('_', 1)[0]
    
    try:
        case = MetCase.create(case_id, config=config)
        # 获取 LABEL/RA 的目录
        label_dir = os.path.join(config.root_path, "CP", "TrainSet", case.region_id, case.case_id, "LABEL", "RA")
        
        if not os.path.exists(label_dir):
            # 尝试另一种常见的目录结构 (Label首字母大写问题)
            label_dir_alt = os.path.join(config.root_path, "CP", "TrainSet", case.region_id, case.case_id, "Label", "RA")
            if os.path.exists(label_dir_alt):
                label_dir = label_dir_alt
            else:
                MLOGE(f"Label dir not found: {label_dir}")
                return False

        frames_data = []
        is_valid_sample = True
        
        for ts in timestamps:
            # ========================================================
            # [修复] 不硬拼文件名，而是根据时间戳搜索文件
            # ========================================================
            file_path = find_file_by_timestamp(label_dir, ts)
            
            if not file_path:
                MLOGE(f"File missing for TS {ts} in {label_dir}")
                is_valid_sample = False
                break
            
            try:
                # 加载数据
                data = np.load(file_path)
            except Exception as e:
                MLOGE(f"Corrupt file {file_path}: {e}")
                is_valid_sample = False
                break
            
            # --- 数据清洗 (保持与 Step 2 一致的逻辑) ---
            data[data == -9] = 0
            
            # --- 核心验证逻辑: 最大值必须大于0 ---
            current_max = np.max(data)
            if current_max <= 0:
                MLOGE(f"Invalid Frame (All Zeros)! Sample: {sample_id}, TS: {ts}")
                is_valid_sample = False
                # 继续循环以便记录完整日志，或者 break 加速
                # break 
            
            frames_data.append(data)
            
        if is_valid_sample and len(frames_data) == len(timestamps):
            # 验证通过，进行可视化
            plot_sample_row(sample_id, frames_data, save_dir)
            return True
        else:
            return False

    except Exception as e:
        MLOGE(f"Error processing sample {sample_id} (Case: {case_id}): {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='验证并可视化生成的样本')
    parser.add_argument('-v', '--version', type=str, default='v1', help='数据版本')
    parser.add_argument('-i', '--interval', type=int, default=10, help='使用的 interval 版本 (默认找 samples.interval10.jsonl)')
    parser.add_argument('-n', '--num', type=int, default=10, help='随机抽样检查的数量 (默认: 10)')
    parser.add_argument('--all', action='store_true', help='检查所有样本 (覆盖 -n 参数)')
    
    args = parser.parse_args()
    config = get_config()
    
    # 输入文件
    jsonl_path = os.path.join("data", args.version, f"samples.interval{args.interval}.jsonl")
    if not os.path.exists(jsonl_path):
        MLOGE(f"JSONL file not found: {jsonl_path}")
        return

    # 输出目录
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"Validation images will be saved to: {os.path.abspath(tmp_dir)}")
    
    # 读取所有样本
    print(f"Loading {jsonl_path}...")
    all_samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_samples.append(json.loads(line))
    
    total_count = len(all_samples)
    print(f"Total samples found: {total_count}")
    
    # 确定要检查的样本列表
    if args.all:
        target_samples = all_samples
        print("Mode: Verify ALL samples.")
    else:
        # 随机抽取 n 个
        import random
        target_samples = random.sample(all_samples, min(args.num, total_count))
        print(f"Mode: Random verify {len(target_samples)} samples.")

    success_count = 0
    
    # 进度条循环
    for sample in tqdm(target_samples, desc="Verifying & Plotting"):
        if verify_and_visualize(sample, config, args.version, tmp_dir):
            success_count += 1
            
    print("-" * 30)
    print(f"Result: {success_count} / {len(target_samples)} samples passed verification.")
    print(f"Please check the '{tmp_dir}' directory for visualization results.")

if __name__ == "__main__":
    main()