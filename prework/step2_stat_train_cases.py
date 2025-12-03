# prework/step2_stat_train_cases.py
import os
import sys
import argparse
import csv
import numpy as np

from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.dataset import MetCase
from metai.utils import MLOGE, MetLabel, MetRadar, MetNwp
from metai.utils import get_config, MetConfig

def process_single_file(file_path: str, file_idx: int) -> tuple[float, float, bool]:
    """
    处理单个npy文件，返回 (min, max, is_unnormal)
    """
    try:
        data = np.load(file_path)
        
        # ==========================================
        # 1. 数据清洗 (Data Cleaning) - 必须放在统计计算之前
        # ==========================================
        
        # --- NWP 处理 ---
        if "NWP" in file_path:
            data[data == -9999] = 0
            
        # --- LABEL 处理 ---
        if "LABEL" in file_path:
            data[data == -9] = 0
            
        # --- RADAR 处理 (int16) ---
        elif "RADAR" in file_path:
            data[data == -32768] = 0  
            # 处理 -1280
            data[data == -1280] = 0
            
        # ==========================================
        # 2. 计算统计量 (Statistics) - 清洗后计算
        # ==========================================
        
        # 注意：如果数据全被清洗成了0，或者本来就是空，min/max 都是 0
        current_min = float(np.min(data))
        current_max = float(np.max(data))
        is_unnormal = False

        # ==========================================
        # 3. 异常标记逻辑 (Flagging)
        # ==========================================
        
        if "LABEL" in file_path:
            if os.path.sep + "RA" + os.path.sep in file_path:
                if current_max <= 0 or current_max >= 400:
                    is_unnormal = True
        elif "RADAR" in file_path:
            check_vars = ["CR", "CAP20", "CAP30", "CAP40", "CAP50", "CAP60", "CAP70", "HBR"]
            if any(var in file_path for var in check_vars) and current_max <= 0:
                is_unnormal = True
                
        return current_min, current_max, is_unnormal

    except Exception as e:
        print(f"Warning: Failed to load {file_path}: {e}")
        return None, None, False  # pyright: ignore[reportReturnType]

def do_task(data_type: MetLabel | MetRadar | MetNwp, config: MetConfig | None = None, version: str = 'v1'):
    if config is None:
        raise ValueError("config must be provided and cannot be None")
    
    root_path = os.path.join(config.root_path, "CP", "TrainSet")
    input_file = os.path.join("data", version, "cases.raw.csv")
    output_file = os.path.join("data", version, "statistics", f"{data_type.value}.csv")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 预先读取所有行，以便使用 tqdm 显示总进度
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = list(csv.reader(f))
    
    header = lines[0]
    data_rows = lines[1:]

    print(f"Processing {data_type.value} -> {output_file}")

    with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        # 使用 newline='' 配合 csv 模块是标准做法
        writer = csv.writer(outfile, escapechar='\\')
        writer.writerow(['case_id', 'min', 'max', "length", 'unnormal_indices'])
        
        # 使用 tqdm 包装循环
        for i, row in enumerate(tqdm(data_rows, desc=f"Stats {data_type.value}")):
            if len(row) < 2: continue
            case_id, length_str = row
            length = int(length_str)
            
            if length < 40:
                continue

            # 初始化该 Case 的统计变量
            # 使用 infinity 确保第一次比较能正确更新
            case_global_min = float('inf')
            case_global_max = float('-inf')
            unnormal_files = []
            has_valid_data = False

            case = MetCase.create(case_id, config=config)
            folder = os.path.join(root_path, case.region_id, case.case_id, data_type.parent, data_type.value)
            
            if not os.path.exists(folder):
                # MLOGE(f"Warning: Folder not found {folder}")
                continue

            # 获取排好序的文件列表
            npy_files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
            
            if not npy_files:
                MLOGE(f"Warning: No .npy files in {folder}")
                # 依然写入一行，但 min/max 为空，保持格式一致性
                writer.writerow([case.case_id, "0", "0", length, ";".join([str(i) for i in range(length)])])
                continue

            for j, file_name in enumerate(npy_files):
                file_path = os.path.join(folder, file_name)
                
                f_min, f_max, is_unnormal = process_single_file(file_path, j)
                
                if f_min is not None: # 读取成功
                    has_valid_data = True
                    if f_min < case_global_min: case_global_min = f_min
                    if f_max > case_global_max: case_global_max = f_max
                    
                    if is_unnormal:
                        unnormal_files.append(j)
            
            if has_valid_data:
                unnormal_files_str = ";".join(map(str, unnormal_files)) if unnormal_files else ""
                writer.writerow([case.case_id, case_global_min, case_global_max, length, unnormal_files_str])
            else:
                writer.writerow([case.case_id, "", "", length, ""])

    print(f"Saved {data_type.value} to {output_file}")

def main():    
    parser = argparse.ArgumentParser(description='样本生成处理脚本')
    parser.add_argument('-v', '--version', type=str, default='v1', help='任务版本')
    parser.add_argument('--types', '-t', choices=['LABEL', 'NWP', 'RADAR', 'ALL'], nargs='+', default=['ALL'], help='要处理的数据类型')
    
    # 动态获取枚举值列表
    label_choices = [x.value for x in MetLabel]
    nwp_choices = [x.value for x in MetNwp]
    radar_choices = [x.value for x in MetRadar]

    parser.add_argument('--label-vars', choices=label_choices, nargs='+', help='要处理的LABEL变量')
    parser.add_argument('--nwp-vars', choices=nwp_choices, nargs='+', help='要处理的NWP变量')
    parser.add_argument('--radar-vars', choices=radar_choices, nargs='+', help='要处理的RADAR变量')
    
    args = parser.parse_args()
    config = get_config()
    print(f"Config Loaded. Root: {config.root_path}")
    
    types = args.types
    version = args.version

    # 处理 LABEL
    if 'ALL' in types or 'LABEL' in types:
        # 如果用户指定了具体的 vars 则使用指定的，否则使用全部
        targets = [MetLabel(v) for v in args.label_vars] if args.label_vars else list(MetLabel)
        for var in targets:
            do_task(var, config, version)
            
    # 处理 RADAR
    if 'ALL' in types or 'RADAR' in types:
        targets = [MetRadar(v) for v in args.radar_vars] if args.radar_vars else list(MetRadar)
        for var in targets:
            do_task(var, config, version)
            
    # 处理 NWP
    if 'ALL' in types or 'NWP' in types:
        targets = [MetNwp(v) for v in args.nwp_vars] if args.nwp_vars else list(MetNwp)
        for var in targets:
            do_task(var, config, version)

if __name__ == "__main__":
    main()