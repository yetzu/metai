import os
import sys
import csv
import numpy as np
import argparse
from tqdm import tqdm

# 添加项目根目录到Python路径，确保能导入 metai 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.dataset import MetCase
from metai.utils import get_config

def is_file_all_zero(file_path: str) -> bool:
    """
    判断单个文件是否全为0 (应用了 MetAI 的清洗逻辑)
    """
    try:
        data = np.load(file_path)
        
        # 参考 step2_stat_train_cases.py 的清洗逻辑
        # RA 属于 LABEL，异常值 -9 应视为 0
        if "LABEL" in file_path or "RA" in file_path:
            data[data == -9] = 0
            
        # 检查是否全为 0
        return not np.any(data)
    except Exception as e:
        print(f"Warning: Failed to load {file_path}: {e}")
        return False

def main():
    # 1. 初始化配置
    config = get_config()
    
    input_csv = "data/cases.test.csv"
    output_csv = "data/cases.test.error.csv"
    
    if not os.path.exists(input_csv):
        print(f"[Error] 输入文件不存在: {input_csv}")
        return

    # 2. 读取 CSV
    print(f"正在读取 {input_csv} ...")
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not rows:
        print("[Warning] CSV 为空")
        return

    if 'sample_id' not in rows[0]:
        print("[Error] CSV 中未找到 'sample_id' 列")
        return

    error_rows = []
    print(f"开始检查 {len(rows)} 个样本...")

    # 3. 遍历检查
    for row in tqdm(rows, desc="Checking Cases"):
        sample_id = row['sample_id']
        
        # 解析 case_id (假设 sample_id 格式为: CP_AH_2016..._Z9556_SA_001)
        # MetCase 需要标准的 case_id (前4部分)
        parts = sample_id.split('_')
        if len(parts) >= 4:
            case_id = "_".join(parts[:4])
        else:
            # 如果 sample_id 本身就是 case_id 格式
            case_id = sample_id

        try:
            # 创建 MetCase 实例 (指定为 TestSet)
            # 注意：根据你的目录结构，这里假设是 TestSet。如果是训练集请改为 is_train=True
            case = MetCase.create(case_id, config=config, is_train=False, test_set="TestSet")
            
            # 获取该 Case 下所有 RA 文件 (LABEL/RA)
            # _load_files 是 MetCase 的内部方法，但在此处调用最方便
            ra_files = case._load_files("LABEL", MetLabel.RA.name, return_full_path=True)
            
            if not ra_files:
                # print(f"Warning: No RA files found for {case_id}")
                continue

            # 核心逻辑：判断是否 *所有* RA 文件都全为 0
            # 只要发现有一个文件不全为0，这就不是一个"全0样本"
            is_all_zero_case = True
            for f_path in ra_files:
                if not is_file_all_zero(f_path):
                    is_all_zero_case = False
                    break
            
            if is_all_zero_case:
                error_rows.append(row)

        except Exception as e:
            # print(f"Error processing {sample_id}: {e}")
            continue

    # 4. 保存结果
    if error_rows:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(error_rows)
        
        print(f"\n[完成] 共发现 {len(error_rows)} 个全 0 样本。")
        print(f"结果已保存至: {output_csv}")
    else:
        print("\n[完成] 未发现全 0 样本。")

if __name__ == "__main__":
    main()