# prework/step3_create_samples.py
import os
import sys
import argparse
import csv
import json
import pandas as pd
from tqdm import tqdm
from typing import Dict, Set, List

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.dataset import MetCase
from metai.utils import get_config

def to_timestr_list(filenames: List[str]) -> List[str]:
    """
    从标签文件名中提取时间戳
    文件名示例：CP_Label_RA_Z9559_20180704-1213.npy
    """
    timestr_list = []
    for filename in filenames:
        # 去掉扩展名
        name_without_ext = os.path.splitext(filename)[0]
        # 获取最后一部分作为时间戳
        parts = name_without_ext.split('_')
        if len(parts) >= 1:
            date_time = parts[-1]
            timestr_list.append(date_time)
    return timestr_list

def load_unnormal_indices(stats_dir: str, var_names: List[str]) -> Dict[str, Set[int]]:
    """
    读取统计CSV文件，合并指定变量的异常索引
    返回格式: { 'case_id': {0, 1, 5, ...} }
    """
    merged_indices: Dict[str, Set[int]] = {}
    
    print(f"Loading statistics for variables: {var_names}")
    
    for var in var_names:
        csv_path = os.path.join(stats_dir, f"{var}.csv")
        if not os.path.exists(csv_path):
            # 如果某个变量的统计文件不存在，打印警告但继续处理
            print(f"Warning: Statistics file not found: {csv_path}")
            continue
            
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                case_id = row['case_id']
                indices_str = row['unnormal_indices']
                
                if case_id not in merged_indices:
                    merged_indices[case_id] = set()
                
                # 解析异常索引字符串 (格式如 "0;1;5")
                if indices_str and indices_str.strip():
                    try:
                        indices = {int(idx) for idx in indices_str.split(';')}
                        merged_indices[case_id].update(indices)
                    except ValueError:
                        pass
                        
    return merged_indices

def main():
    parser = argparse.ArgumentParser(description='根据统计结果生成训练样本 (JSONL)')
    parser.add_argument('-v', '--version', type=str, default='v1', help='数据版本 (default: v1)')
    # 默认同时检查 RA(降水) 和 CR(组合反射率) 的异常情况
    parser.add_argument('--vars', nargs='+', default=['RA', 'CR', 'CAP30', 'CAP50', 'VIL', 'ET'], help='用于过滤异常帧的变量列表')
    parser.add_argument('--length', type=int, default=30, help='单个样本的时序长度 (default: 30)')
    parser.add_argument('--interval', type=int, default=10, help='滑动窗口步长 (default: 10)')
    
    args = parser.parse_args()
    config = get_config()
    
    # 路径定义
    raw_cases_file = os.path.join("data", args.version, "cases.raw.csv")
    stats_dir = os.path.join("data", args.version, "statistics")
    output_file = os.path.join("data", args.version, f"samples.interval{args.interval}.jsonl")
    
    # 1. 基础检查
    if not os.path.exists(raw_cases_file):
        print(f"Error: {raw_cases_file} not found.")
        return

    # 读取所有 Case ID 和长度
    cases_df = pd.read_csv(raw_cases_file, header=0, names=['case_id', 'length'])
    # 初步过滤掉长度不足样本长度的 Case
    cases_df = cases_df[cases_df['length'] >= args.length]
    
    print(f"Total valid cases candidates: {len(cases_df)}")

    # 2. 读取并合并异常索引 map
    unnormal_map = load_unnormal_indices(stats_dir, args.vars)

    total_samples = 0
    valid_cases_count = 0
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 3. 生成样本并写入
    print(f"Generating samples -> {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        
        for _, row in tqdm(cases_df.iterrows(), total=len(cases_df), desc="Sampling"):
            case_id = str(row['case_id'])
            # 这里的长度是 CSV 里的记录，为了安全，后面会校验实际文件数
            
            # 获取该 Case 的所有异常帧索引 (默认为空集合)
            bad_indices = unnormal_map.get(case_id, set())
            
            try:
                # 只需要 Label 的文件列表来作为时间戳基准
                case = MetCase.create(case_id, config=config)
                file_list = sorted(case.label_files) 
                actual_len = len(file_list)
                
                # 如果实际文件数少于样本长度，跳过
                if actual_len < args.length:
                    continue
                    
            except Exception as e:
                print(f"Error loading case {case_id}: {e}")
                continue

            # 开始滑动窗口处理
            case_sample_count = 0
            
            # 窗口范围: range(0, actual_len - length + 1, interval)
            for start_idx in range(0, actual_len - args.length + 1, args.interval):
                end_idx = start_idx + args.length
                
                # 生成当前窗口内的所有索引集合
                current_indices = set(range(start_idx, end_idx))
                
                # 检查交集：isdisjoint 返回 True 表示没有交集（即没有异常帧）
                if not current_indices.isdisjoint(bad_indices):
                    continue
                
                # --- 生成有效样本 ---
                
                # 1. 提取对应的文件名切片
                sample_files = file_list[start_idx : end_idx]
                
                # 2. 转换为时间戳列表
                timestamps = to_timestr_list(sample_files)
                
                # 3. 构建样本对象 (不包含 case_id)
                sample_obj = {
                    "sample_id": f"{case_id}_{case_sample_count:03d}",
                    "timestamps": timestamps
                }
                
                # 4. 写入文件 (JSONL)
                json.dump(sample_obj, f_out, ensure_ascii=False)
                f_out.write('\n')
                
                case_sample_count += 1
                total_samples += 1
            
            if case_sample_count > 0:
                valid_cases_count += 1

    print(f"\nDone.")
    print(f"Processed {len(cases_df)} cases.")
    print(f"Generated {total_samples} samples from {valid_cases_count} valid cases.")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()