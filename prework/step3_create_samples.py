# prework/step3_create_samples.py
import os
import sys
import argparse
import csv
import json
import pandas as pd
from tqdm import tqdm
from typing import Dict, Set, List

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.dataset import MetCase
from metai.utils import get_config, MetRadar

def to_timestr_list(filenames: List[str]) -> List[str]:
    """
    从标签文件名中提取时间戳。
    
    Args:
        filenames: 文件名列表 (如 CP_Label_RA_Z9559_20180704-1213.npy)
    Returns:
        List[str]: 提取出的时间戳列表 (如 20180704-1213)
    """
    timestr_list = []
    for filename in filenames:
        name_without_ext = os.path.splitext(filename)[0]
        parts = name_without_ext.split('_')
        if len(parts) >= 1:
            timestr_list.append(parts[-1])
    return timestr_list

def load_unnormal_indices(stats_dir: str, var_names: List[str]) -> Dict[str, Set[int]]:
    """
    读取统计CSV文件，合并指定变量的异常索引。
    
    Returns:
        Dict[str, Set[int]]: { 'case_id': {index1, index2, ...} }
    """
    merged_indices: Dict[str, Set[int]] = {}
    
    print(f"Loading statistics for variables: {var_names}")
    
    for var in var_names:
        csv_path = os.path.join(stats_dir, f"{var}.csv")
        if not os.path.exists(csv_path):
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
    parser.add_argument('-v', '--version', type=str, default='v1', help='数据版本')
    parser.add_argument('--vars', nargs='+', default=['RA', 'CR', 'CAP30', 'CAP50', 'VIL', 'ET'], help='用于过滤异常帧的变量列表')
    parser.add_argument('--length', type=int, default=30, help='样本时序长度')
    parser.add_argument('--interval', type=int, default=10, help='滑动窗口步长')
    
    args = parser.parse_args()
    config = get_config()
    
    # 路径定义
    raw_cases_file = os.path.join("data", args.version, "cases.raw.csv")
    stats_dir = os.path.join("data", args.version, "statistics")
    output_file = os.path.join("data", args.version, f"samples.interval{args.interval}.jsonl")
    
    if not os.path.exists(raw_cases_file):
        print(f"Error: {raw_cases_file} not found.")
        return

    # 读取 Case 列表并进行长度初筛
    cases_df = pd.read_csv(raw_cases_file, header=0, names=['case_id', 'length'])
    cases_df = cases_df[cases_df['length'] >= args.length]
    
    print(f"Total valid cases candidates: {len(cases_df)}")

    # 加载异常帧索引
    unnormal_map = load_unnormal_indices(stats_dir, args.vars)

    total_samples = 0
    valid_cases_count = 0
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Generating samples -> {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        
        for _, row in tqdm(cases_df.iterrows(), total=len(cases_df), desc="Sampling"):
            case_id = str(row['case_id'])
            bad_indices = unnormal_map.get(case_id, set())
            
            try:
                # 初始化 Case 并获取文件列表
                case = MetCase.create(case_id, config=config)
                file_list = sorted(case.label_files) 
                actual_len = len(file_list)
                
                if actual_len < args.length:
                    continue
                
                # 触发雷达类型自动检测
                # MetCase 默认为 'SA'，需通过验证文件存在性来修正为实际类型 (如 'SB', 'SC')
                if actual_len > 0:
                    first_ts = case._extract_timestamp_from_label_file(file_list[0])
                    if first_ts:
                        case._is_radar_file_valid(first_ts, MetRadar.CR)

            except Exception as e:
                print(f"Error loading case {case_id}: {e}")
                continue

            # 滑动窗口生成样本
            case_sample_count = 0
            
            for start_idx in range(0, actual_len - args.length + 1, args.interval):
                end_idx = start_idx + args.length
                
                # 检查窗口内是否包含异常帧
                current_indices = set(range(start_idx, end_idx))
                if not current_indices.isdisjoint(bad_indices):
                    continue
                
                # 构建样本数据
                sample_files = file_list[start_idx : end_idx]
                timestamps = to_timestr_list(sample_files)
                
                # 构造符合 MetSample 解析规则的 ID
                # 格式: CaseID_RadarType_SampleNo (例如: CP_AH_..._SA_001)
                sample_obj = {
                    "sample_id": f"{case_id}_{case.radar_type}_{case_sample_count:03d}",
                    "timestamps": timestamps
                }
                
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