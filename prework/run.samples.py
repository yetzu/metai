import os
import sys
import argparse
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.utils import MLOGE, MLOGI, get_config
from metai.dataset import MetCase

def save_samples(samples_data, sample_interval=10):
    """保存样本数据到JSONL文件，支持高效追加写入"""
    output_file = f"data/samples.interval{sample_interval}.jsonl"
    
    # 直接追加写入，无需读取现有数据
    with open(output_file, 'a', encoding='utf-8') as f:
        for sample in samples_data:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')  # 每行一个JSON对象

def load_samples(file_path="sample_data.jsonl"):
    """读取JSONL文件中的所有样本数据"""
    samples = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 跳过空行
                    samples.append(json.loads(line))
    except FileNotFoundError:
        MLOGE(f"文件 {file_path} 不存在")
    return samples

def to_timestr_list(filenames: list[str]):
    """
    从标签文件名中提取时间戳
    支持格式：CP_Label_RA_Z9559_20180704-1213.npy 或者 CP_Label_RA_00093_0511-1712.npy
    """
    timestr_list = []
    for filename in sorted(filenames):
        name_without_ext = filename.replace('.npy', '')
        parts = name_without_ext.split('_')
        if len(parts) >= 4:
            date_time = parts[-1]
            timestr_list.append(date_time)
    return timestr_list

def main():
    parser = argparse.ArgumentParser(description='样本生成处理脚本')
    parser.add_argument('-d', '--debug', action='store_true', help='启用调试模式 (默认: False)')
    args = parser.parse_args()
    
    # 确定是否为调试模式
    is_debug = True if args.debug else False
    config = get_config(is_debug=is_debug)
    
    print(config)
    
    total_cnt = right_cnt = sample_cnt = 0
    file_path = os.path.join("data", "cases.csv")
    df = pd.read_csv(file_path, header=0, names=['case_id', 'size'])
    case_ids = df['case_id'].tolist()

    sample_interval = 10
    for case_id in case_ids[:]:
        case = MetCase.create(case_id, config)
        samples = case.to_samples(sample_length=30, sample_interval=sample_interval)
        MLOGI(f"Case id: {case_id}，标签文件数量: {len(case.label_files)}，样本数: {len(samples)}")
        if samples:
            case_samples = [{"sample_id": f"{case_id}_{case.radar_type}_{(idx+1):03d}", "timestamps": to_timestr_list(sample)} 
                          for idx, sample in enumerate(samples)]
            save_samples(case_samples, sample_interval=sample_interval)
            right_cnt += 1
            sample_cnt += len(samples)
        
        total_cnt += 1
    
    MLOGI(f"sample_interval: {sample_interval}，总个例: {total_cnt}，有效个例: {right_cnt} ({100*right_cnt/total_cnt:.1f}%)，总样本: {sample_cnt}")
        
if __name__ == "__main__":
    main()