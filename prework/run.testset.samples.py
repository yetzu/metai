import os
import sys
import argparse
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.utils import MLOGE, MLOGI, get_config
from metai.dataset import MetCase
from metai.utils import scan_directory

def save_samples(samples_data, output_file="data/samples.testset.jsonl"):
    """保存样本数据到JSONL文件，支持高效追加写入"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # 直接追加写入，无需读取现有数据
    with open(output_file, 'a', encoding='utf-8') as f:
        for sample in samples_data:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')  # 每行一个JSON对象

def load_samples(file_path="sample_data.testset.jsonl"):
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
    parser = argparse.ArgumentParser(description='天气过程列表生成脚本')
    parser.add_argument('-d', '--debug', action='store_true', help='启用调试模式 (默认: False)')
    parser.add_argument('-v', '--version', type=str, default='v1', help='任务版本')
    args = parser.parse_args()

    is_debug = True if args.debug else False
    version = args.version

    config = get_config(is_debug=is_debug)
    config.root_path = config.test_root_path
    root_path = os.path.join(config.test_root_path, "CP", "TestSet")
    
    print(root_path)
    
    case_ids = scan_directory(root_path, 2, return_full_path=False)
    
    total_cnt = right_cnt = sample_cnt = 0

    for case_id in case_ids[:]:
        case = MetCase.create(case_id, config, is_train=False)
        
        sample = case.to_infer_sample(sample_length=20)
        MLOGI(f"Case id: {case_id}，标签文件数量: {len(case.label_files)}，样本数: {len(sample[0])}")
        if sample:
            case_samples = [{"sample_id": f"{case_id}_{case.radar_type}_{(idx+1):03d}", "timestamps": to_timestr_list(sample)} for idx, sample in enumerate(sample)]
            save_samples(case_samples)
            right_cnt += 1
            sample_cnt += 1
        
        total_cnt += 1
    
    MLOGI(f"总个例: {total_cnt}，有效个例: {right_cnt} ({100*right_cnt/total_cnt:.1f}%)，总样本: {sample_cnt}")

if __name__ == '__main__':
    main()