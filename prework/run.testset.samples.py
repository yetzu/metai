import os
import sys
import argparse
import json
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.utils import MLOGE, MLOGI, get_config
from metai.dataset import MetCase
from metai.utils import scan_directory

def save_samples(samples_data, output_file="data/samples.testset.jsonl"):
    """保存样本数据到JSONL文件"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # 使用 'a' 模式追加写入
    with open(output_file, 'a', encoding='utf-8') as f:
        for sample in samples_data:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

def to_timestr_list(filenames: list[str]):
    """从标签文件名中提取时间戳"""
    timestr_list = []
    for filename in sorted(filenames):
        name_without_ext = filename.replace('.npy', '')
        parts = name_without_ext.split('_')
        # 文件名示例: CP_Label_RA_Z9559_20180704-1213.npy
        if len(parts) >= 4:
            date_time = parts[-1]
            timestr_list.append(date_time)
    return timestr_list

def generate_future_timestamps(last_timestamp_str, fmt, count=20, interval_minutes=6):
    """
    [新增] 根据最后一个观测时间，推算未来的时间戳
    Args:
        last_timestamp_str: 最后一个观测时间字符串
        fmt: 时间格式 (从 config 载入)
        count: 推算步数 (默认20步)
        interval_minutes: 时间间隔 (默认6分钟)
    """
    try:
        last_time = datetime.strptime(last_timestamp_str, fmt)
    except ValueError:
        MLOGE(f"时间格式解析失败: {last_timestamp_str} with fmt {fmt}")
        return []
    
    future_timestamps = []
    for i in range(count):
        # 每次递增 6 分钟 (或指定间隔)
        future_time = last_time + timedelta(minutes=interval_minutes * (i + 1))
        future_timestamps.append(future_time.strftime(fmt))
    return future_timestamps

def main():
    parser = argparse.ArgumentParser(description='天气过程列表生成脚本 (TestSet)')
    parser.add_argument('-d', '--debug', action='store_true', help='启用调试模式')
    parser.add_argument('-v', '--version', type=str, default='v1', help='任务版本')
    args = parser.parse_args()

    is_debug = True if args.debug else False
    
    # 1. 获取全局配置
    config = get_config(is_debug=is_debug)
    
    # 2. 从配置中载入日期格式
    # metai/config.yaml 中配置了 file_date_format: "%Y%m%d-%H%M"
    date_fmt = config.file_date_format 
    
    # 扫描测试集目录
    root_path = os.path.join(config.root_path, "CP", "TestSet")
    case_ids = scan_directory(root_path, 2, return_full_path=False)
    
    # 每次运行前清理旧文件，防止重复追加
    output_file = "data/samples.testset.jsonl"
    if os.path.exists(output_file):
        os.remove(output_file)
        MLOGI(f"已清理旧样本文件: {output_file}")
    
    total_cnt = right_cnt = sample_cnt = 0

    for case_id in case_ids[:]:
        # 创建 Case 对象
        case = MetCase.create(case_id, config, is_train=False, test_set="TestSet")
        
        # 获取基础样本（过去10帧）
        # to_infer_sample 内部加载文件列表，默认返回最后 sample_length 个文件
        seqs = case.to_infer_sample(sample_length=10)
        
        if seqs:
            case_samples = []
            for idx, seq_files in enumerate(seqs):
                # A. 提取过去 10 帧的时间戳
                past_timestamps = to_timestr_list(seq_files)
                
                if not past_timestamps:
                    continue

                # B. 推算未来 20 帧的时间戳 (使用 config 中的 fmt)
                last_ts = past_timestamps[-1]
                future_timestamps = generate_future_timestamps(
                    last_ts, 
                    fmt=date_fmt, 
                    count=20, 
                    interval_minutes=6
                )
                
                if not future_timestamps:
                    MLOGE(f"Case {case_id}: 未来时间戳生成失败")
                    continue

                # C. 合并时间戳 (共 30 个: 10过去 + 20未来)
                full_timestamps = past_timestamps + future_timestamps
                
                # D. 构造样本
                case_samples.append({
                    "sample_id": f"{case_id}_{case.radar_type}_{(idx+1):03d}",
                    "timestamps": full_timestamps
                })

            if case_samples:
                save_samples(case_samples, output_file)
                right_cnt += 1
                sample_cnt += len(case_samples)
                MLOGI(f"Case {case_id}: 生成 {len(case_samples)} 个样本 (10过去 + 20未来)")
        
        total_cnt += 1
    
    MLOGI(f"总个例: {total_cnt}，有效: {right_cnt}，总样本: {sample_cnt}")

if __name__ == '__main__':
    main()