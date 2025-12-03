import os
import sys
import argparse
import json
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.utils import MLOGE, MLOGI, get_config, MetLabel
from metai.dataset import MetCase
from metai.utils import scan_directory

def save_samples(samples_data, output_file="data/samples.testset.jsonl"):
    """保存样本数据到JSONL文件"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'a', encoding='utf-8') as f:
        for sample in samples_data:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

def extract_timestamp_from_label(filename: str):
    """
    从Label文件名中提取时间戳
    文件名示例: CP_Label_RA_Z9559_20180704-1213.npy
    分割后 parts[-1] 为时间戳
    """
    name_without_ext = filename.replace('.npy', '')
    parts = name_without_ext.split('_')
    if len(parts) >= 4:
        return parts[-1]
    return None

def generate_future_timestamps(last_timestamp_str, fmt, count=20, interval_minutes=6):
    """
    根据最后一个观测时间，推算未来的时间戳
    """
    try:
        last_time = datetime.strptime(last_timestamp_str, fmt)
    except ValueError:
        MLOGE(f"时间格式解析失败: {last_timestamp_str} with fmt {fmt}")
        return []
    
    future_timestamps = []
    for i in range(count):
        # 每次递增 6 分钟
        future_time = last_time + timedelta(minutes=interval_minutes * (i + 1))
        future_timestamps.append(future_time.strftime(fmt))
    return future_timestamps

def main():
    parser = argparse.ArgumentParser(description='天气过程列表生成脚本 (TestSet)')
    parser.add_argument('-v', '--version', type=str, default='v1', help='任务版本')
    args = parser.parse_args()

    # 1. 获取全局配置
    config = get_config()
    date_fmt = config.file_date_format 
    
    # 2. 扫描测试集目录
    root_path = os.path.join(config.root_path, "CP", "TestSet")
    # 假设 scan_directory 返回 case_id 列表
    case_ids = scan_directory(root_path, 2, return_full_path=False)
    
    # 清理旧文件
    output_file = "data/samples.testset.jsonl"
    if os.path.exists(output_file):
        os.remove(output_file)
        MLOGI(f"已清理旧样本文件: {output_file}")
    
    total_cnt = right_cnt = sample_cnt = 0

    # 复制一份列表进行迭代，避免修改原列表（虽然这里没修改）
    for case_id in case_ids[:]:
        # 创建 Case 对象 (is_train=False 表示测试集模式)
        case = MetCase.create(case_id, config, is_train=False, test_set="TestSet")
        
        # [核心修改 1] 获取所有 RA 标签文件
        # MetCase._load_files 默认按文件名排序返回 (这里调用内部方法或通过属性访问)
        # 注意：这里我们使用 MetCase 中定义的 _load_files 方法，参数需对应
        all_label_files = case._load_files("LABEL", MetLabel.RA.name, return_full_path=False)
        
        # [核心修改 2] 校验文件数量，取最后 10 帧
        if len(all_label_files) < 10:
            MLOGE(f"Case {case_id}: 标签文件不足 10 个 (实际: {len(all_label_files)})")
            continue
            
        input_files = all_label_files[-10:]
        
        # 提取过去 10 帧的时间戳
        past_timestamps = []
        for f in input_files:
            ts = extract_timestamp_from_label(f)
            if ts:
                past_timestamps.append(ts)
        
        if len(past_timestamps) != 10:
            MLOGE(f"Case {case_id}: 时间戳解析错误，数量不匹配")
            continue

        # [核心修改 3] 基于最后一帧时间戳，生成未来 20 帧
        last_ts = past_timestamps[-1]
        future_timestamps = generate_future_timestamps(
            last_ts, 
            fmt=date_fmt, 
            count=20, 
            interval_minutes=6
        )
        
        if not future_timestamps:
            continue

        # 合并时间戳 (10过去 + 20未来)
        full_timestamps = past_timestamps + future_timestamps
        
        # 构造样本
        # 样本ID格式: CaseID_RadarType_001 (测试集每个Case通常只生成一个样本序列用于提交)
        sample_data = {
            "sample_id": f"{case_id}_{case.radar_type}_001",
            "timestamps": full_timestamps
        }

        save_samples([sample_data], output_file)
        
        right_cnt += 1
        sample_cnt += 1
        MLOGI(f"Case {case_id}: 样本生成成功 (Last TS: {last_ts})")
        
        total_cnt += 1
    
    MLOGI(f"处理完成 | 总个例: {total_cnt} | 有效生成: {right_cnt}")

if __name__ == '__main__':
    main()