#!/usr/bin/env python3
"""
遍历data/qc/raw下csv文件，从第一列提取case_id，然后从data/cases.csv删除对应的case_id，生成新的csv
"""

import os, sys
import argparse
import pandas as pd

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def extract_case_id_from_path(path_string):
    """
    从路径字符串中提取case_id
    例如: /data/zjobs/SevereWeather_AI_2025/CP/TrainSet/AH/CP_AH_201604021547_Z9551/LABEL/RA
    提取: CP_AH_201604021547_Z9551
    """
    # 使用正则表达式匹配case_id模式
    parts = path_string.split(os.sep)
    return parts[-3]

def main():
    parser = argparse.ArgumentParser(description='天气过程列表重建脚本')
    parser.add_argument('-d', '--debug', action='store_true', help='启用调试模式 (默认: False)')
    parser.add_argument('-v', '--version', type=str, default='v1', help='任务版本')
    args = parser.parse_args()
    
    version = args.version
    
    qc_raw_dir = os.path.join("data", version, "qc")
    cases_raw_file = os.path.join("data", version, "cases.raw.csv")
    output_file = os.path.join("data", version, "cases.csv")
    
    
    # 检查目录和文件是否存在
    if not os.path.exists(qc_raw_dir):
        print(f"错误: 目录 {qc_raw_dir} 不存在")
        return
    
    if not os.path.exists(cases_raw_file):
        print(f"错误: 文件 {cases_raw_file} 不存在")
        return
    
    # 读取cases.csv
    print("正在读取 cases.csv...")
    cases_df = pd.read_csv(cases_raw_file)
    print(f"cases.csv 包含 {len(cases_df)} 条记录")
    
    # 收集所有提取的case_id
    all_extracted_case_ids = set()
    
    # 遍历qc/raw目录下的所有CSV文件
    csv_files = [os.path.join(qc_raw_dir, file) for file in os.listdir(qc_raw_dir) if file.endswith('.csv')]
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    for csv_file in csv_files:
        print(f"处理文件: {csv_file}")
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 从第一列提取case_id
            if len(df) > 0 and 'case_id' in df.columns:
                for path_string in df['case_id']:
                    case_id = extract_case_id_from_path(str(path_string))
                    if case_id:
                        all_extracted_case_ids.add(case_id)
                        print(f"  提取到 case_id: {case_id}")
                    else:
                        print(f"  警告: 无法从路径提取case_id: {path_string}")
            else:
                print(f"  警告: 文件 {csv_file} 为空或没有case_id列")
                
        except Exception as e:
            print(f"  错误: 处理文件 {csv_file} 时出错: {e}")
    
    print(f"\n总共提取到 {len(all_extracted_case_ids)} 个唯一的case_id")
    
    # 从cases.csv中删除匹配的记录
    print("正在从cases.csv中删除匹配的记录...")
    remaining_cases = cases_df[~cases_df['case_id'].isin(all_extracted_case_ids)]
    deleted_cases = cases_df[cases_df['case_id'].isin(all_extracted_case_ids)]
    
    print(f"删除了 {len(deleted_cases)} 条记录")
    print(f"剩余 {len(remaining_cases)} 条记录")
    
    # 保存结果
    remaining_cases.to_csv(output_file, index=False)
    print(f"结果已保存到: {output_file}")
    
    # 显示一些统计信息
    print(f"\n统计信息:")
    print(f"- 原始cases.csv记录数: {len(cases_df)}")
    print(f"- 提取的唯一case_id数: {len(all_extracted_case_ids)}")
    print(f"- 删除的记录数: {len(deleted_cases)}")
    print(f"- 剩余的记录数: {len(remaining_cases)}")
    print(f"- 删除率: {len(deleted_cases)/len(cases_df)*100:.2f}%")
    
    # 显示前几条剩余的记录
    if len(remaining_cases) > 0:
        print(f"\n前5条剩余的记录:")
        print(remaining_cases.head())

if __name__ == "__main__":
    main()
