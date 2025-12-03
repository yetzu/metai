import os
import sys
import argparse
import csv
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.utils import scan_directory
from metai.dataset import MetCase
from metai.utils import get_config, MLOG

def main():
    parser = argparse.ArgumentParser(description='Generate weather case list')
    parser.add_argument('-v', '--version', type=str, default='v1', help='Task version')
    args = parser.parse_args()

    # 1. 获取全局配置，避免在循环中重复加载
    config = get_config()
    root_path = os.path.join(config.root_path, "CP", "TrainSet")
    
    print(f"Scanning directory: {root_path} ...")
    case_ids = scan_directory(root_path, 2, return_full_path=False)
    print(f"Found {len(case_ids)} cases.")

    output_file = os.path.join("data", args.version, "cases.raw.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    valid_count = 0
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['case_id', 'size'])
        
        for case_id in tqdm(case_ids, desc="Processing Cases"):
            try:
                # 优化 1: 在创建对象前进行字符串过滤，减少开销
                parts = case_id.split('_')
                if len(parts) > 1 and parts[1] in ["98", "99"]:
                    continue

                # 优化 2: 使用 MetCase.create，传入已加载的 config
                case = MetCase.create(case_id, config=config)
                
                # 优化 3: 直接使用 MetCase 封装好的属性
                # 访问 label_files 会触发类内部的懒加载和 scandir
                size = len(case.label_files)
                
                if size > 0:
                    writer.writerow([case_id, size])
                    valid_count += 1
                        
            except Exception as e:
                # 打印错误类型，便于排查
                print(f"[ERROR] Processing {case_id}: {e}")
                continue
    
    MLOG(f"Processing complete. {valid_count} valid cases found.")
    MLOG(f"Result saved to {output_file}")
        
if __name__ == "__main__":
    main()