import os
import sys
import argparse
import csv

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.utils import scan_directory
from metai.dataset import MetCase
from metai.utils import MetLabel, get_config, MLOG

def main():
    parser = argparse.ArgumentParser(description='天气过程列表生成脚本')
    parser.add_argument('-d', '--debug', action='store_true', help='启用调试模式 (默认: False)')
    parser.add_argument('-v', '--version', type=str, default='v1', help='任务版本')
    args = parser.parse_args()
    
    # 确定是否为调试模式
    is_debug = True if args.debug else False

    config = get_config(is_debug=is_debug)
    root_path = os.path.join(config.root_path, "CP", "TrainSet")
    case_ids = scan_directory(root_path, 2, return_full_path=False)

    output_file = os.path.join("data", args.version, "cases.raw.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['case_id', 'size'])
        
        for case_id in case_ids[:]:
            case = MetCase.create(case_id, config=config, is_debug=is_debug)
            
            if case.region_id in ["98", "99"]:
                continue
            
            folder = os.path.join(root_path, case.region_id, case.case_id, MetLabel.RA.parent, MetLabel.RA.value)
            MLOG(folder)
            if os.path.exists(folder):
                size = len(os.listdir(folder))     
                writer.writerow([case_id, size])
    
    MLOG(f"已保存case_id到 {output_file}")
        
if __name__ == "__main__":
    main()