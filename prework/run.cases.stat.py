import os
import sys
import argparse
import csv
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.dataset import MetCase
from metai.utils import MLOGE, MetLabel, MetRadar, MetNwp
from metai.utils import get_config, MetConfig


def do_task(data_type: MetLabel | MetRadar | MetNwp, config: MetConfig | None = None,version: str = 'v1'):
    if config is None:
        raise ValueError("config must be provided and cannot be None")
    root_path = os.path.join(config.root_path, "CP", "TrainSet")
    
    input_file = os.path.join("data", version, "cases.raw.csv")
    output_file = os.path.join("data", version, "statistics", data_type.parent, f"{data_type.value}.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
        open(output_file, 'w', encoding='utf-8') as outfile:
            
            reader = csv.reader(infile)
            # 使用自定义的CSV写入器，禁用引号
            writer = csv.writer(outfile, escapechar='\\')
            
            next(reader)  # 跳过首行（标题行）
            writer.writerow(['case_id', 'min', 'max', "length", 'unnormal_indices'])
            
            for i, row in enumerate(reader):
                # if i >= 10:  # 限制只处理前10行
                #     break
                case_id, length = row
                if int(length) < 40:
                    continue
                case = MetCase.create(case_id, config=config)
                folder = os.path.join(root_path, case.region_id, case.case_id, data_type.parent, data_type.value)
                # print(folder)
                results = []
                unnormal_files = []
                for j, file in enumerate(os.listdir(folder)):
                    if file.endswith('.npy'):
                        file_path = os.path.join(folder, file)
                        try:
                            data = np.load(file_path)
                            if "LABEL" in file_path:
                                data[np.where(data==-9)] = 0
                                if "RA" in file_path and  np.all(data <=0):
                                    unnormal_files.append(j)
                            elif "RADAR" in file_path:
                                data[np.where(data==-32768)] = 0
                                data[np.where(data==-1280)] = 0
                                if "CR" in file_path and np.all(data <=0):
                                    unnormal_files.append(j)
                            elif "NWP" in file_path:
                                data[np.where(data==-9999)] = 0
                            results.append(data.flatten())
                        except Exception as e:
                            print(f"Warning: Failed to load {file_path}: {e}")
                            continue
                
                if len(results) == 0:
                    MLOGE(f"Warning: No valid .npy files found in {folder}")
                    min_val = None
                    max_val = None
                else:
                    combined_data = np.concatenate(results)
                    min_val = combined_data.min()
                    max_val = combined_data.max()
                    
                # 处理异常文件列表，使用分号分隔（已禁用CSV引号）
                unnormal_files_str = ";".join([str(idx) for idx in unnormal_files]) if unnormal_files else ""
                writer.writerow([folder, min_val, max_val, length, unnormal_files_str])
                print(folder, min_val, max_val, length, unnormal_files_str)
                
    print(f"已保存 {data_type.value} 到 {output_file}")
        

def main():    
    parser = argparse.ArgumentParser(description='样本生成处理脚本')
    parser.add_argument('-d', '--debug', action='store_true', help='启用调试模式 (默认: False)')
    
    parser.add_argument('-v', '--version', type=str, default='v1', help='任务版本')
    
    parser.add_argument('--types', '-t', 
                       choices=['LABEL', 'NWP', 'RADAR', 'ALL'],
                       nargs='+',
                       default=['ALL'],
                       help='要处理的数据类型 (默认: ALL)')
    
    parser.add_argument('--label-vars', 
                       choices=[MetLabel.RA.value],
                       nargs='+',
                       help='要处理的LABEL变量')
    
    parser.add_argument('--nwp-vars', 
                       choices=[nwp.value for nwp in MetNwp],
                       nargs='+',
                       help='要处理的NWP变量')
    
    parser.add_argument('--radar-vars', 
                       choices=[radar.value for radar in MetRadar],
                       nargs='+',
                       help='要处理的RADAR变量')
    
    args = parser.parse_args()
    
    # 确定是否为调试模式
    is_debug = True if args.debug else False
    version = args.version
    types = args.types
    
    config = get_config(is_debug=is_debug)
    
    print(config)
    # return
        
    if 'ALL' in args.types or 'LABEL' in types:
        label_vars = [MetLabel(var) for var in list(MetLabel)]
        for label_var in label_vars:
            do_task(label_var, config, version)
            
    if 'ALL' in args.types or 'RADAR' in types:
        radar_vars = [MetRadar(var) for var in args.radar_vars] if args.radar_vars else list(MetRadar)
        for radar_var in radar_vars:
            do_task(radar_var, config, version)
            
    if 'ALL' in args.types or 'NWP' in types:
        nwp_vars = [MetNwp(var) for var in args.nwp_vars] if args.nwp_vars else list(MetNwp)
        for nwp_var in nwp_vars:
            do_task(nwp_var, config, version)

if __name__ == "__main__":
    main()