import lmdb
import pickle
import os
import numpy as np
import zlib
import cv2
import json
import argparse
import sys
from tqdm import tqdm
from datetime import datetime, timedelta

# ==============================================================================
# 0. 环境设置与依赖
# ==============================================================================

# 添加项目根目录到路径，以便导入 metai 工具包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.utils import MLOGE, MLOGI, get_config
cfg = get_config()
print(cfg)
ROOT_PATH_DEFAULT = getattr(cfg, 'root_path', "/data/zjobs/SevereWeather_AI_2025/")

# ==============================================================================
# 1. 全局配置
# ==============================================================================

# 输入输出默认路径
JSONL_PATH_DEFAULT = "./data/samples.jsonl"
OUTPUT_DIR_DEFAULT = "/data/zjobs/LMDB"

# 图像处理参数
TARGET_SIZE = 256
MAP_SIZE = 1024 * 1024 * 1024 * 2048  # 2TB (LMDB 最大映射空间)
COMMIT_INTERVAL = 2000                # 每处理多少张提交一次事务

ZLIB_LEVEL = 1                        # 压缩等级 (1=最快, 9=最强)

# 文件命名与时间格式
DEFAULT_DATASET = "TrainSet"
DEFAULT_NWP_PREFIX = cfg.nwp_prefix
DATE_FMT = cfg.file_date_format

# 通道配置表: (父目录, 变量名, 变量类型)
# 类型用于决定存储策略：RADAR/LABEL -> int16, NWP -> float16
CHANNELS_TO_PACK = [
    ("LABEL", "RA",    "LABEL"),
    ("RADAR", "CR",    "RADAR"),
    ("RADAR", "CAP30", "RADAR"),
    ("RADAR", "CAP50", "RADAR"),
    ("RADAR", "ET",    "RADAR"),
    ("RADAR", "VIL",   "RADAR"),
    ("NWP",   "WS925", "NWP"),
    ("NWP",   "WS500", "NWP"),
    ("NWP",   "Q850",  "NWP"),
    ("NWP",   "Q700",  "NWP"),
    ("NWP",   "PWAT",  "NWP"),
    ("NWP",   "CAPE",  "NWP"),
]

# ==============================================================================
# 2. 核心工具函数
# ==============================================================================

def resize_to_target(img, target_size, interpolation=cv2.INTER_NEAREST):
    """调整图像尺寸"""
    if img.shape[0] == target_size and img.shape[1] == target_size:
        return img
    return cv2.resize(img, (target_size, target_size), interpolation=interpolation)

def parse_sample_id(sample_id):
    """解析 Sample ID 获取元数据"""
    parts = sample_id.split('_')
    return {
        'task_id': parts[0],      'region_id': parts[1],
        'case_time': parts[2],    'station_id': parts[3],
        'radar_type': parts[4],   'batch_id': parts[5],
        'case_id': '_'.join(parts[:4])
    }

def get_nwp_timestamp(ts_str):
    """NWP 时间对齐逻辑 (整点归并)"""
    try:
        dt = datetime.strptime(ts_str, DATE_FMT)
        # 如果分钟 >= 30，归并到下一小时
        if dt.minute >= 30:
            dt += timedelta(hours=1)
        return dt.replace(minute=0).strftime(DATE_FMT)
    except Exception:
        return ts_str

def construct_paths(root_path, meta, ts_raw):
    """
    构建该样本所有通道的文件绝对路径
    返回: List[Tuple(abs_path, var_type)]
    """
    base_dir = os.path.join(
        root_path, 
        meta['task_id'], 
        DEFAULT_DATASET, 
        meta['region_id'], 
        meta['case_id']
    )

    file_info_list = []
    
    for parent_dir, var_name, var_type in CHANNELS_TO_PACK:
        sub_dir = var_name 
        filename = ""

        if var_type == "RADAR":
            filename = f"{meta['task_id']}_RADA_{meta['station_id']}_{ts_raw}_{meta['radar_type']}_{var_name}.npy"
        elif var_type == "LABEL":
            filename = f"{meta['task_id']}_Label_{var_name}_{meta['station_id']}_{ts_raw}.npy"
        elif var_type == "NWP":
            nwp_ts = get_nwp_timestamp(ts_raw)
            filename = f"{meta['task_id']}_{DEFAULT_NWP_PREFIX}_{meta['station_id']}_{nwp_ts}_{var_name}.npy"
        
        abs_path = os.path.join(base_dir, parent_dir, sub_dir, filename)
        file_info_list.append((abs_path, var_type))
    
    return file_info_list

# ==============================================================================
# 3. 写入逻辑 (Writer)
# ==============================================================================

def write_region_lmdb(region, file_info_list, output_dir, root_path):
    """
    将单个 Region 的所有文件写入对应的 LMDB
    """
    MLOGI(f"Processing Region: {region} | Total Files: {len(file_info_list)}")
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    db_path = os.path.join(output_dir, f"{region}.lmdb")
    
    # 打开 LMDB 环境
    env = lmdb.open(db_path, map_size=MAP_SIZE)
    txn = env.begin(write=True)
    
    count = 0        # 成功写入计数
    skipped_count = 0 # 已存在跳过计数
    missing_count = 0 # 缺失补零计数
    
    # 进度条
    pbar = tqdm(file_info_list, desc=f"Writing {region}", unit="file")
    
    try:
        for abs_path, var_type in pbar:
            try:
                # ---------------- 1. 生成 Key ----------------
                # Key 规则: 相对路径 (反斜杠转正斜杠)
                rel_path = os.path.relpath(abs_path, root_path)
                key = rel_path.replace("\\", "/").encode('ascii')
                
                # ---------------- 2. 检查是否已存在 ----------------
                # 如果 LMDB 中已有该 Key，直接跳过读取步骤
                if txn.get(key) is not None:
                    skipped_count += 1
                    continue

                # ---------------- 3. 确定目标数据类型 ----------------
                # RADAR/LABEL -> int16 (节省空间)
                # NWP         -> float16 (保留精度)
                is_int_storage = (var_type in ["RADAR", "LABEL"])
                target_dtype = np.int16 if is_int_storage else np.float16
                
                # ---------------- 4. 读取与处理数据 ----------------
                if not os.path.exists(abs_path):
                    # 文件缺失 -> 生成全0矩阵
                    data = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=target_dtype)
                    missing_count += 1
                else:
                    # 加载原始数据
                    data = np.load(abs_path)
                    
                    # 尺寸调整
                    if data.shape[0] != TARGET_SIZE or data.shape[1] != TARGET_SIZE:
                        data = resize_to_target(data, TARGET_SIZE)
                    
                    # 类型转换
                    if is_int_storage:
                        # 假设: 源数据已经是放大10倍后的数值 (如 35.5dBZ 存为 355)
                        # 直接转换类型
                        data = data.astype(np.int16)
                    else:
                        # NWP 数据保留浮点特性，转半精度
                        data = data.astype(np.float16)

                # ---------------- 5. 序列化与压缩 ----------------
                # Pickle 序列化
                serialized = pickle.dumps(data)
                
                # Zlib 压缩 (Level 1 速度优先)
                # 对于稀疏的 Radar 数据，压缩率极高
                compressed = zlib.compress(serialized, level=ZLIB_LEVEL)
                
                # ---------------- 6. 写入 ----------------
                txn.put(key, compressed)
                
                count += 1
                
                # ---------------- 7. 定期 Commit ----------------
                if count % COMMIT_INTERVAL == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                    
            except Exception as e:
                MLOGE(f"Error processing file {abs_path}: {e}")

        # 循环结束，提交剩余事务
        txn.commit()
        MLOGI(f"Region {region} Finished. Written: {count}, Skipped: {skipped_count}, Missing: {missing_count}")

    except Exception as e:
        MLOGE(f"Critical Error in {region}: {e}")
        try: txn.abort()
        except: pass
    finally:
        env.close()

# ==============================================================================
# 4. 主程序流程
# ==============================================================================

def scan_tasks(jsonl_path, root_path):
    """扫描 JSONL 生成任务列表"""
    tasks = {} # format: {region_id: [(abs_path, var_type), ...]}
    
    MLOGI(f"Scanning JSONL: {jsonl_path}")
    if not os.path.exists(jsonl_path):
        MLOGE(f"File not found: {jsonl_path}")
        return {}

    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
        
    MLOGI(f"Found {len(lines)} samples. Parsing metadata...")
    
    for line in tqdm(lines, desc="Parsing Metadata"):
        if not line.strip(): continue
        try:
            rec = json.loads(line)
            meta = parse_sample_id(rec['sample_id'])
            reg = meta['region_id']
            
            if reg not in tasks: 
                tasks[reg] = []
            
            # 遍历该样本的所有时间步
            for ts in rec['timestamps']:
                # 生成该时间步所有通道的文件路径
                files = construct_paths(root_path, meta, ts)
                tasks[reg].extend(files)
                
        except Exception as e:
            MLOGE(f"Parse error: {e}")
            continue
            
    return tasks

def main():
    parser = argparse.ArgumentParser(description="Generate LMDB Datasets for SevereWeather AI")
    parser.add_argument("--jsonl", type=str, default=JSONL_PATH_DEFAULT, help="Path to samples.jsonl")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR_DEFAULT, help="Output directory for LMDBs")
    parser.add_argument("--root", type=str, default=ROOT_PATH_DEFAULT, help="Root path of source .npy files")
    
    args = parser.parse_args()
    
    jsonl_path = args.jsonl
    output_dir = args.output
    root_path = args.root

    MLOGI("================ CONFIG ================")
    MLOGI(f"Source Root : {root_path}")
    MLOGI(f"JSONL Index : {jsonl_path}")
    MLOGI(f"Output Dir  : {output_dir}")
    MLOGI(f"Compression : ZLIB Level {ZLIB_LEVEL}")
    MLOGI("========================================")

    # 1. 扫描任务
    tasks = scan_tasks(jsonl_path, root_path)
    
    if not tasks:
        MLOGE("No tasks found. Exiting.")
        return

    MLOGI(f"Found {len(tasks)} regions to process.")

    # 2. 按区域执行写入
    for reg, file_infos in tasks.items():
        # 简单去重 (同一文件可能被不同样本引用，尽管在时序数据中较少见，但去重更安全)
        # file_infos 是 list of (path, type)
        unique_map = {} 
        for p, t in file_infos:
            unique_map[p] = t # 路径为 Key 去重
            
        unique_infos = [(p, t) for p, t in unique_map.items()]
        
        write_region_lmdb(reg, unique_infos, output_dir, root_path)

    MLOGI(f"All Done. LMDBs saved to {output_dir}")

if __name__ == "__main__":
    main()