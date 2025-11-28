import json
import lmdb
import os
import numpy as np
import pickle
import cv2
from tqdm import tqdm
from datetime import datetime, timedelta
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.utils import MLOGE, MLOGI, get_config

_cfg = get_config()

# ==============================================================================
# 1. 全局配置
# ==============================================================================

JSONL_PATH = "./data/samples.jsonl"
SOURCE_ROOT = getattr(_cfg, 'root_path', "/data/zjobs/SevereWeather_AI_2025/")
OUTPUT_DIR = "/data/zjobs/LMDB"

TARGET_SIZE = 256
MAP_SIZE = 1024 * 1024 * 1024 * 2048  # 2TB
COMMIT_INTERVAL = 2000

DEFAULT_DATASET = "TrainSet"
DEFAULT_NWP_PREFIX = "ERA5"
DATE_FMT = "%Y%m%d-%H%M"

# (ParentDir, VarName, FileType)
CHANNELS_TO_PACK = [
    ("LABEL", "RA", "LABEL"),
    ("RADAR", "CR", "RADAR"),
    ("RADAR", "CAP30", "RADAR"),
    ("RADAR", "CAP50", "RADAR"),
    ("RADAR", "ET", "RADAR"),
    ("RADAR", "VIL", "RADAR"),
    ("NWP", "WS925", "NWP"),
    ("NWP", "WS500", "NWP"),
    ("NWP", "Q850", "NWP"),
    ("NWP", "Q700", "NWP"),
    ("NWP", "PWAT", "NWP"),
    ("NWP", "CAPE", "NWP"),
]

# ==============================================================================
# 2. 核心逻辑函数
# ==============================================================================

def resize_to_target(img, target_size, interpolation=cv2.INTER_NEAREST):
    if img.shape[0] == target_size and img.shape[1] == target_size:
        return img
    return cv2.resize(img, (target_size, target_size), interpolation=interpolation)

def parse_sample_id(sample_id):
    parts = sample_id.split('_')
    return {
        'task_id': parts[0],      'region_id': parts[1],
        'case_time': parts[2],    'station_id': parts[3],
        'radar_type': parts[4],   'batch_id': parts[5],
        'case_id': '_'.join(parts[:4])
    }

def get_nwp_timestamp(ts_str):
    try:
        dt = datetime.strptime(ts_str, DATE_FMT)
        if dt.minute >= 30:
            dt += timedelta(hours=1)
        return dt.replace(minute=0).strftime(DATE_FMT)
    except Exception:
        return ts_str

def construct_paths(meta, ts_raw):
    """构建文件的绝对路径"""
    root_path = getattr(_cfg, 'root_path', SOURCE_ROOT)
    
    base_dir = os.path.join(
        root_path, 
        meta['task_id'],      
        DEFAULT_DATASET,      
        meta['region_id'],    
        meta['case_id']       
    )

    paths = []
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
        else:
            continue

        abs_path = os.path.join(base_dir, parent_dir, sub_dir, filename)
        paths.append(abs_path)
    
    return paths

# ==============================================================================
# 3. 流程控制函数
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, default=JSONL_PATH)
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    return parser.parse_args()

def scan_tasks(jsonl_path):
    MLOGI(f"Scanning JSONL: {jsonl_path}")
    if not os.path.exists(jsonl_path):
        MLOGE(f"File not found: {jsonl_path}")
        return {}

    tasks_by_region = {}
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
        
    MLOGI(f"Found {len(lines)} samples. Parsing metadata...")
    
    for line in tqdm(lines, desc="Parsing Metadata"):
        if not line.strip(): continue
        try:
            record = json.loads(line)
            meta = parse_sample_id(record['sample_id'])
            region = meta['region_id']
            
            if region not in tasks_by_region:
                tasks_by_region[region] = set()
            
            for ts in record['timestamps']:
                file_paths = construct_paths(meta, ts)
                for p in file_paths:
                    tasks_by_region[region].add(p)
        except Exception as e:
            MLOGE(f"Parse error: {e}")
            continue
            
    return tasks_by_region

def write_region_lmdb(region, unique_files, output_dir, root_path):
    MLOGI(f"Region: {region} | Total Files: {len(unique_files)}")
    
    db_path = os.path.join(output_dir, f"{region}.lmdb")
    env = lmdb.open(db_path, map_size=MAP_SIZE)
    
    txn = env.begin(write=True)
    
    count = 0        # 实际写入次数
    skipped_exist = 0 # 数据库已存在跳过
    filled_zeros = 0  # 文件缺失补0次数
    
    try:
        for abs_path in tqdm(unique_files, desc=f"Writing {region}"):
            try:
                # 1. 生成 Key
                rel_path = os.path.relpath(abs_path, root_path)
                key = rel_path.replace("\\", "/").encode('ascii')
                
                # 2. [极速检查] 如果 LMDB 里已经有，直接跳过
                if txn.get(key) is not None:
                    skipped_exist += 1
                    continue

                # 3. 读取数据或补0
                data = None
                if not os.path.exists(abs_path):
                    # 文件不存在 -> 补全 0
                    MLOGE(f"❌ [MISSING -> ZERO] {abs_path}")
                    data = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
                    filled_zeros += 1
                else:
                    # 文件存在 -> 读取
                    data = np.load(abs_path)
                
                # 4. 预处理 (Resize & Typecast)
                # 即使是补0的矩阵，也检查一下形状确保万无一失
                if data.shape[0] != TARGET_SIZE or data.shape[1] != TARGET_SIZE:
                    data = resize_to_target(data, TARGET_SIZE)
                
                if data.dtype != np.float32:
                    data = data.astype(np.float32)
                    
                # 5. 写入
                txn.put(key, pickle.dumps(data))
                
                count += 1
                if count % COMMIT_INTERVAL == 0:
                    txn.commit()
                    txn = env.begin(write=True)

            except Exception as e:
                MLOGE(f"Error processing {abs_path}: {e}")
        
        txn.commit()
        MLOGI(f"Region {region} Finished.")
        MLOGI(f"Summary -> Written: {count}, Skipped(Exist): {skipped_exist}, Filled Zeros: {filled_zeros}")

    except Exception as e:
        MLOGE(f"Critical Error in {region}: {e}")
        try: txn.abort()
        except: pass
    finally:
        env.close()

def main():
    args = parse_args()
    
    jsonl_path = args.jsonl
    output_dir = args.output
    root_path = getattr(_cfg, 'root_path', SOURCE_ROOT)

    MLOGI("=== Config ===")
    MLOGI(f"Source: {root_path}")
    MLOGI(f"JSONL : {jsonl_path}")
    MLOGI(f"Output: {output_dir}")
    MLOGI("==============")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tasks = scan_tasks(jsonl_path)

    MLOGI("Starting LMDB Generation...")
    for region, files in tasks.items():
        write_region_lmdb(region, files, output_dir, root_path)

    MLOGI(f"All Done. Saved to {output_dir}")

if __name__ == "__main__":
    main()