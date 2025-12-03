# run.create_lmdb.py
import lmdb
import pickle
import os
import numpy as np
import zlib
import cv2
import json
import argparse
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from datetime import datetime, timedelta
from typing import List

# ==============================================================================
# 0. 环境设置与依赖
# ==============================================================================

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 假设 metai 包在您的环境中可用
from metai.utils import MetConfig, MetLabel, MetRadar, MetNwp, MetGis, MetVarType
from metai.utils import MLOGE, MLOGI, get_config

cfg = get_config()
ROOT_PATH_DEFAULT = getattr(cfg, 'root_path', "/data/zjobs/SevereWeather_AI_2025/")

# ==============================================================================
# 1. 全局配置
# ==============================================================================

JSONL_PATH_DEFAULT = "./data/samples.jsonl"
OUTPUT_DIR_DEFAULT = "/data/zjobs/LMDB"

TARGET_SIZE = 256
MAP_SIZE = 1024 * 1024 * 1024 * 2048  # 2TB
COMMIT_INTERVAL = 2000
ZLIB_LEVEL = 1

DEFAULT_DATASET = "TrainSet"
DEFAULT_NWP_PREFIX = getattr(cfg, 'nwp_prefix', 'RRA')
DATE_FMT = getattr(cfg, 'file_date_format', '%m%d-%H%M')

_DEFAULT_CHANNELS: List[MetVarType] = [
    MetLabel.RA, 
    MetRadar.CR, MetRadar.CAP30, MetRadar.CAP50, MetRadar.ET, MetRadar.VIL,
    MetNwp.WS925, MetNwp.WS500, MetNwp.Q850, MetNwp.Q700, MetNwp.PWAT, MetNwp.CAPE,
]

# ==============================================================================
# 2. 核心工具函数
# ==============================================================================

def resize_to_target(img, target_size, interpolation=cv2.INTER_NEAREST):
    if img.shape[0] == target_size and img.shape[1] == target_size:
        return img
    return cv2.resize(img, (target_size, target_size), interpolation=interpolation)

def parse_sample_id(sample_id):
    parts = sample_id.split('_')
    if len(parts) < 6:
        raise ValueError(f"Invalid Sample ID format: {sample_id}")
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

def construct_paths(root_path, meta, ts_raw):
    base_dir = os.path.join(
        root_path, meta['task_id'], DEFAULT_DATASET, meta['region_id'], meta['case_id']
    )
    file_info_list = []
    
    for channel in _DEFAULT_CHANNELS:
        var_name = channel.name 
        parent_dir = channel.parent 
        filename = ""
        
        if isinstance(channel, MetRadar):
            filename = f"{meta['task_id']}_RADA_{meta['station_id']}_{ts_raw}_{meta['radar_type']}_{var_name}.npy"
        elif isinstance(channel, MetLabel):
            filename = f"{meta['task_id']}_Label_{var_name}_{meta['station_id']}_{ts_raw}.npy"
        elif isinstance(channel, MetNwp):
            nwp_ts = get_nwp_timestamp(ts_raw)
            filename = f"{meta['task_id']}_{DEFAULT_NWP_PREFIX}_{meta['station_id']}_{nwp_ts}_{var_name}.npy"
        else:
            continue

        abs_path = os.path.join(base_dir, parent_dir, var_name, filename)
        file_info_list.append((abs_path, channel))
    
    return file_info_list

# ==============================================================================
# 3. 写入逻辑 (Writer - Modified)
# ==============================================================================

def write_region_lmdb(region, file_info_list, output_dir, root_path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    db_path = os.path.join(output_dir, f"{region}.lmdb")
    env = lmdb.open(db_path, map_size=MAP_SIZE, writemap=True)
    txn = env.begin(write=True)
    
    count, skipped_count, missing_count = 0, 0, 0
    pbar = tqdm(file_info_list, desc=f"Reg: {region}", unit="file", position=1, leave=False)
    
    try:
        for abs_path, channel in pbar:
            try:
                rel_path = os.path.relpath(abs_path, root_path)
                key = rel_path.replace("\\", "/").encode('ascii')
                
                # 检查 Key 是否已存在 (断点续传)
                if txn.get(key) is not None:
                    skipped_count += 1
                    continue

                # ==========================================================
                # 逻辑分支：文件是否存在
                # ==========================================================
                if os.path.exists(abs_path):
                    # --- 情况 A: 文件存在 -> 正常读取、处理、写入 ---
                    
                    var_parent = channel.parent
                    missing_val = getattr(channel, 'missing_value', None)
                    min_val = getattr(channel, 'min', None)
                    max_val = getattr(channel, 'max', None)

                    is_int_storage = (var_parent in ["RADAR", "LABEL"])
                    target_dtype = np.int16 if is_int_storage else np.float16
                    
                    # 1. 读取
                    raw_data = np.load(abs_path)
                    mask = None

                    # 2. 生成 Mask (仅 RA)
                    if channel == MetLabel.RA:
                        if missing_val is not None:
                            mask = (raw_data != missing_val) & np.isfinite(raw_data)
                        else:
                            mask = np.ones_like(raw_data, dtype=bool)

                    # 3. 数据清洗
                    data_processed = raw_data.copy()
                    if missing_val is not None:
                        data_processed[data_processed == missing_val] = 0
                    data_processed[~np.isfinite(data_processed)] = 0
                    if min_val is not None:
                        data_processed[data_processed < min_val] = min_val
                    if max_val is not None:
                        data_processed[data_processed > max_val] = max_val

                    # 4. Resize
                    data = None
                    if data_processed.shape[0] != TARGET_SIZE or data_processed.shape[1] != TARGET_SIZE:
                        data = resize_to_target(data_processed, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
                        if mask is not None:
                            mask_uint8 = mask.astype(np.uint8)
                            mask_resized = resize_to_target(mask_uint8, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                            mask = mask_resized.astype(bool)
                    else:
                        data = data_processed
                    
                    # 5. 写入数据
                    data = data.astype(target_dtype)
                    txn.put(key, zlib.compress(pickle.dumps(data), level=ZLIB_LEVEL))
                    
                    # 6. 写入 Mask (仅 RA)
                    if channel == MetLabel.RA and mask is not None:
                        key_str = key.decode('ascii')
                        mask_key_str = key_str.replace("RA", "MASK") # 保持原样
                        mask_key = mask_key_str.encode('ascii')
                        txn.put(mask_key, zlib.compress(pickle.dumps(mask), level=ZLIB_LEVEL))

                    count += 1

                else:
                    # --- 情况 B: 文件不存在 ---
                    missing_count += 1
                    
                    # 1. 如果是普通文件: 此时什么都不做，不写入 Data Key，也不补0
                    
                    # 2. 如果是 RA Label: 需要补一个全 0 的 Mask
                    if channel == MetLabel.RA:
                        # 直接生成目标尺寸的全 False Mask
                        mask = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=bool)
                        
                        key_str = key.decode('ascii')
                        mask_key_str = key_str.replace("RA", "MASK")
                        mask_key = mask_key_str.encode('ascii')
                        
                        txn.put(mask_key, zlib.compress(pickle.dumps(mask), level=ZLIB_LEVEL))
                        # 注意：此处依然不写入 RA 的 data key

                # 事务提交检查
                if (count + missing_count) % COMMIT_INTERVAL == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                    
            except Exception as e:
                MLOGE(f"Error processing file {abs_path}: {e}")

        txn.commit()
        return f"{region}: OK (W:{count}, S:{skipped_count}, M:{missing_count})"

    except Exception as e:
        MLOGE(f"Critical Error in {region}: {e}")
        try: txn.abort()
        except: pass
        return f"{region}: FAILED ({str(e)})"
    finally:
        env.close()

# ==============================================================================
# 4. 主流程 (Multiprocessing)
# ==============================================================================

def process_task_wrapper(args):
    return write_region_lmdb(*args)

def scan_tasks(jsonl_path, root_path):
    tasks = {} 
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
            if reg not in tasks: tasks[reg] = []
            for ts in rec['timestamps']:
                tasks[reg].extend(construct_paths(root_path, meta, ts))
        except Exception as e:
            print(e)
            continue
    return tasks

def main():
    parser = argparse.ArgumentParser(description="Generate LMDB Datasets (Sparse Data, Full RA Mask)")
    parser.add_argument("--jsonl", type=str, default=JSONL_PATH_DEFAULT)
    parser.add_argument("--output", type=str, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--root", type=str, default=ROOT_PATH_DEFAULT)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    
    MLOGI("================ CONFIG ================")
    MLOGI(f"Source Root : {args.root}")
    MLOGI(f"JSONL Index : {args.jsonl}")
    MLOGI(f"Output Dir  : {args.output}")
    MLOGI("========================================")

    tasks = scan_tasks(args.jsonl, args.root)
    if not tasks:
        MLOGE("No tasks found.")
        return

    job_args = []
    for reg, file_infos in tasks.items():
        unique_map = {p: t for p, t in file_infos}
        unique_infos = list(unique_map.items())
        job_args.append((reg, unique_infos, args.output, args.root))

    if args.workers > 0:
        max_workers = args.workers
    else:
        cpu_count = multiprocessing.cpu_count()
        max_workers = max(1, cpu_count - 1) if cpu_count > 4 else cpu_count

    MLOGI(f"Starting parallel processing with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(process_task_wrapper, job_args), 
            total=len(job_args), 
            desc="Total Progress", 
            unit="region",
            position=0
        ))

    MLOGI(f"All Done. LMDBs saved to {args.output}")

if __name__ == "__main__":
    main()