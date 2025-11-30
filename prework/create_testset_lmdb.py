# prework/create_testset_lmdb.py
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

from metai.utils import MetConfig, MetLabel, MetRadar, MetNwp, MetGis, MetVarType
from metai.utils import MLOGE, MLOGI, get_config

cfg = get_config()
ROOT_PATH_DEFAULT = getattr(cfg, 'root_path', "/data/zjobs/SevereWeather_AI_2025/")

# ==============================================================================
# 1. 全局配置 (TestSetA)
# ==============================================================================

# [配置] 输入为测试集样本列表
JSONL_PATH_DEFAULT = "./data/samples.testset.jsonl"
OUTPUT_DIR_DEFAULT = "/data/zjobs/LMDB"

TARGET_SIZE = 256
MAP_SIZE = 1024 * 1024 * 1024 * 3072  # 3TB (调大容量以容纳所有数据)
COMMIT_INTERVAL = 2000
ZLIB_LEVEL = 1

# [配置] 数据集目录名称
DEFAULT_DATASET = "TestSet"
DEFAULT_NWP_PREFIX = getattr(cfg, 'nwp_prefix', 'RRA')
DATE_FMT = getattr(cfg, 'file_date_format', '%m%d-%H%M')

# 目标输出文件名 (不带后缀)
TARGET_LMDB_NAME = "TestSetA"

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
    # 兼容测试集ID格式
    if len(parts) < 5:
        # 尝试容错，有些测试集ID可能较短
        if len(parts) >= 4:
             return {
                'task_id': parts[0],      'region_id': parts[1],
                'case_time': parts[2],    'station_id': parts[3],
                'radar_type': 'SA',       'batch_id': '000',
                'case_id': '_'.join(parts[:4])
            }
        raise ValueError(f"Invalid Sample ID format: {sample_id}")
    
    return {
        'task_id': parts[0],      'region_id': parts[1],
        'case_time': parts[2],    'station_id': parts[3],
        'radar_type': parts[4],   'batch_id': parts[5] if len(parts) > 5 else "000",
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
    # 这里依然使用 meta['region_id'] 来定位源文件所在的文件夹
    # 即使我们最终写入同一个 LMDB，源文件还是按区域存放的
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
# 3. 写入逻辑 (Writer)
# ==============================================================================

def write_region_lmdb(name_key, file_info_list, output_dir, root_path):
    """
    Args:
        name_key: LMDB文件名 (例如 "TestSetA")
        ...
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    db_path = os.path.join(output_dir, f"{name_key}.lmdb")
    MLOGI(f"Writing to single LMDB: {db_path} | Total files: {len(file_info_list)}")
    
    env = lmdb.open(db_path, map_size=MAP_SIZE, writemap=True)
    txn = env.begin(write=True)
    
    count, skipped_count, missing_count = 0, 0, 0
    pbar = tqdm(file_info_list, desc=f"Writing {name_key}", unit="file", position=1, leave=False)
    
    try:
        for abs_path, channel in pbar:
            try:
                # Key 的构造逻辑不变，保持 Task/TestSet/Region/Case/...
                # 这样即使在同一个文件中，也能通过 Key 前缀区分（如果需要的话）
                rel_path = os.path.relpath(abs_path, root_path)
                key = rel_path.replace("\\", "/").encode('ascii')
                
                if txn.get(key) is not None:
                    skipped_count += 1
                    continue

                if os.path.exists(abs_path):
                    # --- A. 文件存在 ---
                    var_parent = channel.parent
                    missing_val = getattr(channel, 'missing_value', None)
                    min_val = getattr(channel, 'min', None)
                    max_val = getattr(channel, 'max', None)

                    is_int_storage = (var_parent in ["RADAR", "LABEL"])
                    target_dtype = np.int16 if is_int_storage else np.float16
                    
                    raw_data = np.load(abs_path)
                    mask = None

                    if channel == MetLabel.RA:
                        if missing_val is not None:
                            mask = (raw_data != missing_val) & np.isfinite(raw_data)
                        else:
                            mask = np.ones_like(raw_data, dtype=bool)

                    data_processed = raw_data.copy()
                    if missing_val is not None:
                        data_processed[data_processed == missing_val] = 0
                    data_processed[~np.isfinite(data_processed)] = 0
                    if min_val is not None:
                        data_processed[data_processed < min_val] = min_val
                    if max_val is not None:
                        data_processed[data_processed > max_val] = max_val

                    data = None
                    if data_processed.shape[0] != TARGET_SIZE or data_processed.shape[1] != TARGET_SIZE:
                        data = resize_to_target(data_processed, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
                        if mask is not None:
                            mask_uint8 = mask.astype(np.uint8)
                            mask_resized = resize_to_target(mask_uint8, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                            mask = mask_resized.astype(bool)
                    else:
                        data = data_processed
                    
                    data = data.astype(target_dtype)
                    txn.put(key, zlib.compress(pickle.dumps(data), level=ZLIB_LEVEL))
                    
                    if channel == MetLabel.RA and mask is not None:
                        key_str = key.decode('ascii')
                        mask_key_str = key_str.replace("RA", "MASK")
                        mask_key = mask_key_str.encode('ascii')
                        txn.put(mask_key, zlib.compress(pickle.dumps(mask), level=ZLIB_LEVEL))

                    count += 1

                else:
                    # --- B. 文件不存在 (测试集补0策略) ---
                    missing_count += 1
                    
                    # 仅针对 Label RA 补一个全 False 的 Mask，保证 Dataset 读取不崩
                    # 测试集通常没有 Label，但 DataModule 逻辑可能还是会尝试读取
                    if channel == MetLabel.RA:
                        mask = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=bool)
                        key_str = key.decode('ascii')
                        mask_key_str = key_str.replace("RA", "MASK")
                        mask_key = mask_key_str.encode('ascii')
                        txn.put(mask_key, zlib.compress(pickle.dumps(mask), level=ZLIB_LEVEL))

                if (count + missing_count) % COMMIT_INTERVAL == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                    
            except Exception as e:
                MLOGE(f"Error processing {abs_path}: {e}")

        txn.commit()
        return f"{name_key}: OK (W:{count}, S:{skipped_count}, M:{missing_count})"

    except Exception as e:
        MLOGE(f"Critical Error in {name_key}: {e}")
        try: txn.abort()
        except: pass
        return f"{name_key}: FAILED ({str(e)})"
    finally:
        env.close()

# ==============================================================================
# 4. 主流程
# ==============================================================================

def process_task_wrapper(args):
    return write_region_lmdb(*args)

def scan_tasks(jsonl_path, root_path):
    # [修改] 强制使用单个 key 来聚合所有数据
    single_task_key = TARGET_LMDB_NAME
    tasks = {single_task_key: []} 
    
    MLOGI(f"Scanning TestSet JSONL: {jsonl_path}")
    if not os.path.exists(jsonl_path):
        MLOGE(f"File not found: {jsonl_path}")
        return {}

    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
        
    MLOGI(f"Found {len(lines)} samples. Aggregating into '{single_task_key}'...")
    
    for line in tqdm(lines, desc="Parsing Metadata"):
        if not line.strip(): continue
        try:
            rec = json.loads(line)
            meta = parse_sample_id(rec['sample_id'])
            
            # 不再使用 meta['region_id'] 作为 key，而是全部放入 single_task_key
            for ts in rec['timestamps']:
                tasks[single_task_key].extend(construct_paths(root_path, meta, ts))
        except Exception as e:
            continue
    return tasks

def main():
    parser = argparse.ArgumentParser(description="Generate Single LMDB for TestSet")
    parser.add_argument("--jsonl", type=str, default=JSONL_PATH_DEFAULT)
    parser.add_argument("--output", type=str, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--root", type=str, default=ROOT_PATH_DEFAULT)
    # 对于单个 LMDB 文件的写入，必须强制单进程，否则 LMDB 会锁冲突
    # 我们依然保留 workers 参数接口但仅用于兼容性，实际只启动一个任务
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    
    MLOGI("================ CONFIG (TESTSET SINGLE FILE) ================")
    MLOGI(f"Source Root : {args.root}")
    MLOGI(f"Target Name : {TARGET_LMDB_NAME}.lmdb")
    MLOGI(f"JSONL Index : {args.jsonl}")
    MLOGI(f"Output Dir  : {args.output}")
    MLOGI("==============================================================")

    tasks = scan_tasks(args.jsonl, args.root)
    if not tasks:
        MLOGE("No tasks found.")
        return

    # 这里实际上只有 1 个任务: ('TestSetA', all_files, ...)
    job_args = []
    for name_key, file_infos in tasks.items():
        # 去重
        unique_map = {p: t for p, t in file_infos}
        unique_infos = list(unique_map.items())
        job_args.append((name_key, unique_infos, args.output, args.root))

    # 因为写入同一个文件，LMDB 不支持多进程并发写（除非极其复杂的配置），
    # 所以这里只能有一个 Worker 处理这个巨大的列表。
    MLOGI(f"Starting processing... (Single Writer for {TARGET_LMDB_NAME})")

    # 使用 ProcessPoolExecutor(max_workers=1) 或直接调用
    with ProcessPoolExecutor(max_workers=1) as executor:
        results = list(tqdm(
            executor.map(process_task_wrapper, job_args), 
            total=len(job_args), 
            desc="Total Progress", 
            unit="task",
            position=0
        ))

    MLOGI(f"All Done. LMDB saved to {os.path.join(args.output, TARGET_LMDB_NAME + '.lmdb')}")

if __name__ == "__main__":
    main()