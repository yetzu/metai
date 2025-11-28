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
from typing import List, Union

# ==============================================================================
# 0. 环境设置与依赖
# ==============================================================================

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.utils import MetConfig, MetLabel, MetRadar, MetNwp, MetGis, MetVarType
from metai.utils import MLOGE, MLOGI, get_config

cfg = get_config()
ROOT_PATH_DEFAULT = getattr(cfg, 'root_path', "/data/zjobs/SevereWeather_AI_2025/")

# ==============================================================================
# 1. 全局配置
# ==============================================================================

# 路径配置
JSONL_PATH_DEFAULT = "./data/samples.jsonl"
OUTPUT_DIR_DEFAULT = "/data/zjobs/LMDB"

# LMDB与图像参数
TARGET_SIZE = 256
MAP_SIZE = 1024 * 1024 * 1024 * 2048  # 2TB (根据需要调整)
COMMIT_INTERVAL = 2000                # 事务提交间隔
ZLIB_LEVEL = 1                        # 压缩等级 (1=最快)

# 命名与时间格式
DEFAULT_DATASET = "TrainSet"
DEFAULT_NWP_PREFIX = getattr(cfg, 'nwp_prefix', 'RRA')
DATE_FMT = getattr(cfg, 'file_date_format', '%m%d-%H%M')

# 通道配置表 (需要处理的通道)
_DEFAULT_CHANNELS: List[MetVarType] = [
    # Label
    MetLabel.RA, 
    # Radar
    MetRadar.CR, MetRadar.CAP30, MetRadar.CAP50, MetRadar.ET, MetRadar.VIL,
    # NWP
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
        # 兼容部分可能的短ID，或者抛出异常
        raise ValueError(f"Invalid Sample ID format: {sample_id}")
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
        if dt.minute >= 30:
            dt += timedelta(hours=1)
        return dt.replace(minute=0).strftime(DATE_FMT)
    except Exception:
        return ts_str

def construct_paths(root_path, meta, ts_raw):
    """构建该样本所有通道的文件绝对路径"""
    base_dir = os.path.join(
        root_path, meta['task_id'], DEFAULT_DATASET, meta['region_id'], meta['case_id']
    )
    file_info_list = []
    
    for channel in _DEFAULT_CHANNELS:
        var_name = channel.name 
        parent_dir = channel.parent 
        filename = ""
        
        # 根据类型生成文件名
        if isinstance(channel, MetRadar):
            filename = f"{meta['task_id']}_RADA_{meta['station_id']}_{ts_raw}_{meta['radar_type']}_{var_name}.npy"
        elif isinstance(channel, MetLabel):
            filename = f"{meta['task_id']}_Label_{var_name}_{meta['station_id']}_{ts_raw}.npy"
        elif isinstance(channel, MetNwp):
            nwp_ts = get_nwp_timestamp(ts_raw)
            filename = f"{meta['task_id']}_{DEFAULT_NWP_PREFIX}_{meta['station_id']}_{nwp_ts}_{var_name}.npy"
        else:
            MLOGE(f"Skipping unknown channel type: {channel}")
            continue

        abs_path = os.path.join(base_dir, parent_dir, var_name, filename)

        # 返回 (绝对路径, channel对象)
        file_info_list.append((abs_path, channel))
    
    return file_info_list

# ==============================================================================
# 3. 写入逻辑 (Writer)
# ==============================================================================

def write_region_lmdb(region, file_info_list, output_dir, root_path):
    MLOGI(f"Processing Region: {region} | Total Files: {len(file_info_list)}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    db_path = os.path.join(output_dir, f"{region}.lmdb")
    env = lmdb.open(db_path, map_size=MAP_SIZE)
    txn = env.begin(write=True)
    
    count, skipped_count, missing_count = 0, 0, 0
    pbar = tqdm(file_info_list, desc=f"Writing {region}", unit="file")
    
    try:
        for abs_path, channel in pbar:
            try:
                # 生成 Key (相对路径转 Key)
                rel_path = os.path.relpath(abs_path, root_path)
                key = rel_path.replace("\\", "/").encode('ascii')
                
                # 如果 key 存在，说明该文件已处理
                if txn.get(key) is not None:
                    skipped_count += 1
                    continue

                # 1. 获取元数据信息 (min, max, missing_value)
                var_parent = channel.parent
                missing_val = getattr(channel, 'missing_value', None)
                min_val = getattr(channel, 'min', None)
                max_val = getattr(channel, 'max', None)

                # 确定存储类型: RADAR/LABEL -> int16, NWP -> float16
                is_int_storage = (var_parent in ["RADAR", "LABEL"])
                target_dtype = np.int16 if is_int_storage else np.float16
                
                # 初始化变量
                mask = None
                data = None

                # 2. 读取数据
                if not os.path.exists(abs_path):
                    # 文件缺失：使用全0数据
                    # 对于RA，Mask为全False
                    data = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=target_dtype)
                    if channel == MetLabel.RA:
                        mask = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=bool)
                    missing_count += 1
                else:
                    raw_data = np.load(abs_path)
                    
                    # ---------------------------------------------------------
                    # [Step 1] 优先生成 Mask (仅针对 RA)
                    # ---------------------------------------------------------
                    if channel == MetLabel.RA:
                        if missing_val is not None:
                            # 原始数据不等于缺失值，且是有限数 -> 有效
                            mask = (raw_data != missing_val) & np.isfinite(raw_data)
                        else:
                            mask = np.ones_like(raw_data, dtype=bool)
                    
                    # ---------------------------------------------------------
                    # [Step 2] 数据清洗: Missing -> 0, Clip [Min, Max]
                    # ---------------------------------------------------------
                    # 使用副本以免修改原数组
                    data_processed = raw_data.copy()

                    # 2.1 将 missing_value 替换为 0
                    if missing_val is not None:
                        # 对于浮点数，使用 isclose 可能更安全，但这里通常是离散标记
                        data_processed[data_processed == missing_val] = 0
                    
                    # 2.2 清理 inf/nan
                    data_processed[~np.isfinite(data_processed)] = 0

                    # 2.3 根据 min 截断
                    if min_val is not None:
                        data_processed[data_processed < min_val] = min_val
                    
                    # 2.4 根据 max 截断
                    if max_val is not None:
                        data_processed[data_processed > max_val] = max_val

                    # ---------------------------------------------------------
                    # [Step 3] Resize (统一到 256x256)
                    # ---------------------------------------------------------
                    if data_processed.shape[0] != TARGET_SIZE or data_processed.shape[1] != TARGET_SIZE:
                        # 数据: 线性插值
                        data = resize_to_target(data_processed, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
                        
                        # Mask (仅 RA): 最近邻插值
                        if mask is not None:
                            mask_uint8 = mask.astype(np.uint8)
                            mask_resized = resize_to_target(mask_uint8, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                            mask = mask_resized.astype(bool)
                    else:
                        data = data_processed
                    
                    data = data.astype(target_dtype)

                # 3. 存储 Data (压缩)
                txn.put(key, zlib.compress(pickle.dumps(data), level=ZLIB_LEVEL))
                
                # 4. 存储 Mask (仅 RA, 且 key 中 RA 替换为 MASK)
                if channel == MetLabel.RA and mask is not None:
                    key_str = key.decode('ascii')
                    # 替换路径中的 RA 为 MASK, e.g., .../LABEL/RA/... -> .../LABEL/MASK/...
                    # 以及文件名中的 RA
                    mask_key_str = key_str.replace("RA", "MASK")
                    mask_key = mask_key_str.encode('ascii')
                    
                    txn.put(mask_key, zlib.compress(pickle.dumps(mask), level=ZLIB_LEVEL))

                count += 1
                
                # 定期提交事务
                if count % COMMIT_INTERVAL == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                    
            except Exception as e:
                MLOGE(f"Error processing file {abs_path}: {e}")

        # 最后的提交
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
            MLOGE(f"Parse error: {e}")
            continue
    return tasks

def main():
    parser = argparse.ArgumentParser(description="Generate LMDB Datasets (Cleaned Data + RA Mask Only)")
    parser.add_argument("--jsonl", type=str, default=JSONL_PATH_DEFAULT)
    parser.add_argument("--output", type=str, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--root", type=str, default=ROOT_PATH_DEFAULT)
    args = parser.parse_args()
    
    MLOGI("================ CONFIG ================")
    MLOGI(f"Source Root : {args.root}")
    MLOGI(f"JSONL Index : {args.jsonl}")
    MLOGI(f"Output Dir  : {args.output}")
    MLOGI(f"Channels    : {len(_DEFAULT_CHANNELS)}")
    MLOGI("========================================")

    tasks = scan_tasks(args.jsonl, args.root)
    if not tasks:
        MLOGE("No tasks found. Exiting.")
        return

    MLOGI(f"Found {len(tasks)} regions to process.")

    for reg, file_infos in tasks.items():
        # 去重 (以路径为Key)
        unique_map = {p: t for p, t in file_infos}
        unique_infos = list(unique_map.items())
        
        write_region_lmdb(reg, unique_infos, args.output, args.root)

    MLOGI(f"All Done. LMDBs saved to {args.output}")

if __name__ == "__main__":
    main()