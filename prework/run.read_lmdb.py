import lmdb
import pickle
import os
import json
import numpy as np
import cv2
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader
import torch

# ==============================================================================
# 1. 必须复用的配置与辅助函数 (必须与写入逻辑保持一致)
# ==============================================================================

# 配置 (请修改为你实际的路径)
LMDB_DIR = "/data/zjobs/LMDB"           # LMDB 存放目录
JSONL_PATH = "./data/samples.jsonl"     # 索引文件路径
ROOT_PATH_DURING_WRITE = "/data/zjobs/SevereWeather_AI_2025/" # 写入时使用的 root_path (用于计算相对路径)

DEFAULT_DATASET = "TrainSet"
DEFAULT_NWP_PREFIX = "ERA5"
DATE_FMT = "%Y%m%d-%H%M"

# 通道配置 (顺序必须一致，方便后续 stack)
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

def parse_sample_id(sample_id):
    """解析 sample_id，逻辑与写入时一致"""
    parts = sample_id.split('_')
    return {
        'task_id': parts[0],      'region_id': parts[1],
        'case_time': parts[2],    'station_id': parts[3],
        'radar_type': parts[4],   'batch_id': parts[5],
        'case_id': '_'.join(parts[:4])
    }

def get_nwp_timestamp(ts_str):
    """NWP 时间对齐逻辑，逻辑与写入时一致"""
    try:
        dt = datetime.strptime(ts_str, DATE_FMT)
        if dt.minute >= 30:
            dt += timedelta(hours=1)
        return dt.replace(minute=0).strftime(DATE_FMT)
    except Exception:
        return ts_str

def generate_keys(meta, ts_raw, root_path):
    """
    核心函数：根据元数据生成 LMDB 的 Key
    逻辑：构造绝对路径 -> 转相对路径 -> 替换分隔符 -> encode
    """
    base_dir = os.path.join(
        root_path, 
        meta['task_id'],      
        DEFAULT_DATASET,      
        meta['region_id'],    
        meta['case_id']       
    )

    keys = []
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
        
        # 构造绝对路径
        abs_path = os.path.join(base_dir, parent_dir, sub_dir, filename)
        
        # 转换为 Key (相对路径 + 格式化)
        rel_path = os.path.relpath(abs_path, root_path)
        key = rel_path.replace("\\", "/").encode('ascii')
        keys.append(key)
    
    return keys

# ==============================================================================
# 2. PyTorch Dataset 实现 (推荐用于训练)
# ==============================================================================

class WeatherLMDBDataset(Dataset):
    def __init__(self, jsonl_path, lmdb_root, original_root_path):
        self.samples = []
        self.lmdb_root = lmdb_root
        self.original_root_path = original_root_path
        
        # 缓存 LMDB 环境句柄，避免重复打开
        self.envs = {} 
        
        print(f"Loading index from {jsonl_path}...")
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        print(f"Loaded {len(self.samples)} samples.")

    def _get_env(self, region_id):
        """懒加载获取对应 Region 的 LMDB 环境"""
        if region_id not in self.envs:
            lmdb_path = os.path.join(self.lmdb_root, f"{region_id}.lmdb")
            if not os.path.exists(lmdb_path):
                raise FileNotFoundError(f"LMDB not found: {lmdb_path}")
            # readonly=True, lock=False 是读取速度最快的配置
            self.envs[region_id] = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        return self.envs[region_id]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        record = self.samples[idx]
        meta = parse_sample_id(record['sample_id'])
        region = meta['region_id']
        
        # 这里假设我们只取第一个时间步作为输入，你可以根据需求修改循环读取所有 timestamps
        target_ts = record['timestamps'][0] 
        
        # 1. 生成所有通道的 Key
        keys = generate_keys(meta, target_ts, self.original_root_path)
        
        # 2. 获取 LMDB 环境句柄
        env = self._get_env(region)
        
        channels_data = []
        
        with env.begin(write=False) as txn:
            for key in keys:
                # 3. 读取数据
                byte_data = txn.get(key)
                
                if byte_data is None:
                    # 如果 Key 不存在（写入时没有补0且没写进去），这里需要处理
                    # 根据你的写入代码，写入时已经做了 zeros 填充，所以理论上这里应该都有值
                    print(f"Warning: Key not found {key}")
                    img = np.zeros((256, 256), dtype=np.float32)
                else:
                    # 4. 反序列化 (Unpickle)
                    img = pickle.loads(byte_data)
                
                channels_data.append(img)
        
        # 5. 堆叠数据 (Channels, Height, Width)
        # 结果形状: (12, 256, 256) -> 12 是 CHANNELS_TO_PACK 的长度
        data_tensor = torch.from_numpy(np.stack(channels_data, axis=0))
        
        return data_tensor, meta['sample_id']

# ==============================================================================
# 3. 简单调试函数 (不依赖 PyTorch)
# ==============================================================================

def debug_read_one_key(lmdb_path, key_str):
    """
    读取单个 Key 用于测试
    """
    print(f"Debug reading from {lmdb_path}...")
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    key_bytes = key_str.encode('ascii')
    
    with env.begin() as txn:
        val = txn.get(key_bytes)
        if val:
            data = pickle.loads(val)
            print(f"Key Found!")
            print(f"Shape: {data.shape}")
            print(f"Dtype: {data.dtype}")
            print(f"Mean:  {data.mean():.4f}")
            print(f"Max:   {data.max():.4f}")
            print(f"Min:   {data.min():.4f}")
        else:
            print("Key NOT found.")
    env.close()

# ==============================================================================
# 4. 主程序运行示例
# ==============================================================================

if __name__ == "__main__":
    # --- 方式 A: 使用 PyTorch DataLoader 批量读取 ---
    print("\n=== Testing Dataset Reader ===")
    
    # 确保这里的 root_path 和你写入代码中的 root_path 完全一致！
    # 因为 Key 是基于相对路径生成的。
    # dataset = WeatherLMDBDataset(
    #     jsonl_path=JSONL_PATH, 
    #     lmdb_root=LMDB_DIR,
    #     original_root_path=ROOT_PATH_DURING_WRITE 
    # )
    
    # if len(dataset) > 0:
    #     loader = DataLoader(dataset, batch_size=4, num_workers=0) # num_workers=0 for simple debug
        
    #     for batch_idx, (data, sample_ids) in enumerate(loader):
    #         print(f"Batch {batch_idx}:")
    #         print(f"  Tensor Shape: {data.shape}") # 应为 [4, 12, 256, 256]
    #         print(f"  Sample IDs: {sample_ids}")
    #         break
            
    # --- 方式 B: 手动调试读取某个特定的 Key ---
    # 如果你知道某个 Region 下的具体相对路径，可以这样测：
    print("\n=== Testing Single Key ===")
    test_region = "AH" # 假设的区域
    test_lmdb = os.path.join(LMDB_DIR, f"{test_region}.lmdb")
    test_key = "CP/TrainSet/AH/CP_AH_201605061442_Z9552/RADAR/CR/CP_RADA_Z9552_20160506-1249_SA_CR.npy" 
    debug_read_one_key(test_lmdb, test_key)