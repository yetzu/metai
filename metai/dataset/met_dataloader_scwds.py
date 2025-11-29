# metai/dataset/met_dataloader_scwds.py
import os
import json
import cv2
import lmdb
import numpy as np
from typing import List, Dict, Any, Optional
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F 
from metai.utils import MLOGI, MLOGE
from metai.dataset import MetSample
from metai.utils.met_config import get_config

# 关键设置：防止多进程死锁与CPU资源争抢
# Critical setting: Prevent multi-process deadlocks and CPU resource contention
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class ScwdsDataset(Dataset):
    """
    SCWDS (Severe Convective Weather Dataset) 数据集加载器
    
    特性:
    1. 支持 LMDB 高效读取
    2. 针对多进程 Worker 优化的 LMDB 环境缓存
    3. 智能 Resize: 默认尺寸(256)下完全跳过处理，仅在需要变分辨率时激活
    """
    def __init__(self, data_path: str, is_train: bool = True, resize_shape: Optional[tuple] = None):
        """
        初始化数据集
        
        Args:
            data_path: 样本列表文件路径 (.jsonl)
            is_train: 是否为训练模式 (决定是否加载标签)
            resize_shape: 目标尺寸 (H, W)。如果为 (256, 256) 或 None，则跳过 Resize 操作。
        """
        self.data_path = data_path
        self.config = get_config()
        self.samples = self._load_samples_from_jsonl(data_path)
        self.is_train = is_train
        
        # [优化] 初始化阶段检测冗余 (Init-stage Redundancy Check)
        # MetSample 固定输出 256x256。如果目标尺寸也是 256x256，则强制置为 None 以关闭所有 Resize 逻辑。
        # MetSample outputs 256x256 by default. If target is also 256x256, disable resize logic completely.
        if resize_shape == (256, 256):
            self.resize_shape = None
        else:
            self.resize_shape = resize_shape 
        
        # LMDB 环境句柄缓存 (Per-worker LMDB Environment Cache)
        self.envs: Dict[str, lmdb.Environment] = {}
        
    def __len__(self):
        return len(self.samples)

    def _resize_array(self, data: np.ndarray, mode=cv2.INTER_NEAREST) -> np.ndarray:
        """
        数组缩放函数 (仅在 resize_shape 非空时调用)
        """
        # 即使在 _resize_array 内部，保留短路检查也是个好习惯，防止意外调用
        if self.resize_shape is None:
            return data

        dsize = (self.resize_shape[1], self.resize_shape[0])
        
        if data.ndim == 4: 
            T, C, H, W = data.shape
            reshaped = data.reshape(T * C, H, W)
            N = T * C
        elif data.ndim == 3: 
            C, H, W = data.shape
            reshaped = data
            N = C
        else:
            return data

        # 预分配内存 (Pre-allocate memory)
        resized = np.empty((N, dsize[1], dsize[0]), dtype=data.dtype)
        
        for i in range(N):
            cv2.resize(reshaped[i], dsize, dst=resized[i], interpolation=mode)
            
        if data.ndim == 4:
            return resized.reshape(T, C, *self.resize_shape)
        
        return resized

    def __getitem__(self, idx: int):
        """
        获取单个样本数据
        """
        record = self.samples[idx]
        
        # 1. 创建 MetSample 对象
        sample = MetSample(
            sample_id=record.get("sample_id"),
            timestamps=record.get("timestamps"),
            met_config=self.config,
            is_train=self.is_train,
        )
        
        # 2. LMDB 环境注入 (Environment Injection)
        region = sample.region_id
        if region not in self.envs:
            try:
                lmdb_path = os.path.join(sample.lmdb_root, f"{region}.lmdb")
                self.envs[region] = lmdb.open(
                    lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
                )
            except Exception as e:
                MLOGE(f"Failed to open LMDB for region {region}: {e}")
        
        if region in self.envs:
            sample.set_env(self.envs[region])
        
        # 3. 读取数据 (MetSample 默认返回 256x256)
        metadata, input_np, target_np, input_mask_np, target_mask_np = sample.to_numpy()
        
        # 4. 执行 Resize (仅当 self.resize_shape 非 None 时执行)
        # 如果初始化时已确定为 256x256，这里 self.resize_shape 为 None，直接跳过，零开销。
        if self.resize_shape is not None:
            FAST_MODE = cv2.INTER_NEAREST
            input_np = self._resize_array(input_np, mode=FAST_MODE)
            if target_np is not None:
                target_np = self._resize_array(target_np, mode=FAST_MODE)
            if target_mask_np is not None:
                target_mask_float = target_mask_np.astype(np.float32)
                target_mask_np = self._resize_array(target_mask_float, mode=FAST_MODE).astype(bool)
            if input_mask_np is not None:
                input_mask_float = input_mask_np.astype(np.float32)
                input_mask_np = self._resize_array(input_mask_float, mode=FAST_MODE).astype(bool)
        
        return metadata, input_np, target_np, target_mask_np, input_mask_np
                        
    def _load_samples_from_jsonl(self, file_path: str)-> List[Dict[str, Any]]:
        """从 JSONL 文件加载样本列表"""
        samples = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
        except FileNotFoundError:
            MLOGE(f"Dataset file not found: {file_path}")
        return samples
    
    def __del__(self):
        """析构函数: 清理 LMDB 资源"""
        for env in self.envs.values():
            try:
                env.close()
            except:
                pass
        self.envs.clear()

class ScwdsDataModule(LightningDataModule):
    """
    Lightning DataModule for SCWDS
    """
    def __init__(
        self,
        data_path: str = "data/samples.jsonl",
        resize_shape: tuple[int, int] = (256, 256),
        aft_seq_length: int = 20,
        batch_size: int = 4,
        num_workers: int = 8,
        pin_memory: bool = True,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.data_path = data_path
        self.resize_shape = resize_shape
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.pin_memory = pin_memory
        self.seed = seed

    def setup(self, stage: Optional[str] = None):
        if stage == "infer":
            self.infer_dataset = ScwdsDataset(
                self.data_path, 
                is_train=False,
                resize_shape=self.resize_shape
            )
            MLOGI(f"Infer dataset size: {len(self.infer_dataset)}")
            return
        
        if not hasattr(self, 'dataset'):
            self.dataset = ScwdsDataset(
                data_path=self.data_path,
                is_train=True,
                resize_shape=self.resize_shape 
            )
            
            total_size = len(self.dataset)
            if total_size == 0:
                MLOGI("Warning: Dataset is empty, skipping split")
                return
            
            train_size = int(self.train_split * total_size)
            val_size = int(self.val_split * total_size)
            test_size = max(0, total_size - train_size - val_size)
            
            if train_size == 0 and total_size > 0:
                train_size = total_size
                val_size = 0
                test_size = 0

            generator = torch.Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
                self.dataset, [train_size, val_size, test_size], generator=generator
            )
            
            MLOGI(f"Dataset split: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}, Test={len(self.test_dataset)}")

    def _collate_fn(self, batch):
        metadata_batch = []
        input_tensors = []
        target_tensors = []
        target_mask_tensors = []
        input_mask_tensors = []

        for metadata, input_np, target_np, target_mask_np, input_mask_np in batch:
            metadata_batch.append(metadata)
            input_tensors.append(torch.from_numpy(input_np).float())
            target_tensors.append(torch.from_numpy(target_np).float())
            target_mask_tensors.append(torch.from_numpy(target_mask_np).bool())
            input_mask_tensors.append(torch.from_numpy(input_mask_np).bool())

        input_batch = torch.stack(input_tensors, dim=0).contiguous()
        target_batch = torch.stack(target_tensors, dim=0).contiguous()
        target_mask_batch = torch.stack(target_mask_tensors, dim=0).contiguous()
        input_mask_batch = torch.stack(input_mask_tensors, dim=0).contiguous()
        
        return metadata_batch, input_batch, target_batch, target_mask_batch, input_mask_batch

    def _collate_fn_infer(self, batch):
        metadata_batch = []
        input_tensors = []
        input_mask_tensors = []

        for metadata, input_np, _, _, input_mask_np in batch:
            metadata_batch.append(metadata)
            input_tensors.append(torch.from_numpy(input_np).float())
            input_mask_tensors.append(torch.from_numpy(input_mask_np).bool())
        
        input_batch = torch.stack(input_tensors, dim=0).contiguous()
        input_mask_batch = torch.stack(input_mask_tensors, dim=0).contiguous()
        
        return metadata_batch, input_batch, input_mask_batch

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory, 
            persistent_workers=self.num_workers > 0, 
            collate_fn=self._collate_fn
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory, 
            persistent_workers=self.num_workers > 0, 
            collate_fn=self._collate_fn
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory, 
            persistent_workers=self.num_workers > 0, 
            collate_fn=self._collate_fn
        )

    def infer_dataloader(self) -> Optional[DataLoader]:
        return DataLoader(
            self.infer_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory, 
            persistent_workers=self.num_workers > 0, 
            collate_fn=self._collate_fn_infer
        )