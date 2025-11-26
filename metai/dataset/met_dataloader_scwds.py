# metai/dataset/met_dataloader_scwds.py
import json
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F 
from metai.utils import MLOGI
from metai.dataset import MetSample
from metai.utils.met_config import get_config

# 🚨 关键设置：防止多进程死锁
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class ScwdsDataset(Dataset):
    def __init__(self, data_path: str, is_train: bool = True, resize_shape: Optional[tuple] = None):
        self.data_path = data_path
        self.config = get_config()
        self.samples = self._load_samples_from_jsonl(data_path)
        self.is_train = is_train
        self.resize_shape = resize_shape 
        
    def __len__(self):
        return len(self.samples)

    def _resize_array(self, data, mode=cv2.INTER_NEAREST):
        """
        [优化版] 高效 Resize
        1. 支持 mode 指定 (推荐 INTER_NEAREST 以追求极致速度)
        2. 使用内存预分配 (Pre-allocation)，减少内存碎片和GC压力
        """
        if self.resize_shape is None:
            return data
        
        # cv2.resize 接受 (W, H)
        dsize = (self.resize_shape[1], self.resize_shape[0])
        
        # 1. 统一数据视图为 (N, H, W)
        if data.ndim == 4: 
            T, C, H, W = data.shape
            reshaped = data.reshape(T * C, H, W)
            N = T * C
        elif data.ndim == 3: 
            C, H, W = data.shape
            reshaped = data
            N = C
        else:
            return data # 不处理其他维度

        # 2. [核心优化] 预分配连续内存
        # 避免在循环中反复创建 numpy array，大幅提升速度
        resized = np.empty((N, dsize[1], dsize[0]), dtype=data.dtype)
        
        # 3. 循环填充 (由于预分配了内存，此处极快)
        for i in range(N):
            # dst=resized[i] 让 opencv 直接写入目标内存
            cv2.resize(reshaped[i], dsize, dst=resized[i], interpolation=mode)
            
        # 4. 还原维度
        if data.ndim == 4:
            return resized.reshape(T, C, *self.resize_shape)
        
        return resized # (C, H_new, W_new)

    def __getitem__(self, idx):
        record = self.samples[idx]
        
        sample = MetSample.create(
            record.get("sample_id"),
            record.get("timestamps"),
            config=self.config,
            is_train=self.is_train,
        )
        
        # 1. 获取原始数据
        metadata, input_np, target_np, target_mask_np, input_mask_np = sample.to_numpy(is_train=self.is_train)
        
        # 2. 执行 Resize
        if self.resize_shape is not None:
            # [核心修改] 全部统一使用最邻近插值 (INTER_NEAREST) 以降低开销
            # 气象数据通常是稠密的，Nearest 不会造成显著的物理性质丢失，且不会引入平滑模糊
            FAST_MODE = cv2.INTER_NEAREST
            
            input_np = self._resize_array(input_np, mode=FAST_MODE)
            
            if target_np is not None:
                target_np = self._resize_array(target_np, mode=FAST_MODE)
            
            if target_mask_np is not None:
                # Mask 必须是 Nearest
                target_mask_float = target_mask_np.astype(np.float32)
                target_mask_np = self._resize_array(target_mask_float, mode=cv2.INTER_NEAREST).astype(bool)
            
            if input_mask_np is not None:
                input_mask_float = input_mask_np.astype(np.float32)
                input_mask_np = self._resize_array(input_mask_float, mode=cv2.INTER_NEAREST).astype(bool)
        
        return metadata, input_np, target_np, target_mask_np, input_mask_np
                        
    def _load_samples_from_jsonl(self, file_path: str)-> List[Dict[str, Any]]:
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

class ScwdsDataModule(LightningDataModule):
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
        self.original_shape = (301, 301)
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
                train_size = 1
                test_size = max(0, total_size - train_size - val_size)
            
            lengths = [train_size, val_size, test_size]

            generator = torch.Generator().manual_seed(self.seed)

            self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
                self.dataset, lengths, generator=generator
            )
            
            MLOGI(f"Dataset split: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}, Test={len(self.test_dataset)}")

    def _interpolate_batch(self, batch_tensor: torch.Tensor, mode: str = 'bilinear') -> torch.Tensor:
        if batch_tensor.shape[-2:] == self.resize_shape:
            return batch_tensor
        B, T, C, H, W = batch_tensor.shape
        batch_tensor = batch_tensor.view(B * T, C, H, W)
        batch_tensor = F.interpolate(batch_tensor, size=self.resize_shape, mode=mode, align_corners=False if mode == 'bilinear' else None)
        return batch_tensor.view(B, T, C, *self.resize_shape)

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

    def train_dataloader(self):
        if not hasattr(self, 'train_dataset'): self.setup('fit')
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=True if self.num_workers > 0 else False, collate_fn=self._collate_fn)

    def val_dataloader(self) -> Optional[DataLoader]:
        if not hasattr(self, 'val_dataset'): self.setup('fit')
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=True if self.num_workers > 0 else False, collate_fn=self._collate_fn)

    def test_dataloader(self) -> Optional[DataLoader]:
        if not hasattr(self, 'test_dataset'): self.setup('test')
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=True if self.num_workers > 0 else False, collate_fn=self._collate_fn)

    def infer_dataloader(self) -> Optional[DataLoader]:
        if not hasattr(self, 'infer_dataset'): self.setup('infer')
        return DataLoader(self.infer_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=True if self.num_workers > 0 else False, collate_fn=self._collate_fn_infer)