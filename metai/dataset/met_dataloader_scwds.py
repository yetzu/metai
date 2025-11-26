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

# 🚨 关键修复：防止多进程死锁
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

    def _resize_array(self, data, mode=cv2.INTER_LINEAR):
        if self.resize_shape is None:
            return data
        
        # cv2.resize 接受 (W, H)
        dsize = (self.resize_shape[1], self.resize_shape[0])
        
        if data.ndim == 4: 
            T, C, H, W = data.shape
            reshaped = data.reshape(T * C, H, W)
            # 此时 cv2 是单线程运行，安全
            resized = np.array([cv2.resize(img, dsize, interpolation=mode) for img in reshaped])
            return resized.reshape(T, C, *self.resize_shape)
            
        elif data.ndim == 3: 
            C, H, W = data.shape
            resized = np.array([cv2.resize(img, dsize, interpolation=mode) for img in data])
            return resized.reshape(C, *self.resize_shape)
            
        return data

    def __getitem__(self, idx):
        record = self.samples[idx]
        
        sample = MetSample.create(
            record.get("sample_id"),
            record.get("timestamps"),
            config=self.config,
            is_train=self.is_train,
        )
        
        metadata, input_np, target_np, target_mask_np, input_mask_np = sample.to_numpy(is_train=self.is_train)
        
        if self.resize_shape is not None:
            input_np = self._resize_array(input_np, mode=cv2.INTER_LINEAR)
            
            if target_np is not None:
                target_np = self._resize_array(target_np, mode=cv2.INTER_LINEAR)
            
            if target_mask_np is not None:
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
                    sample = json.loads(line)
                    samples.append(sample)
        return samples

class ScwdsDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str = "data/samples.jsonl",
        resize_shape: tuple[int, int] = (256, 256),
        aft_seq_length: int = 20,
        batch_size: int = 4,
        num_workers: int = 8,  # 建议在脚本中覆盖此值
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
            test_size = total_size - train_size - val_size
            
            # 边界情况处理
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
        if not hasattr(self, 'train_dataset'):
            self.setup('fit')
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=True if self.num_workers > 0 else False, collate_fn=self._collate_fn)

    def val_dataloader(self) -> Optional[DataLoader]:
        if not hasattr(self, 'val_dataset'):
            self.setup('fit')
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=True if self.num_workers > 0 else False, collate_fn=self._collate_fn)

    def test_dataloader(self) -> Optional[DataLoader]:
        if not hasattr(self, 'test_dataset'):
            self.setup('test')
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=True if self.num_workers > 0 else False, collate_fn=self._collate_fn)

    def infer_dataloader(self) -> Optional[DataLoader]:
        if not hasattr(self, 'infer_dataset'):
            self.setup('infer')
        return DataLoader(self.infer_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=True if self.num_workers > 0 else False, collate_fn=self._collate_fn_infer)