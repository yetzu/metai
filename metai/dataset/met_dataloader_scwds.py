import json
from typing import List, Dict, Any, Optional
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F 
from metai.utils import MLOGI
from metai.dataset import MetSample
from metai.utils.met_config import get_config

class ScwdsDataset(Dataset):
    def __init__(self, data_path: str, is_train: bool = True):
        self.data_path = data_path
        self.config = get_config()
        self.samples = self._load_samples_from_jsonl(data_path)
        self.is_train = is_train
        
    def __len__(self):
        """返回样本数量"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取指定索引的样本数据
        
        Args:
            idx: 样本索引
            
        Returns: 
            tuple[Dict, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
                - (metadata, input_data, target_data, target_mask, input_mask)
                - target_data/target_mask 在推理模式下为 None
        """
        record = self.samples[idx]
        
        sample = MetSample.create(
            record.get("sample_id"),
            record.get("timestamps"),
            config=self.config,
            is_train=self.is_train,
        )
        
        return sample.to_numpy(is_train=self.is_train) 
                        
    def _load_samples_from_jsonl(self, file_path: str)-> List[Dict[str, Any]]:
        """加载JSONL文件中的样本数据"""
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
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
        """设置数据集"""
        if stage == "infer":
            # 推理模式下，直接创建推理数据集，不需要进行 train/val/test 分割
            self.infer_dataset = ScwdsDataset(
                self.data_path, 
                is_train=False
            )
            MLOGI(f"Infer dataset size: {len(self.infer_dataset)}")
            return
        
        # 训练/验证/测试模式下，进行数据集分割
        if not hasattr(self, 'dataset'):
            self.dataset = ScwdsDataset(
                data_path=self.data_path,
                is_train=True
            )
            
            total_size = len(self.dataset)
            
            # 如果数据集为空或太小，跳过分割
            if total_size == 0:
                MLOGI("Warning: Dataset is empty, skipping split")
                return
            
            # 计算划分的尺寸
            train_size = int(self.train_split * total_size)
            val_size = int(self.val_split * total_size)
            test_size = total_size - train_size - val_size
            
            # 确保至少有一个样本在训练集中（如果数据集不为空）
            if train_size == 0 and total_size > 0:
                train_size = 1
                test_size = total_size - train_size - val_size
            
            lengths = [train_size, val_size, test_size]

            # 创建随机生成器以确保划分的可复现性
            generator = torch.Generator().manual_seed(self.seed)

            # 使用 torch.utils.data.random_split 进行随机划分
            self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
                self.dataset, lengths, generator=generator
            )
            
            MLOGI(f"Dataset split: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}, Test={len(self.test_dataset)}")

    def _interpolate_batch(self, batch_tensor: torch.Tensor, mode: str = 'bilinear') -> torch.Tensor:
        """
        高效的批量插值函数，处理 (B, T, C, H, W) 格式的tensor
        """
        B, T, C, H, W = batch_tensor.shape
        batch_tensor = batch_tensor.view(B * T, C, H, W)
        batch_tensor = F.interpolate(batch_tensor, size=self.resize_shape, mode=mode, align_corners=False if mode == 'bilinear' else None)
        return batch_tensor.view(B, T, C, *self.resize_shape)

    def _collate_fn(self, batch):
        """
        自定义collate函数，用于训练/验证/测试。
        """
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
        """
        自定义collate函数，用于推理。
        """
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
        """返回训练数据加载器"""
        if not hasattr(self, 'train_dataset'):
            self.setup('fit')
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """返回验证数据加载器"""
        if not hasattr(self, 'val_dataset'):
            self.setup('fit')
            
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """返回测试数据加载器"""
        if not hasattr(self, 'test_dataset'):
            self.setup('test')
            
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self._collate_fn,
        )

    def infer_dataloader(self) -> Optional[DataLoader]:
        """返回推理数据加载器"""
        if not hasattr(self, 'infer_dataset'):
            self.setup('infer')
            
        return DataLoader(
            self.infer_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self._collate_fn_infer # 推理使用三个元素的 collate_fn_infer
        )