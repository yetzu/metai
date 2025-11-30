# metai/dataset/met_dataset_scwds.py
import os
import json
import cv2
import lmdb
import numpy as np
from typing import List, Dict, Any, Optional
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

# 引入项目工具
from metai.utils import MLOGI, MLOGE
from metai.dataset import MetSample
from metai.utils.met_config import get_config

# ==============================================================================
# 全局配置
# ==============================================================================

# 关键设置：防止 OpenCV 多线程与 PyTorch DataLoader 多进程冲突导致的死锁与 CPU 争抢
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


# ==============================================================================
# 数据集定义 (Dataset)
# ==============================================================================

class ScwdsDataset(Dataset):
    """
    SCWDS (Severe Convective Weather Dataset) 数据集加载器。
    
    核心特性:
    1. LMDB 高效读取: 基于 LMDB 存储，支持随机访问。
    2. 环境句柄缓存: 针对 DataLoader 多进程 Worker 优化，每个 Worker 维护独立的 LMDB 环境句柄，避免频繁打开/关闭文件。
    3. 智能 Resize: 
       - 初始化时检测目标尺寸，若与源尺寸一致则完全关闭 Resize 逻辑（零开销）。
       - 运行时采用内存预分配与短路机制，最大化性能。
    """

    def __init__(
                self, 
                data_path: str, 
                is_train: bool = True, 
                resize_shape: Optional[tuple] = None,
                testset_name: str = "TestSet",        # [新增] 对应Key中的数据集部分
                lmdb_filename: Optional[str] = None   # [新增] 强制指定LMDB文件名(无后缀)
                ):
        """
        初始化数据集。
        
        Args:
            data_path (str): 样本列表文件路径 (.jsonl)。
            is_train (bool): 是否为训练模式 (决定是否加载标签数据)。
            resize_shape (tuple, optional): 目标尺寸 (H, W)。如果为 (256, 256) 或 None，则跳过 Resize 操作。
            testset_name (str): 推理模式下的数据集名称（如 "TestSet"）。
            lmdb_filename (str, optional): 强制指定读取的 LMDB 文件名（不带后缀）。
                                         如果指定（如 "TestSetA"），则忽略样本中的 region_id，
                                         统一从该文件读取。
        """
        self.data_path = data_path
        self.config = get_config()
        self.samples = self._load_samples_from_jsonl(data_path)
        self.is_train = is_train
        self.testset_name = testset_name
        
        # [优化] 初始化阶段的冗余检测 (Redundancy Check)
        # MetSample 默认输出 256x256。如果目标尺寸也是 256x256，则强制置为 None 以关闭所有 Resize 逻辑。
        if resize_shape == (256, 256):
            self.resize_shape = None
        else:
            self.resize_shape = resize_shape 
        
        # [优化] LMDB 环境句柄缓存 (Per-worker LMDB Environment Cache)
        # 这是一个字典，key 为 region_id，value 为打开的 lmdb.Environment 对象。
        # 仅在 __getitem__ 中按需初始化，并在 worker 销毁时清理。
        self.envs: Dict[str, lmdb.Environment] = {}
        
    def __len__(self):
        return len(self.samples)

    def _resize_array(self, data: np.ndarray, mode=cv2.INTER_NEAREST) -> np.ndarray:
        """
        高效数组缩放函数。
        
        优化策略:
        1. 短路检查: 如果尺寸已符合要求，直接返回。
        2. 内存预分配: 提前分配目标内存，cv2.resize 直接写入，减少内存碎片。
        
        Args:
            data: 输入数组，形状为 (..., H, W)。
            mode: 插值模式，默认 Nearest。
            
        Returns:
            np.ndarray: 缩放后的数组。
        """
        # 双重检查，防止意外调用
        if self.resize_shape is None:
            return data

        # 短路检查：如果最后两个维度符合要求，直接返回 (Zero-Copy)
        if data.shape[-2] == self.resize_shape[0] and data.shape[-1] == self.resize_shape[1]:
            return data

        dsize = (self.resize_shape[1], self.resize_shape[0]) # cv2 使用 (W, H)
        
        # 初始化变量以通过静态类型检查
        T, C, N = 0, 0, 0
        
        # 展平多维数组以便批量处理
        if data.ndim == 4: 
            T, C, H, W = data.shape
            reshaped = data.reshape(T * C, H, W)
            N = T * C
        elif data.ndim == 3: 
            C, H, W = data.shape
            reshaped = data
            N = C
        else:
            # 不支持的维度直接返回
            return data

        # [优化] 预分配连续内存
        resized = np.empty((N, dsize[1], dsize[0]), dtype=data.dtype)
        
        # 批量执行 Resize
        for i in range(N):
            cv2.resize(reshaped[i], dsize, dst=resized[i], interpolation=mode)
            
        # 还原原始维度结构
        if data.ndim == 4:
            return resized.reshape(T, C, *self.resize_shape)
        
        return resized

    def __getitem__(self, idx: int):
        """
        获取单个样本数据。
        """
        record = self.samples[idx]
        
        # 1. 创建 MetSample 对象
        # [安全] 使用 [] 访问确保 key 存在，显式转 str 满足类型要求
        sample = MetSample(
            sample_id=str(record["sample_id"]), 
            timestamps=record["timestamps"],
            met_config=self.config,
            is_train=self.is_train,
            testset_name=self.testset_name
        )
        
        # 2. LMDB 环境注入 (Environment Injection)
        if self.is_train:
            # 训练时按区域读取 (例如 AH.lmdb)
            db_key = sample.region_id
        else:
            # 推理时读取统一的测试集文件 (例如 TestSet.lmdb)
            db_key = self.testset_name
 
        if db_key not in self.envs:
            try:
                # 显式转 str 以满足 os.path.join 类型要求
                root_path = str(sample.lmdb_root)
                lmdb_path = os.path.join(root_path, f"{db_key}.lmdb")
                
                # 使用只读、无锁模式打开，最大化并发读取性能
                self.envs[db_key] = lmdb.open(
                    lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
                )
            except Exception as e:
                MLOGE(f"Failed to open LMDB for region {db_key}: {e}")
        
        # 将缓存的 env 注入 sample，复用连接
        if db_key in self.envs:
            sample.set_env(self.envs[db_key])
        
        # 3. 读取数据 (MetSample 默认返回 256x256)
        metadata, input_np, target_np, input_mask_np, target_mask_np = sample.to_numpy()
        
        # 4. 执行 Resize (仅当需要变分辨率时执行)
        if self.resize_shape is not None:
            # [优化] 使用 Nearest 插值保持物理极值
            FAST_MODE = cv2.INTER_NEAREST
            
            input_np = self._resize_array(input_np, mode=FAST_MODE)
            
            if target_np is not None:
                target_np = self._resize_array(target_np, mode=FAST_MODE)
            
            # [优化] Mask 转 uint8 -> resize -> bool
            # 避免使用 float32 进行插值，uint8 更快且内存更省
            if target_mask_np is not None:
                target_mask_uint8 = target_mask_np.astype(np.uint8)
                target_mask_np = self._resize_array(target_mask_uint8, mode=FAST_MODE).astype(bool)
            
            if input_mask_np is not None:
                input_mask_uint8 = input_mask_np.astype(np.uint8)
                input_mask_np = self._resize_array(input_mask_uint8, mode=FAST_MODE).astype(bool)
        
        # 注意：返回顺序需与 Collate Fn 对应
        return metadata, input_np, target_np, input_mask_np, target_mask_np
                        
    def _load_samples_from_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """从 JSONL 文件加载样本列表索引。"""
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
        """析构函数: 尝试关闭所有缓存的 LMDB 环境句柄。"""
        for env in self.envs.values():
            try:
                env.close()
            except:
                pass
        self.envs.clear()


# ==============================================================================
# Lightning 数据模块 (DataModule)
# ==============================================================================

class ScwdsDataModule(LightningDataModule):
    """
    Lightning DataModule for SCWDS.
    
    负责数据的划分、加载器构建以及 Batch 的堆叠 (Collate)。
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
        testset_name: str = "TestSet"
    ):
        super().__init__()
        # [重要] 保存超参数，以便 Checkpoint 加载时能恢复 DataModule 配置
        self.save_hyperparameters()
        
        self.data_path = data_path
        self.resize_shape = resize_shape
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.pin_memory = pin_memory
        self.seed = seed

        self.testset_name = testset_name

    def setup(self, stage: Optional[str] = None):
        """
        准备数据集 (Splitting & Setup)。
        """
        # 1. 推理模式 (Inference Mode)
        if stage == "infer":
            self.infer_dataset = ScwdsDataset(
                self.data_path, 
                is_train=False,
                resize_shape=self.resize_shape,
                testset_name=self.testset_name
            )
            MLOGI(f"Infer dataset size: {len(self.infer_dataset)}")
            return
        
        # 2. 训练/验证/测试模式 (Fit/Test Mode)
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
            
            # 计算切分大小
            train_size = int(self.train_split * total_size)
            val_size = int(self.val_split * total_size)
            test_size = max(0, total_size - train_size - val_size)
            
            # 边界情况：样本极少时的兜底策略
            if train_size == 0 and total_size > 0:
                train_size = total_size
                val_size = 0
                test_size = 0

            # 使用固定 Seed 进行随机切分，确保实验可复现
            generator = torch.Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
                self.dataset, [train_size, val_size, test_size], generator=generator
            )
            
            MLOGI(f"Dataset split: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}, Test={len(self.test_dataset)}")

    def _collate_fn(self, batch):
        """
        训练/验证 Collate 函数。
        
        [优化] 使用 torch.as_tensor 替代 torch.from_numpy + .float()，
        在数据类型匹配时实现内存共享 (Zero-Copy)。
        """
        metadata_batch = []
        input_tensors = []
        target_tensors = []
        input_mask_tensors = []
        target_mask_tensors = []

        for metadata, input_np, target_np, input_mask_np, target_mask_np in batch:
            metadata_batch.append(metadata)
            input_tensors.append(torch.as_tensor(input_np, dtype=torch.float32))
            target_tensors.append(torch.as_tensor(target_np, dtype=torch.float32))
            input_mask_tensors.append(torch.as_tensor(input_mask_np, dtype=torch.bool))
            target_mask_tensors.append(torch.as_tensor(target_mask_np, dtype=torch.bool))

        input_batch = torch.stack(input_tensors, dim=0)
        target_batch = torch.stack(target_tensors, dim=0)
        input_mask_batch = torch.stack(input_mask_tensors, dim=0)
        target_mask_batch = torch.stack(target_mask_tensors, dim=0)
        
        return metadata_batch, input_batch, target_batch, input_mask_batch, target_mask_batch

    def _collate_fn_infer(self, batch):
        """
        推理模式 Collate 函数 (无标签)。
        """
        metadata_batch = []
        input_tensors = []
        input_mask_tensors = []

        for metadata, input_np, _, input_mask_np, _ in batch:
            metadata_batch.append(metadata)
            input_tensors.append(torch.as_tensor(input_np, dtype=torch.float32))
            input_mask_tensors.append(torch.as_tensor(input_mask_np, dtype=torch.bool))
        
        input_batch = torch.stack(input_tensors, dim=0)
        input_mask_batch = torch.stack(input_mask_tensors, dim=0)
        
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