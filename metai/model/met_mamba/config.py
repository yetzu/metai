# metai/model/met_mamba/config.py

from pydantic import BaseModel, ConfigDict
from typing import Tuple

class ModelConfig(BaseModel):
    """
    MeteoMamba 模型配置类 (Configuration)
    
    集成功能：
    1. 自动加权 Loss (Kendall's Weighting)
    2. 物理约束 (Physics Constraints)
    3. 稀疏计算 (Sparse Computation)
    """
    
    # =========================================================
    # 1. 基础环境配置
    # =========================================================
    data_path: str = "data/samples.jsonl"
    save_dir: str = "./output"
    
    # =========================================================
    # 2. 数据与形状参数
    # =========================================================
    in_shape: Tuple[int, int, int] = (31, 256, 256)
    obs_seq_len: int = 10
    pred_seq_len: int = 20
    
    # =========================================================
    # 3. 模型架构参数 (Model Architecture)
    # =========================================================
    hid_S: int = 128
    hid_T: int = 512
    N_S: int = 4
    N_T: int = 8
    
    # Mamba 核心参数
    mamba_d_state: int = 32
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    use_checkpoint: bool = True
    
    # [新增] 稀疏计算率 (Sparse Ratio)
    # 控制 Mamba 层的 Token 剪枝比例。
    # 0.0 = 关闭稀疏 (稠密计算)
    # 0.5 = 剪枝 50% (加速计算，仅保留高激活 Token)
    mamba_sparse_ratio: float = 0.5 
    
    # =========================================================
    # 4. 训练超参数
    # =========================================================
    batch_size: int = 4
    accumulate_grad_batches: int = 1
    max_epochs: int = 100
    
    opt: str = "adamw"
    lr: float = 1e-3
    min_lr: float = 1e-5
    warmup_lr: float = 1e-5
    warmup_epoch: int = 20
    weight_decay: float = 0.05
    momentum: float = 0.9
    filter_bias_and_bn: bool = True
    
    sched: str = "cosine"
    decay_epoch: int = 30
    decay_rate: float = 0.1
    
    # =========================================================
    # 5. 高级策略配置
    # =========================================================
    # 课程学习：这对稳定光流 (Advective Projection) 至关重要
    use_curriculum_learning: bool = True 
    
    # 时间权重：配合自动加权 Loss 使用
    use_temporal_weight: bool = True
    
    model_config = ConfigDict(protected_namespaces=())
    
    def to_dict(self):
        return self.model_dump()