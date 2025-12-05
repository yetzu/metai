# metai/model/met_mamba/config.py

from pydantic import BaseModel, ConfigDict
from typing import Tuple

class ModelConfig(BaseModel):
    """
    MeteoMamba 模型配置类 (No-GAN Version)
    """
    
    # 1. 基础环境
    data_path: str = "data/samples.jsonl"
    save_dir: str = "./output"
    
    # 2. 数据参数
    in_shape: Tuple[int, int, int] = (31, 256, 256)
    obs_seq_len: int = 10
    pred_seq_len: int = 20
    out_channels: int = 1
    
    # 3. 架构参数
    hid_S: int = 128
    hid_T: int = 512
    N_S: int = 4
    N_T: int = 8
    
    mamba_d_state: int = 32
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    use_checkpoint: bool = True
    
    mamba_sparse_ratio: float = 0.5
    anneal_start_epoch: int = 5
    anneal_end_epoch: int = 15
    
    # 4. 噪声配置 (保留，用于EvolutionNet随机性)
    noise_dim: int = 32
    
    # 5. 训练参数
    batch_size: int = 4
    accumulate_grad_batches: int = 1
    max_epochs: int = 100
    
    opt: str = "adamw"
    lr: float = 1e-3
    min_lr: float = 1e-5
    warmup_lr: float = 1e-5
    warmup_epoch: int = 5
    weight_decay: float = 0.05
    momentum: float = 0.9
    filter_bias_and_bn: bool = True
    
    sched: str = "cosine"
    decay_epoch: int = 30
    decay_rate: float = 0.1
    
    # 6. 策略配置
    use_curriculum_learning: bool = True 
    blur_sigma_max: float = 2.0 
    blur_epochs: int = 20 
    use_temporal_weight: bool = True
    
    model_config = ConfigDict(protected_namespaces=())
    
    def to_dict(self):
        return self.model_dump()