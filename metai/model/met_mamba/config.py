# metai/model/met_mamba/config.py

from pydantic import BaseModel, ConfigDict
from typing import Tuple

class ModelConfig(BaseModel):
    """
    MeteoMamba V3 模型配置类
    包含架构参数、优化器参数及 CVAE 训练策略参数
    """
    
    # 1. 基础环境
    data_path: str = "data/samples.jsonl"
    save_dir: str = "./output"
    
    # 2. 数据参数
    in_shape: Tuple[int, int, int] = (31, 256, 256)
    in_seq_len: int = 10
    out_seq_len: int = 20
    out_channels: int = 1
    
    # 3. 架构参数 (Architecture)
    hid_S: int = 64
    hid_T: int = 256
    N_S: int = 4
    N_T: int = 8
    
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    use_checkpoint: bool = True
    
    # 4. CVAE 生成参数
    noise_dim: int = 32  # 隐变量 z 的维度
    
    # 5. 训练参数 (Training)
    batch_size: int = 4
    accumulate_grad_batches: int = 1
    max_epochs: int = 50
    
    opt: str = "adamw"
    lr: float = 1e-3
    min_lr: float = 1e-5
    warmup_lr: float = 1e-5
    warmup_epoch: int = 5
    weight_decay: float = 0.01
    
    sched: str = "cosine"
    
    # 6. 损失与策略参数 (Loss & Strategy) - [修改]
    weight_mae: float = 10.0
    weight_csi: float = 1.0
    weight_corr: float = 1.0
    
    # KL 退火策略 (防止 Posterior Collapse)
    kl_weight_max: float = 0.01  # KL Loss 的最大权重
    kl_anneal_epochs: int = 10   # 在前 N 个 epoch 内线性增加 KL 权重
    
    # 7. 其它
    use_temporal_weight: bool = True # 是否在 Loss 中对时间步加权
    
    model_config = ConfigDict(protected_namespaces=())
    
    def to_dict(self):
        return self.model_dump()