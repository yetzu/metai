from pydantic import BaseModel, ConfigDict
from typing import Tuple

class ModelConfig(BaseModel):
    """
    MeteoMamba 模型配置类
    """
    # 基础配置
    data_path: str = "data/samples.jsonl"
    save_dir: str = "./output"
    
    # 形状参数
    in_shape: Tuple[int, int, int] = (31, 256, 256) 
    obs_seq_len: int = 10 
    pred_seq_len: int = 20 
    
    # 模型架构 (STMamba)
    hid_S: int = 128
    hid_T: int = 512
    N_S: int = 4
    N_T: int = 8 
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    use_checkpoint: bool = True # 显存优化
    
    # 训练参数
    batch_size: int = 4 
    accumulate_grad_batches: int = 1 
    max_epochs: int = 100
    
    # 优化器
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
    use_curriculum_learning: bool = True 
    
    # --- 损失权重 (Unified Naming) ---
    weight_focal: float = 1.0
    weight_msssim: float = 1.0
    weight_corr: float = 0.5
    weight_csi: float = 1.0
    weight_evo: float = 0.5
    
    # --- 损失细节参数 ---
    focal_alpha: float = 2.0
    focal_gamma: float = 1.0
    intensity_weights: Tuple[float, ...] = (0.1, 1.0, 2.0, 5.0, 10.0, 20.0)
    false_alarm_penalty: float = 5.0
    corr_smooth_eps: float = 1e-4
    
    model_config = ConfigDict(protected_namespaces=())
    
    def to_dict(self):
        return self.model_dump()