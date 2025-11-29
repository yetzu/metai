# metai/model/met_mamba/config.py

from pydantic import BaseModel, ConfigDict
from typing import Tuple

class ModelConfig(BaseModel):
    """
    ModelConfig 模型配置 
    """
    # --- 路径与基础 ---
    data_path: str = "data/samples.jsonl"
    save_dir: str = "./output" # 修改输出目录以区分实验
    
    # --- 形状参数 ---
    # C=31 (多通道), H=W=256
    in_shape: Tuple[int, int, int] = (31, 256, 256) 
    obs_seq_len: int = 10 
    pred_seq_len: int = 20 
    
    # --- 模型参数 (A800 高性能配置) ---
    # [优化] hid_S 从 64 提升至 128。
    hid_S: int = 128   
    
    # [保持] hid_T 维持 512。
    hid_T: int = 512  
    
    N_S: int = 4
    N_T: int = 8 

    # [新增] Mamba 核心参数
    mamba_d_state: int = 16  
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    
    # --- 训练参数 ---
    batch_size: int = 32 
    accumulate_grad_batches: int = 1 
    
    # [优化] 增加 Epochs 至 100。
    max_epochs: int = 100
    
    # --- 优化器与调度器 ---
    opt: str = "adamw"
    lr: float = 1e-3 
    min_lr: float = 1e-5
    
    # [优化] 预热 Epoch 从 5 增加至 10。
    warmup_lr: float = 1e-5
    warmup_epoch: int = 10 
    
    weight_decay: float = 0.05 
    momentum: float = 0.9
    sched: str = "cosine"
    decay_epoch: int = 30 
    decay_rate: float = 0.1
    
    # --- 策略 ---
    use_curriculum_learning: bool = False 
    
    # --- 损失函数权重 ---
    loss_weight_l1: float = 1.0  
    loss_weight_gdl: float = 1.0
    
    model_config = ConfigDict(protected_namespaces=())
    
    def to_dict(self):
        return self.model_dump()