from pydantic import BaseModel, ConfigDict
from typing import Tuple

class ModelConfig(BaseModel):
    """
    ModelConfig 模型配置
    """
    # --- 路径与基础 ---
    data_path: str = "data/samples.jsonl"
    save_dir: str = "./output"
    
    # --- 形状参数 ---
    # Input Shape: (C, H, W)
    in_shape: Tuple[int, int, int] = (31, 256, 256) 
    obs_seq_len: int = 10 
    pred_seq_len: int = 20 
    
    # --- 模型参数 ---
    hid_S: int = 128   
    hid_T: int = 512  
    N_S: int = 4
    N_T: int = 8 

    # --- Mamba 核心参数 ---
    mamba_d_state: int = 16  
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    
    # --- 训练参数 ---
    batch_size: int = 32 
    accumulate_grad_batches: int = 1 
    max_epochs: int = 100
    
    # --- 优化器与调度器 ---
    opt: str = "adamw"
    lr: float = 1e-3 
    min_lr: float = 1e-5
    warmup_lr: float = 1e-5
    warmup_epoch: int = 10 
    weight_decay: float = 0.05 
    momentum: float = 0.9
    
    sched: str = "cosine"
    decay_epoch: int = 30 
    decay_rate: float = 0.1
    
    # --- 策略 ---
    use_curriculum_learning: bool = False 
    
    # --- 损失函数权重 (已同步 loss.py 策略) ---
    loss_weight_l1: float = 1.0    # MAE 基准权重
    loss_weight_gdl: float = 10.0  # [修改] 放大以匹配 MAE 量级
    loss_weight_corr: float = 0.5  # [新增] 相关性损失权重
    loss_weight_dice: float = 1.0  # [新增] TS/Dice 损失权重
    
    model_config = ConfigDict(protected_namespaces=())
    
    def to_dict(self):
        return self.model_dump()