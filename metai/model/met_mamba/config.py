from pydantic import BaseModel, Field, ConfigDict
from typing import Tuple

class MeteoMambaConfig(BaseModel):
    # --- 路径与基础 ---
    data_path: str = "data/samples.jsonl"
    save_dir: str = "./output/mamba"
    
    # --- 形状参数 ---
    in_shape: Tuple[int, int, int, int] = (10, 31, 256, 256)
    aft_seq_length: int = 20
    
    # --- 模型参数 ---
    hid_S: int = 128
    hid_T: int = 512
    N_S: int = 4
    N_T: int = 8
    spatio_kernel_enc: int = 3
    spatio_kernel_dec: int = 3
    
    # --- 训练参数 ---
    batch_size: int = 4
    accumulate_grad_batches: int = 4 
    max_epochs: int = 50
    
    # --- 优化器与调度器 (Fix: 移除了硬编码，参数化配置) ---
    opt: str = "adamw"
    lr: float = 1e-3
    min_lr: float = 1e-5
    warmup_lr: float = 1e-5
    warmup_epoch: int = 5
    weight_decay: float = 0.01
    momentum: float = 0.9
    sched: str = "cosine"
    decay_epoch: int = 30      # Step Decay 的间隔
    decay_rate: float = 0.1    # Step Decay 的衰减率
    
    # --- 策略 ---
    use_curriculum_learning: bool = True
    
    # --- 损失函数初始权重 (Fix: 显式定义初始权重) ---
    loss_weight_l1: float = 1.0
    loss_weight_ssim: float = 0.5
    loss_weight_csi: float = 1.0
    loss_weight_spectral: float = 0.1
    loss_weight_evo: float = 0.5
    loss_weight_focal: float = 0.0 # 初始为0，通过 Curriculum Learning 增加
    
    model_config = ConfigDict(protected_namespaces=())
    
    def to_dict(self):
        return self.model_dump()