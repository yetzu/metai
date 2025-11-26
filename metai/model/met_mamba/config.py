# metai/model/met_mamba/config.py

from pydantic import BaseModel, Field, ConfigDict
from typing import Tuple

class MeteoMambaConfig(BaseModel):
    # --- 路径与基础 ---
    data_path: str = "data/samples.jsonl"
    save_dir: str = "./output/mamba"
    
    # --- 形状参数 ---
    # T=10 输入, 预测 20 帧
    in_shape: Tuple[int, int, int, int] = (10, 31, 256, 256) 
    aft_seq_length: int = 20
    
    # --- 模型参数 (轻量化配置) ---
    # [修改] 64 通道对于 Pure 2D Encoder + 256x256 图像足够，速度快
    hid_S: int = 64   
    hid_T: int = 512  # Mamba 核心维持较大容量
    N_S: int = 4
    N_T: int = 8
    spatio_kernel_enc: int = 3
    spatio_kernel_dec: int = 3
    
    # --- 训练参数 ---
    # [修改] 显存优化后，Batch Size 可以开大
    batch_size: int = 8 
    accumulate_grad_batches: int = 1 
    max_epochs: int = 50
    
    # --- 优化器与调度器 ---
    opt: str = "adamw"
    lr: float = 1e-3
    min_lr: float = 1e-5
    warmup_lr: float = 1e-5
    warmup_epoch: int = 5
    weight_decay: float = 0.01
    momentum: float = 0.9
    sched: str = "cosine"
    decay_epoch: int = 30
    decay_rate: float = 0.1
    
    # --- 策略 ---
    use_curriculum_learning: bool = False # 简单起见关闭
    
    # --- 损失函数权重 (适配 Sparse + GDL) ---
    # l1_weight 控制 SparseBalancedL1Loss 的整体幅度
    loss_weight_l1: float = 1.0  
    # gdl_weight 控制抗模糊力度
    loss_weight_gdl: float = 1.0
    
    model_config = ConfigDict(protected_namespaces=())
    
    def to_dict(self):
        return self.model_dump()