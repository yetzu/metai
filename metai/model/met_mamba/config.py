# metai/model/met_mamba/config.py

from pydantic import BaseModel, Field, ConfigDict
from typing import Tuple

class MeteoMambaConfig(BaseModel):
    """
    MeteoMamba 模型配置 - A800 优化版
    针对 NVIDIA A800 (80GB) 显卡与 20k 样本量进行调优。
    """
    # --- 路径与基础 ---
    data_path: str = "data/samples.jsonl"
    save_dir: str = "./output/mamba_a800_opt" # 修改输出目录以区分实验
    
    # --- 形状参数 ---
    # T_in=10 (过去1小时), T_out=20 (未来2小时), C=31 (多通道), H=W=256
    in_shape: Tuple[int, int, int, int] = (10, 31, 256, 256) 
    aft_seq_length: int = 20
    
    # --- 模型参数 (A800 高性能配置) ---
    # [优化] hid_S 从 64 提升至 128。
    # A800 显存充足，增加 2D Encoder 通道数可显著提升对微小纹理(强降水中心)的捕捉能力。
    hid_S: int = 128   
    
    # [保持] hid_T 维持 512。
    # 针对 20,000 样本，512 是最佳平衡点。过大(如1024)容易导致过拟合，过小则欠拟合。
    hid_T: int = 512  
    
    N_S: int = 4
    N_T: int = 8 # 保持 8 层，若模型收敛很快但上限不够，可尝试改为 12
    spatio_kernel_enc: int = 3
    spatio_kernel_dec: int = 3

    # [新增] Mamba 核心参数
    mamba_d_state: int = 16  # 默认值，脚本中可覆盖
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    
    # --- 训练参数 ---
    # [优化] Batch Size 从 8 提升至 32。
    # A800 80GB 显存允许大 Batch，这能提供更准确的梯度估计，使 Loss 下降更平稳。
    # 注意：务必配合 BF16 (bfloat16) 精度训练使用。
    batch_size: int = 32 
    accumulate_grad_batches: int = 1 
    
    # [优化] 增加 Epochs 至 100。
    # 大 Batch Size 通常需要更多的迭代次数来收敛。
    max_epochs: int = 100
    
    # --- 优化器与调度器 ---
    opt: str = "adamw"
    lr: float = 1e-3 # 保持 1e-3，大 Batch 下此学习率通常安全
    min_lr: float = 1e-5
    
    # [优化] 预热 Epoch 从 5 增加至 10。
    # Mamba 类模型初期梯度波动大，大 Batch 下更需要充分预热。
    warmup_lr: float = 1e-5
    warmup_epoch: int = 10 
    
    weight_decay: float = 0.05 # 稍微增加正则化(原0.01)，防止 2w 样本过拟合
    momentum: float = 0.9
    sched: str = "cosine"
    decay_epoch: int = 30 # Cosine 策略下此参数主要用于 StepLR，保留即可
    decay_rate: float = 0.1
    
    # --- 策略 ---
    use_curriculum_learning: bool = False 
    
    # --- 损失函数权重 ---
    # 建议配合 loss.py 中的 rain_weight=8.0 修改使用
    loss_weight_l1: float = 1.0  
    loss_weight_gdl: float = 1.0
    
    model_config = ConfigDict(protected_namespaces=())
    
    def to_dict(self):
        return self.model_dump()