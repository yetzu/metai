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
    4. [新增] 模糊度课程学习 (Blurring Curriculum)
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
    
    # 稀疏计算率 (Sparse Ratio)
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
    # 5. 高级策略配置 (Advanced Strategies)
    # =========================================================
    # 课程学习：启用序列长度和模糊度课程
    use_curriculum_learning: bool = True 
    
    # [新增] 模糊度课程参数
    # 初始最大高斯模糊半径 (sigma)，随着训练衰减
    blur_sigma_max: float = 2.0 
    # 应用模糊课程的 Epoch 数量 (例如前20个Epoch)
    blur_epochs: int = 20 
    
    # 时间权重：配合自动加权 Loss 使用
    use_temporal_weight: bool = True
    
    # [新增] 物理约束参数
    # 漏报强度的惩罚系数 (非对称 Loss)
    cons_under_penalty: float = 2.0 
    
    model_config = ConfigDict(protected_namespaces=())
    
    def to_dict(self):
        return self.model_dump()