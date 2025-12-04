# metai/model/met_mamba/config.py

from pydantic import BaseModel, ConfigDict
from typing import Tuple

class ModelConfig(BaseModel):
    """
    MeteoMamba 模型配置类 (Configuration) - v2.0
    
    集成功能：
    1. 自动加权 Loss (Kendall's Weighting)
    2. 物理约束 (Physics Constraints)
    3. 稀疏计算 (Sparse Computation & Attention)
    4. 模糊度课程学习 (Blurring Curriculum)
    5. [新增] 生成式对抗训练 (GAN & Noise Injection)
    """
    
    # =========================================================
    # 1. 基础环境配置
    # =========================================================
    data_path: str = "data/samples.jsonl"
    save_dir: str = "./output"
    
    # =========================================================
    # 2. 数据与形状参数
    # =========================================================
    in_shape: Tuple[int, int, int] = (31, 256, 256) # (C, H, W)
    obs_seq_len: int = 10
    pred_seq_len: int = 20
    out_channels: int = 1 # 预测目标的通道数 (通常为1: 反射率)
    
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
    
    # 稀疏计算与退火参数 (Sparse Computation & Annealing)
    mamba_sparse_ratio: float = 0.5  # 目标稀疏率 (0.0 表示关闭稀疏)
    anneal_start_epoch: int = 5      # 开始进行稀疏化退火的 Epoch
    anneal_end_epoch: int = 15       # 达到目标稀疏率的 Epoch
    
    # =========================================================
    # 4. 生成式能力配置 (Generative & GAN) [新增]
    # =========================================================
    # 对抗训练配置
    use_gan: bool = True             # 是否启用 GAN 对抗训练
    gan_start_epoch: int = 0         # 从第几个 Epoch 开始训练 GAN (建议预热后再开)
    gan_weight: float = 0.01         # 对抗损失 (Adversarial Loss) 的初始权重
    
    # 判别器配置
    discriminator_base_dim: int = 64 # 判别器基础通道数
    discriminator_lr: float = 2e-4   # 判别器学习率 (通常比生成器低)
    
    # 噪声注入配置 (解决模糊问题)
    noise_dim: int = 32              # 注入 EvolutionNet 的随机噪声维度
    
    # =========================================================
    # 5. 训练超参数
    # =========================================================
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
    
    # =========================================================
    # 6. 高级策略配置 (Advanced Strategies)
    # =========================================================
    # 课程学习：启用序列长度和模糊度课程
    use_curriculum_learning: bool = True 
    
    # 模糊度课程参数
    # 初始最大高斯模糊半径 (sigma)，随着训练衰减
    blur_sigma_max: float = 2.0 
    # 应用模糊课程的 Epoch 数量 (例如前20个Epoch)
    blur_epochs: int = 20 
    
    # 时间权重：配合自动加权 Loss 使用，给长时预测更高权重
    use_temporal_weight: bool = True
    
    # 物理约束参数
    # 漏报强度的惩罚系数 (非对称 Loss)
    cons_under_penalty: float = 2.0 
    
    model_config = ConfigDict(protected_namespaces=())
    
    def to_dict(self):
        return self.model_dump()