# metai/model/met_mamba/config.py

from pydantic import BaseModel, ConfigDict
from typing import Tuple

class ModelConfig(BaseModel):
    """
    MeteoMamba 模型配置类 (Configuration)
    
    管理模型架构、训练超参数以及损失函数权重的统一配置接口。
    基于 Pydantic 构建，支持类型检查与自动验证。
    """
    
    # =========================================================
    # 1. 基础环境配置 (Basic Settings)
    # =========================================================
    data_path: str = "data/samples.jsonl"  # 训练数据索引文件路径 (.jsonl)
    save_dir: str = "./output"             # 模型检查点与日志的保存目录
    
    # =========================================================
    # 2. 数据与形状参数 (Data & Shape)
    # =========================================================
    in_shape: Tuple[int, int, int] = (31, 256, 256) # 输入张量形状 (Channels, Height, Width)
    obs_seq_len: int = 10   # 观测序列长度 (输入帧数, e.g., 过去1小时)
    pred_seq_len: int = 20  # 预测序列长度 (输出帧数, e.g., 未来2小时)
    
    # =========================================================
    # 3. 模型架构参数 (Model Architecture - STMamba)
    # =========================================================
    hid_S: int = 128        # 空间编码器 (Spatial Encoder) 的隐藏层通道数
    hid_T: int = 512        # 时间演变模块 (Temporal Evolution) 的隐藏层维度
    N_S: int = 4            # 空间下采样/上采样层数 (决定了 Latent 的空间分辨率)
    N_T: int = 8            # 时间演变模块的堆叠层数 (Mamba Block 数量)
    
    # Mamba 核心参数
    mamba_d_state: int = 32 # SSM 状态维度 (State Space Dimension)
    mamba_d_conv: int = 4   # 局部卷积核大小 (Local Conv Kernel Size)
    mamba_expand: int = 2   # 扩展因子 (Expansion Factor, e.g., 2 * d_model)
    use_checkpoint: bool = True # 是否开启梯度检查点 (Gradient Checkpointing) 以节省显存
    
    # =========================================================
    # 4. 训练超参数 (Training Hyperparameters)
    # =========================================================
    batch_size: int = 4              # 单卡批次大小 (Batch Size per GPU)
    accumulate_grad_batches: int = 1 # 梯度累积步数 (用于模拟大 Batch)
    max_epochs: int = 100            # 最大训练轮数
    
    # 优化器配置
    opt: str = "adamw"           # 优化器类型 (adamw, sgd, etc.)
    lr: float = 1e-3             # 初始学习率 (Learning Rate)
    min_lr: float = 1e-5         # 最小学习率 (Cosine Annealing 下限)
    warmup_lr: float = 1e-5      # 预热起始学习率
    warmup_epoch: int = 20       # 预热轮数 (Warmup Epochs)
    weight_decay: float = 0.05   # 权重衰减 (L2 Regularization)
    momentum: float = 0.9        # 动量因子 (仅用于 SGD)
    filter_bias_and_bn: bool = True # 是否不对 Bias 和 BN 参数做 Weight Decay
    
    # 学习率调度器 (Scheduler)
    sched: str = "cosine"        # 调度策略 (cosine, step, etc.)
    decay_epoch: int = 30        # 衰减周期 (用于 Step Scheduler)
    decay_rate: float = 0.1      # 衰减率
    
    # =========================================================
    # 5. 损失函数配置 (Loss Functions Configuration)
    # =========================================================
    weight_bal_mse: float = 1.0   # 强度基准
    weight_dice: float = 1.0      # 抗坍塌核心
    weight_csi: float = 1.0       # 指标优化核心
    weight_gdl: float = 1.0       # 基础锐化核心
    
    # [可选 Loss]
    weight_msssim: float = 0.1    # 【推荐】结构维持，权重给小一点 (0.1)
    weight_facl: float = 0.01     # 【可选】频域纹理，权重极小 (0.01)
    weight_lpips: float = 0.0     # 【关闭】不推荐
    
    # =========================================================
    # 6. 高级训练策略 (Advanced Strategy)
    # =========================================================
    use_curriculum_learning: bool = True 
    use_temporal_weight: bool = True
    
    # --- Pydantic 配置 ---
    model_config = ConfigDict(protected_namespaces=())
    
    def to_dict(self):
        """将配置转换为字典格式，用于 HParams 记录"""
        return self.model_dump()