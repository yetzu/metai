# metai/model/simvp/simvp_config.py

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Tuple, Literal, Union, List

class SimVPConfig(BaseModel):
    """
    SimVP 模型配置类 (SOTA 优化版)
    适配 10帧 -> 20帧 预测任务，以及 30 通道输入 (含周期性 GIS 编码)。
    """

    # 1. 基础配置
    model_name: str = Field(default="scwds_simvp", description="模型名称")
    data_path: str = Field(default="data/samples.jsonl", description="数据索引文件路径 (.jsonl)")
    save_dir: str = Field(default="./output/simvp", description="训练输出目录")
    
    # [关键修改] 输入形状调整
    # T=10: 输入 10 帧 (过去1小时)
    # C=30: 28 (原通道) + 2 (GIS 时间周期编码增量) = 30
    in_shape: Tuple[int, int, int, int] = Field(default=(10, 54, 256, 256), description="输入形状 (T, C, H, W)")

    # [关键新增] 输出序列长度
    # T_out=20: 预测 20 帧 (未来2小时)
    aft_seq_length: int = Field(default=20, description="输出序列长度 (预测帧数)")

    @field_validator('in_shape')
    @classmethod
    def validate_in_shape(cls, v) -> Tuple[int, int, int, int]:
        if len(v) != 4 or any(x <= 0 for x in v):
            raise ValueError(f"in_shape 必须是 4 个正数元素的元组 (T, C, H, W)，当前为 {v}")
        return (int(v[0]), int(v[1]), int(v[2]), int(v[3]))

    # [优化] 增加训练轮数以适配课程学习 Phase 3
    max_epochs: int = Field(default=150, description="最大训练轮数")

    # 2. 数据加载器配置
    batch_size: int = Field(default=4, description="批大小 (单卡)")
    seed: int = Field(default=42, description="全局随机种子")
    num_workers: int = Field(default=4, description="DataLoader 工作线程数")
    train_split: float = Field(default=0.8, description="训练集比例")
    val_split: float = Field(default=0.1, description="验证集比例")
    test_split: float = Field(default=0.1, description="测试集比例")
    task_mode: str = Field(default='precipitation', description="任务模式")

    # 3. Trainer 配置
    precision: Literal["16-mixed", "32", "64", "16-true", "bf16-mixed", "bf16-true", "32-true"] = Field(default="16-mixed", description="训练精度")
    accelerator: Literal["auto", "cpu", "cuda"] = Field(default="auto", description="加速器类型")
    devices: Union[int, str, List[int]] = Field(default="auto", description="设备编号")
    log_every_n_steps: int = Field(default=10, description="日志记录频率")
    val_check_interval: float = Field(default=1.0, description="验证频率")
    gradient_clip_val: float = Field(default=1.0, description="梯度裁剪阈值")
    gradient_clip_algorithm: Literal["norm", "value"] = Field(default="norm", description="梯度裁剪算法")
    deterministic: bool = Field(default=True, description="是否使用确定性算法")
    enable_progress_bar: bool = Field(default=True, description="显示进度条")
    enable_model_summary: bool = Field(default=True, description="显示模型摘要")
    accumulate_grad_batches: int = Field(default=1, description="梯度累积步数")
    detect_anomaly: bool = Field(default=False, description="PyTorch 异常检测")
    profiler: Literal["simple", "advanced", None] = Field(default=None, description="性能分析器")
    limit_train_batches: Union[int, float] = Field(default=1.0, description="限制训练数据量")
    limit_val_batches: Union[int, float] = Field(default=1.0, description="限制验证数据量")
    num_sanity_val_steps: int = Field(default=2, description="训练前健全性检查步数")

    # 4. SimVP 模型结构参数 (SOTA 容量)
    hid_S: int = Field(default=128, description="空间编码器隐藏层通道数")
    hid_T: int = Field(default=512, description="时序转换器隐藏层通道数")
    N_S: int = Field(default=4, description="空间编码器层数")
    N_T: int = Field(default=12, description="时序转换器层数") # 增加层数以捕捉长时依赖
    model_type: str = Field(default='mamba', description="时序模块类型")
    mlp_ratio: float = Field(default=8.0, description="MLP 扩展比例")
    drop: float = Field(default=0.0, description="Dropout 比率")
    drop_path: float = Field(default=0.1, description="Drop Path 比率")
    spatio_kernel_enc: int = Field(default=3, description="编码器卷积核大小")
    spatio_kernel_dec: int = Field(default=3, description="解码器卷积核大小")
    out_channels: int = Field(default=1, description="输出通道数")

    # 5. 损失函数配置
    loss_weight_l1: float = Field(default=1.0, description="L1 Loss 权重")
    loss_weight_ssim: float = Field(default=0.5, description="MS-SSIM 权重")
    loss_weight_csi: float = Field(default=1.0, description="Soft-CSI 权重")
    loss_weight_spectral: float = Field(default=0.1, description="Spectral 权重")
    loss_weight_evo: float = Field(default=0.5, description="Evolution 权重")
    
    # 6. 课程学习配置
    use_curriculum_learning: bool = Field(default=True, description="是否启用课程学习")

    # 7. 早停 (Early Stopping)
    early_stop_monitor: str = Field(default="val_score", description="监控指标")
    early_stop_mode: str = Field(default="max", description="早停模式")
    early_stop_min_delta: float = Field(default=1e-4, description="最小改善阈值")
    early_stop_patience: int = Field(default=50, description="容忍 Epoch 数 (加长以适应 Phase 3)")

    # 8. 优化器与调度器
    opt: str = Field(default="adamw", description="优化器")
    lr: float = Field(default=1e-3, description="初始学习率")
    weight_decay: float = Field(default=1e-2, description="权重衰减")
    filter_bias_and_bn: bool = Field(default=True, description="是否对 Bias 和 BN 层免除权重衰减")
    momentum: float = Field(default=0.9, description="SGD 动量")
    sched: str = Field(default="cosine", description="调度器")
    min_lr: float = Field(default=1e-5, description="最小学习率 (提高以支持 Phase 3)")
    warmup_lr: float = Field(default=1e-5, description="Warmup 初始学习率")
    warmup_epoch: int = Field(default=5, description="Warmup Epoch 数")
    decay_epoch: int = Field(default=30, description="Step Decay 的间隔")
    decay_rate: float = Field(default=0.1, description="Step Decay 的衰减率")

    @property
    def pre_seq_length(self) -> int:
        return self.in_shape[0]

    @property
    def channels(self) -> int:
        return self.in_shape[1]

    @property
    def resize_shape(self) -> Tuple[int, int]:
        return (self.in_shape[2], self.in_shape[3])

    def to_dict(self) -> dict:
        data = self.model_dump()
        data['pre_seq_length'] = self.pre_seq_length
        # aft_seq_length 已在 model_dump 中
        data['channels'] = self.channels
        data['resize_shape'] = self.resize_shape
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SimVPConfig':
        return cls(**data)

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())