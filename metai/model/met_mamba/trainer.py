# metai/model/met_mamba/trainer.py

import lightning as l
import torch
from typing import Optional, Union, Any, Dict
import math

# [新增] 引入高斯模糊用于课程学习
from torchvision.transforms.functional import gaussian_blur

# 引入项目核心组件
from metai.model.core import get_optim_scheduler, timm_schedulers
from .model import MeteoMamba
from .loss import HybridLoss
from .metrices import MetScore
from .config import ModelConfig

class MeteoMambaModule(l.LightningModule):
    """
    MeteoMamba 训练与验证模块 (LightningModule)
    
    核心特性：
    1. 集成 STMamba 模型架构与混合损失函数。
    2. 实现双重课程学习 (Dual Curriculum Learning)：
       - 序列长度课程 (Sequence Length): 逐步增加预测时步，稳定自回归生成。
       - 模糊度课程 (Blurring): 初期模糊 Target，迫使模型优先学习宏观平流 (Advection)，后期细化纹理。
    3. 自动化的优化器与调度器配置。
    """
    
    def __init__(self, config: Optional[Union[ModelConfig, dict]] = None, **kwargs):
        super().__init__()
        
        # 1. 配置参数初始化
        # 支持传入 Config 对象或字典，增强调用灵活性
        if config is None:
            self.config = ModelConfig(**kwargs)
        elif isinstance(config, dict):
            config.update(kwargs)
            self.config = ModelConfig(**config)
        else:
            self.config = config
            # 覆盖 kwargs 中的参数
            for k, v in kwargs.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)

        # 保存超参数 (排除大数据路径，避免 Checkpoint 过大)
        hparams = self.config.to_dict()
        for key in ['batch_size', 'data_path']:
            hparams.pop(key, None)
        self.save_hyperparameters(hparams)
        
        # 2. 初始化模型架构
        self.model = MeteoMamba(
            in_shape=self.config.in_shape,
            in_seq_len=self.config.obs_seq_len,
            out_seq_len=self.config.pred_seq_len,
            hid_S=self.config.hid_S,
            hid_T=self.config.hid_T,
            N_S=self.config.N_S,
            N_T=self.config.N_T,
            mamba_d_state=self.config.mamba_d_state,
            mamba_d_conv=self.config.mamba_d_conv,
            mamba_expand=self.config.mamba_expand,
            use_checkpoint=self.config.use_checkpoint,
            mamba_sparse_ratio=self.config.mamba_sparse_ratio # 传递稀疏率
        )
        
        # 3. 初始化混合损失函数
        # HybridLoss 内部包含可学习参数，注册为子模块后会被自动优化
        self.criterion = HybridLoss(
            use_temporal_weight=getattr(self.config, 'use_temporal_weight', True)
        )
        
        # 4. 初始化评估指标
        self.valid_scorer = MetScore()

    def forward(self, x):
        """标准推理入口"""
        # 确保输出在 [0, 1] 范围内 (物理约束)
        return torch.clamp(self.model(x), 0.0, 1.0)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        配置优化器与学习率调度器。
        使用 metai.model.core 中的通用工具，支持 AdamW/SGD 等及 Cosine/Step 等调度策略。
        """
        # 注意: self.parameters() 会自动包含 self.criterion.parameters() (即自动权重的 s 参数)
        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.config, self.config.max_epochs, self
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch" if by_epoch else "step"
            }
        }

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        # set_to_none=True 通常比 zero_grad() 稍快
        optimizer.zero_grad(set_to_none=True)

    def lr_scheduler_step(self, scheduler: Any, metric: Any):
        # 兼容 timm 的调度器接口与 Lightning 的接口
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            scheduler.step(metric) if metric is not None else scheduler.step()

    def on_train_epoch_start(self):
        """Epoch 开始时的钩子：记录课程学习进度"""
        if self.config.use_curriculum_learning:
            progress = self.current_epoch / float(self.config.max_epochs)
            self.log('curriculum/progress', progress, on_step=False, on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        """
        单步训练逻辑。
        包含：数据解包 -> 课程策略应用 -> 前向传播 -> Loss计算 -> 日志记录
        """
        _, x, y, _, t_mask = batch
        # 确保 mask 类型正确
        if t_mask.dtype != torch.float32: t_mask = t_mask.float()
        
        # ====================================================
        # 课程学习策略 (Curriculum Learning Strategies)
        # ====================================================
        if self.config.use_curriculum_learning:
            
            # --- 策略 A: 序列长度课程 (Sequence Length Curriculum) ---
            # 目的: 在训练初期仅回传短序列的 Loss，防止长序列累积误差导致梯度爆炸或模式崩溃。
            total_epochs = float(self.config.max_epochs)
            progress = self.current_epoch / total_epochs
            
            start_ratio = 0.2  # 0-20% Epochs: 保持最小长度
            end_ratio = 0.7    # 70% Epochs: 达到最大长度
            min_seq = 10       # 起始预测长度
            max_seq = self.config.pred_seq_len
            
            if progress < start_ratio:
                valid_len = min_seq
            elif progress < end_ratio:
                # 线性增长阶段
                ramp_progress = (progress - start_ratio) / (end_ratio - start_ratio)
                valid_len = int(min_seq + (max_seq - min_seq) * ramp_progress)
            else:
                valid_len = max_seq
            
            valid_len = max(1, min(valid_len, max_seq))
            
            # 如果当前 Mask 长度超过课程长度，则截断 Mask (不计算后续时步的 Loss)
            if t_mask.shape[1] > valid_len:
                # 创建新 Mask 避免原地修改影响 Dataset
                time_mask_curriculum = torch.ones_like(t_mask)
                time_mask_curriculum[:, valid_len:, ...] = 0.0
                t_mask = t_mask * time_mask_curriculum
                
            self.log('curriculum/seq_len', float(valid_len), on_step=False, on_epoch=True)

            # --- 策略 B: 模糊度课程 (Blurring Curriculum) [新增] ---
            # 目的: 训练初期降低 Target 的高频信息，引导模型优先拟合大尺度的流体运动 (Advection)。
            # 随着训练进行，逐渐减小模糊半径，让模型学习微观纹理。
            max_sigma = 2.0  # 初始最大高斯核半径
            blur_epochs = 20 # 在前 20 个 Epoch 内应用模糊策略
            
            if self.current_epoch < blur_epochs:
                # 线性衰减 sigma: max_sigma -> 0.0
                sigma = max_sigma * (1 - self.current_epoch / blur_epochs)
                
                # 仅当 sigma 足够大时执行模糊，节省计算资源
                if sigma > 0.1:
                    # 计算 Kernel Size: 必须为奇数，通常取 6*sigma + 1 覆盖绝大部分能量
                    k_size = int(2 * int(3 * sigma) + 1)
                    if k_size % 2 == 0: k_size += 1
                    
                    # 对 Ground Truth (y) 应用高斯模糊
                    y = gaussian_blur(y, kernel_size=k_size, sigma=sigma)
                    
                    self.log('curriculum/blur_sigma', sigma, on_step=False, on_epoch=True)

        # ====================================================
        # 前向与反向传播
        # ====================================================
        
        # 1. 前向传播
        pred = self(x) # 内部已包含 clamp(0,1)
        
        # 2. 计算损失
        # 传入 current_epoch 供 Loss 内部可能的动态调整使用
        loss, loss_dict = self.criterion(
            pred, y, mask=t_mask,
            current_epoch=self.current_epoch,
            total_epochs=self.config.max_epochs
        )
        
        # 3. 日志记录
        bs = x.size(0)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        
        # 记录详细分项 Loss 和学习到的权重
        for k, v in loss_dict.items():
            if k != 'total':
                self.log(f'loss_metrics/{k}', v, on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
                
        return loss

    def validation_step(self, batch, batch_idx):
        """
        验证逻辑。
        注意：验证阶段不应用任何课程学习策略（无 Blur，无 Mask 截断），
        必须使用最原始、最完整的 Ground Truth 评估模型真实性能。
        """
        _, x, y, _, t_mask = batch
        if t_mask.dtype != torch.float32: t_mask = t_mask.float()
        
        pred = self(x)
        
        # 计算验证 Loss
        loss, _ = self.criterion(
            pred, y, mask=t_mask,
            current_epoch=self.config.max_epochs, 
            total_epochs=self.config.max_epochs
        )
        
        # 计算气象评估指标 (CSI, HSS, etc.)
        metric_results = self.valid_scorer(pred, y, mask=t_mask)
        
        bs = x.size(0)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        self.log('val_score', metric_results['total_score'], on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        
        # (可选) 记录具体的 CSI 指标，便于分析不同阈值的表现
        if 'csi_30' in metric_results: # 假设 metrics 返回了 key 为 csi_30 的指标
             self.log('val_csi_30', metric_results['csi_30'], on_epoch=True, batch_size=bs, sync_dist=True)

    def test_step(self, batch, batch_idx):
        """测试逻辑：仅返回预测结果供后续可视化或评估"""
        _, x, y, _, t_mask = batch
        pred = self(x)
        return {'inputs': x, 'trues': y, 'preds': pred}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """生产环境预测接口"""
        _, x, _ = batch
        return self(x)