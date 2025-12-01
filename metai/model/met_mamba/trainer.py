# metai/model/met_mamba/trainer.py
import lightning as l
import torch
from typing import Optional, Union, Any, Dict

from metai.model.core import get_optim_scheduler, timm_schedulers
from .model import MeteoMamba
from .loss import HybridLoss
from .metrices import MetScore
from .config import ModelConfig

class MeteoMambaModule(l.LightningModule):
    """
    MeteoMamba 训练模块
    集成模型架构、混合损失函数、优化器配置及训练流程控制。
    """
    def __init__(self, config: Optional[Union[ModelConfig, dict]] = None, **kwargs):
        super().__init__()
        
        # 1. 配置初始化
        # 支持传入 ModelConfig 对象或字典，优先使用传入参数覆盖默认配置
        if config is None:
            self.config = ModelConfig(**kwargs)
        elif isinstance(config, dict):
            config.update(kwargs)
            self.config = ModelConfig(**config)
        else:
            self.config = config
            for k, v in kwargs.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)

        # 保存超参数 (用于 Checkpoint 恢复及日志记录)
        hparams = self.config.to_dict()
        for key in ['batch_size', 'data_path']:
            hparams.pop(key, None)
        self.save_hyperparameters(hparams)
        
        # 2. 初始化模型架构 (STMamba)
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
            use_checkpoint=self.config.use_checkpoint
        )
        
        # 3. 初始化混合损失函数 (HybridLoss)
        # 包含：平衡MSE、频域损失、梯度损失、软CSI指标及结构相似性
        # 使用 getattr 确保配置兼容性
        self.criterion = HybridLoss(
            weight_bal_mse=getattr(self.config, 'weight_bal_mse', 1.0),
            weight_facl=getattr(self.config, 'weight_facl', 0.05),
            weight_gdl=getattr(self.config, 'weight_gdl', 0.5),
            weight_csi=getattr(self.config, 'weight_csi', 0.5),
            weight_msssim=getattr(self.config, 'weight_msssim', 0.5),
            weight_lpips=getattr(self.config, 'weight_lpips', 0.0),
            use_curriculum=getattr(self.config, 'use_curriculum_learning', True),
            use_temporal_weight=getattr(self.config, 'use_temporal_weight', True)
        )
        
        # 验证集评估指标计算器
        self.valid_scorer = MetScore()

    def forward(self, x):
        """标准推理入口"""
        return self.model(x)

    def configure_optimizers(self) -> Dict[str, Any]:
        """配置优化器与学习率调度器"""
        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.config, self.config.max_epochs, self.model
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch" if by_epoch else "step"
            }
        }

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        """梯度置零优化"""
        optimizer.zero_grad(set_to_none=True)

    def lr_scheduler_step(self, scheduler: Any, metric: Any):
        """调度器步进逻辑 (兼容 timm)"""
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            scheduler.step(metric) if metric is not None else scheduler.step()

    def on_train_epoch_start(self):
        """Epoch 开始时的钩子：记录训练进度"""
        if self.config.use_curriculum_learning:
            progress = self.current_epoch / float(self.config.max_epochs)
            self.log('epoch_progress', progress, on_step=False, on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        _, x, y, _, t_mask = batch
        # 确保 mask 类型正确
        if t_mask.dtype != torch.float32: t_mask = t_mask.float()
        
        # ====================================================
        # 线性课程学习机制 (Linear Curriculum Learning)
        # ====================================================
        # 策略:
        # 1. 热身期 (0% ~ 10%): 固定训练 10 帧，稳固基础。
        # 2. 增长期 (10% ~ 50%): 序列长度随 Epoch 线性从 10 增加到 20。
        # 3. 稳定期 (50% ~ 100%): 训练全长 20 帧，攻克长时效。
        # ====================================================
        if self.config.use_curriculum_learning:
            total_epochs = float(self.config.max_epochs)
            current_epoch = self.current_epoch
            progress = current_epoch / total_epochs
            
            # 定义阶段参数
            start_ratio = 0.1  # 热身结束点
            end_ratio = 0.5    # 增长结束点
            min_seq = 10       # 最小序列长度
            max_seq = self.config.pred_seq_len # 最大序列长度 (通常为20)
            
            if progress < start_ratio:
                # 阶段 1: 热身
                valid_len = min_seq
            elif progress < end_ratio:
                # 阶段 2: 线性增长
                # 归一化进度 (0.0 -> 1.0)
                ramp_progress = (progress - start_ratio) / (end_ratio - start_ratio)
                # 线性插值
                valid_len = int(min_seq + (max_seq - min_seq) * ramp_progress)
            else:
                # 阶段 3: 全长
                valid_len = max_seq
            
            # 边界保护
            valid_len = max(1, min(valid_len, max_seq))
            
            # 应用 Mask (若当前 valid_len 小于数据长度，则遮蔽后续帧)
            if t_mask.shape[1] > valid_len:
                # 创建新 Mask 避免原地修改
                time_mask = torch.ones_like(t_mask)
                time_mask[:, valid_len:, ...] = 0.0
                t_mask = t_mask * time_mask
                
                # 记录当前训练的序列长度，方便监控课程进度
                self.log('curriculum/seq_len', float(valid_len), on_step=False, on_epoch=True)

        # 模型前向传播 (约束输出范围 [0, 1])
        pred = torch.clamp(self.model(x), 0.0, 1.0)
        
        # 计算损失
        # 传入当前 Epoch 信息，触发 Loss 内部的纹理渐进策略
        loss, loss_dict = self.criterion(
            pred, y, mask=t_mask,
            current_epoch=self.current_epoch,
            total_epochs=self.config.max_epochs
        )
        
        bs = x.size(0)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        
        # 记录各分量损失
        for k, v in loss_dict.items():
            if k != 'total':
                self.log(f'loss_{k}', v, on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
                
        return loss

    def validation_step(self, batch, batch_idx):
        _, x, y, _, t_mask = batch
        if t_mask.dtype != torch.float32: t_mask = t_mask.float()
        
        pred = torch.clamp(self.model(x), 0.0, 1.0)
        
        # 验证阶段使用全量权重和全长序列进行评估，反映最终性能
        loss, _ = self.criterion(
            pred, y, mask=t_mask,
            current_epoch=self.config.max_epochs, 
            total_epochs=self.config.max_epochs
        )
        
        metric_results = self.valid_scorer(pred, y, mask=t_mask)
        
        bs = x.size(0)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        self.log('val_score', metric_results['total_score'], on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)

    def test_step(self, batch, batch_idx):
        _, x, y, _, t_mask = batch
        pred = torch.clamp(self.model(x), 0.0, 1.0)
        return {'inputs': x, 'trues': y, 'preds': pred}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        _, x, _ = batch
        return torch.clamp(self.model(x), 0.0, 1.0)