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
    集成模型架构、智能混合损失函数、优化器配置及训练流程控制。
    """
    def __init__(self, config: Optional[Union[ModelConfig, dict]] = None, **kwargs):
        super().__init__()
        
        # 1. 配置初始化
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

        # 保存超参数
        hparams = self.config.to_dict()
        for key in ['batch_size', 'data_path']:
            hparams.pop(key, None)
        self.save_hyperparameters(hparams)
        
        # 2. 初始化模型架构 (STMamba + Advective Projection)
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
        
        # 3. 初始化混合损失函数 (HybridLoss - Auto Weighted)
        # 不再需要传入具体的 loss 权重，只需配置是否启用时间加权
        self.criterion = HybridLoss(
            use_temporal_weight=getattr(self.config, 'use_temporal_weight', True)
        )
        
        # 验证集评估指标计算器
        self.valid_scorer = MetScore()

    def forward(self, x):
        """标准推理入口"""
        return self.model(x)

    def configure_optimizers(self) -> Dict[str, Any]:
        """配置优化器与学习率调度器"""
        # 注意：self.criterion.parameters() (即自动权重的 s 参数) 
        # 会自动包含在 self.parameters() 中，因为 HybridLoss 是本模块的子模块
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
        optimizer.zero_grad(set_to_none=True)

    def lr_scheduler_step(self, scheduler: Any, metric: Any):
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            scheduler.step(metric) if metric is not None else scheduler.step()

    def on_train_epoch_start(self):
        """Epoch 开始时的钩子"""
        if self.config.use_curriculum_learning:
            progress = self.current_epoch / float(self.config.max_epochs)
            self.log('curriculum/progress', progress, on_step=False, on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        _, x, y, _, t_mask = batch
        if t_mask.dtype != torch.float32: t_mask = t_mask.float()
        
        # ====================================================
        # 序列长度课程学习 (Sequence Length Curriculum)
        # 这一步对于稳定 AdvectiveProjection (光流Warp) 至关重要
        # ====================================================
        if self.config.use_curriculum_learning:
            total_epochs = float(self.config.max_epochs)
            progress = self.current_epoch / total_epochs
            
            start_ratio = 0.1  # 热身期
            end_ratio = 0.5    # 增长期结束
            min_seq = 10       # 起始长度
            max_seq = self.config.pred_seq_len
            
            if progress < start_ratio:
                valid_len = min_seq
            elif progress < end_ratio:
                ramp_progress = (progress - start_ratio) / (end_ratio - start_ratio)
                valid_len = int(min_seq + (max_seq - min_seq) * ramp_progress)
            else:
                valid_len = max_seq
            
            valid_len = max(1, min(valid_len, max_seq))
            
            # 动态调整 Mask：遮蔽超出当前课程长度的时间步
            if t_mask.shape[1] > valid_len:
                # 必须 clone 或 new tensor，避免修改原数据影响后续流程
                time_mask = torch.ones_like(t_mask)
                time_mask[:, valid_len:, ...] = 0.0
                t_mask = t_mask * time_mask
                
            self.log('curriculum/seq_len', float(valid_len), on_step=False, on_epoch=True)

        # 模型前向传播
        pred = torch.clamp(self.model(x), 0.0, 1.0)
        
        # 计算自动加权损失
        loss, loss_dict = self.criterion(
            pred, y, mask=t_mask,
            current_epoch=self.current_epoch,
            total_epochs=self.config.max_epochs
        )
        
        bs = x.size(0)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        
        # 记录详细 Loss 分量和自动学习到的权重
        for k, v in loss_dict.items():
            if k != 'total':
                self.log(f'loss_metrics/{k}', v, on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
                
        return loss

    def validation_step(self, batch, batch_idx):
        _, x, y, _, t_mask = batch
        if t_mask.dtype != torch.float32: t_mask = t_mask.float()
        
        pred = torch.clamp(self.model(x), 0.0, 1.0)
        
        # 验证阶段始终使用全长序列
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