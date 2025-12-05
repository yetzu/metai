# metai/model/met_mamba/trainer.py

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any
from lightning.pytorch import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# 引入项目模块
from metai.model.met_mamba.model import MeteoMamba
from metai.model.met_mamba.loss import HybridLoss  # 仅引用新的 HybridLoss
from metai.model.met_mamba.metrices import MetMetricCollection

class MeteoMambaModule(LightningModule):
    """
    MeteoMamba Lightning Module (No-GAN Version)
    专注于纯监督学习，优化回归与评分指标。
    """
    
    def __init__(
        self, 
        # --- 模型架构超参 ---
        in_shape: Tuple[int, int, int] = (19, 256, 256),
        in_seq_len: int = 10,
        out_seq_len: int = 20,
        out_channels: int = 1,
        hid_S: int = 64,
        hid_T: int = 256,
        N_S: int = 4,
        N_T: int = 8,
        mamba_sparse_ratio: float = 0.5,
        anneal_start_epoch: int = 5,
        anneal_end_epoch: int = 15, 
        
        # --- 优化器超参 ---
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        max_epochs: int = 50,
        
        # --- Loss 配置 ---
        use_temporal_weight: bool = True,
        
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = MeteoMamba(
            in_shape=in_shape,
            in_seq_len=in_seq_len,
            out_seq_len=out_seq_len,
            out_channels=out_channels,
            hid_S=hid_S,
            hid_T=hid_T,
            N_S=N_S,
            N_T=N_T,
            mamba_sparse_ratio=mamba_sparse_ratio,
            anneal_start_epoch=anneal_start_epoch,
            anneal_end_epoch=anneal_end_epoch,
            **kwargs
        )
        
        # 初始化新的 HybridLoss
        # 建议参数: MAE权重加大以平衡数值量级
        self.criterion_content = HybridLoss(
            lambda_mae=10.0, 
            lambda_csi=1.0, 
            lambda_corr=1.0
        )
        
        self.train_metrics = MetMetricCollection(prefix="train_")
        self.val_metrics = MetMetricCollection(prefix="val_")
        self.test_metrics = MetMetricCollection(prefix="test_")

        self.automatic_optimization = False

    def forward(self, x: torch.Tensor, current_epoch: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(x, current_epoch=current_epoch)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        meta, x, y, x_mask, y_mask = batch
        
        # 前向传播
        y_pred, flows = self.model(x, current_epoch=self.current_epoch)
        
        # 计算 Loss (自动处理 clamp 和 mask)
        total_loss, loss_dict = self.criterion_content(
            pred=y_pred, target=y, mask=y_mask, current_epoch=self.current_epoch
        )
        
        # 反向传播与优化
        opt.zero_grad()
        self.manual_backward(total_loss)
        
        # 梯度裁剪 (防止梯度爆炸)
        self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        
        opt.step()
        
        # 日志记录
        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log_dict(loss_dict, prog_bar=False, on_step=False, on_epoch=True)
        self.train_metrics(y_pred.detach(), y, y_mask)

    def validation_step(self, batch, batch_idx):
        meta, x, y, x_mask, y_mask = batch
        y_hat, flows = self(x, current_epoch=self.current_epoch)
        
        # 验证集仅计算 Loss 用于监控
        loss, _ = self.criterion_content(y_hat, y, mask=y_mask)
        
        self.val_metrics(y_hat, y, y_mask)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        meta, x, y, x_mask, y_mask = batch
        y_hat, flows = self(x, current_epoch=self.current_epoch)
        loss, _ = self.criterion_content(y_hat, y, mask=y_mask)
        self.test_metrics(y_hat, y, y_mask)
        self.log('test_loss', loss, on_epoch=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if len(batch) >= 2: x = batch[1]
        else: x = batch[0]
        # 推理时使用最终 Epoch 的逻辑 (如稀疏度全开)
        y_hat, _ = self(x, current_epoch=self.hparams.max_epochs)
        return torch.clamp(y_hat, 0.0, 1.0)

    def on_train_epoch_end(self):
        train_metrics = self.train_metrics.compute()
        scalar_metrics = {k: v for k, v in train_metrics.items() if v.numel() == 1}
        self.log_dict(scalar_metrics, prog_bar=False)
        self.train_metrics.reset()
        
        sch = self.lr_schedulers()
        sch.step()

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        metrics = {k: v.to(self.device) for k, v in metrics.items()}
        
        scalar_metrics = {k: v for k, v in metrics.items() if v.numel() == 1}
        tensor_metrics = {k: v for k, v in metrics.items() if v.numel() > 1}
        
        # 使用 TS 均值作为 val_score 监控指标
        val_score = scalar_metrics.get('val_total_score', None)
        if val_score is None and 'val_ts_levels' in tensor_metrics:
             val_score = tensor_metrics['val_ts_levels'].mean()
        
        if val_score is not None:
            self.log('val_score', val_score, prog_bar=True, logger=True, sync_dist=True)

        self.log_dict(scalar_metrics, prog_bar=False, logger=True, sync_dist=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        metrics = {k: v.to(self.device) for k, v in metrics.items()}
        scalar_metrics = {k: v for k, v in metrics.items() if v.numel() == 1}
        self.log_dict(scalar_metrics, prog_bar=False, sync_dist=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        # 仅返回生成器优化器
        params = self.model.parameters()
        opt = AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = CosineAnnealingLR(opt, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return [opt], [sch]