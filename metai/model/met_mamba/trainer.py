# metai/model/met_mamba/trainer.py

import torch
from typing import Tuple, Dict, Any
from lightning.pytorch import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from metai.model.met_mamba.model import MeteoMamba
from metai.model.met_mamba.loss import HybridLoss
from metai.model.met_mamba.metrices import MetMetricCollection

class MeteoMambaModule(LightningModule):
    def __init__(
        self, 
        # --- 模型架构参数 ---
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
        
        # --- 优化参数 ---
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        max_epochs: int = 50,
        
        # --- [改进] Loss 权重参数化 ---
        lambda_mae: float = 10.0,
        lambda_csi: float = 1.0,
        lambda_corr: float = 1.0,
        
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
        
        # 使用参数初始化的 Loss
        self.criterion_content = HybridLoss(
            lambda_mae=lambda_mae, 
            lambda_csi=lambda_csi, 
            lambda_corr=lambda_corr
        )
        
        self.train_metrics = MetMetricCollection(prefix="train_")
        self.val_metrics = MetMetricCollection(prefix="val_")
        self.test_metrics = MetMetricCollection(prefix="test_")

    def forward(self, x: torch.Tensor, current_epoch: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(x, current_epoch=current_epoch)

    def training_step(self, batch, batch_idx):
        meta, x, y, x_mask, y_mask = batch
        y_pred, flows = self.model(x, current_epoch=self.current_epoch)
        
        total_loss, loss_dict = self.criterion_content(
            pred=y_pred, target=y, mask=y_mask, current_epoch=self.current_epoch
        )
        
        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log_dict(loss_dict, prog_bar=False, on_step=False, on_epoch=True)
        self.train_metrics(y_pred.detach(), y, y_mask)
        return total_loss

    def validation_step(self, batch, batch_idx):
        meta, x, y, x_mask, y_mask = batch
        y_hat, flows = self(x, current_epoch=self.current_epoch)
        loss, _ = self.criterion_content(y_hat, y, mask=y_mask)
        
        # 记录验证 Loss (自动同步)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        
        # 计算并累积本地指标
        self.val_metrics(y_hat, y, y_mask)
        return loss

    def on_validation_epoch_end(self):
        # 计算本地聚合指标
        metrics = self.val_metrics.compute()
        
        # 将 Tensor 移动到设备以防万一
        metrics = {k: v.to(self.device) for k, v in metrics.items()}
        
        scalar_metrics = {k: v for k, v in metrics.items() if v.numel() == 1}
        
        # 获取主要监控指标
        val_score = scalar_metrics.get('val_total_score', None)
        
        # [注意] 这里依然使用 sync_dist=True 进行平均。
        # 如果需要绝对精确的 DDP 全局 TS 评分，建议后续引入 torchmetrics 替换 MetMetricCollection
        if val_score is not None:
            self.log('val_score', val_score, prog_bar=True, logger=True, sync_dist=True)

        self.log_dict(scalar_metrics, prog_bar=False, logger=True, sync_dist=True)
        self.val_metrics.reset()

    def configure_optimizers(self):
        params = self.model.parameters()
        opt = AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = CosineAnnealingLR(opt, T_max=self.hparams.max_epochs, eta_min=1e-6)
        
        # [改进] 显式配置 Scheduler
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "interval": "epoch",
                "frequency": 1
            }
        }