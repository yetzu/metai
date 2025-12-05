# metai/model/met_mamba/trainer.py

import torch
from typing import Tuple
from lightning.pytorch import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from metai.model.met_mamba.model import MeteoMamba
from metai.model.met_mamba.loss import HybridLoss
from metai.model.met_mamba.metrices import MetMetricCollection

class MeteoMambaModule(LightningModule):
    """
    MeteoMamba 训练模块。
    负责：Loss计算、优化器配置、指标记录、KL退火调度。
    """
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
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        
        # --- 优化参数 ---
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        max_epochs: int = 50,
        
        # --- Loss 参数 ---
        lambda_mae: float = 10.0,
        lambda_csi: float = 1.0,
        lambda_corr: float = 1.0,
        kl_weight_max: float = 0.01, # KL Loss 最大权重
        kl_anneal_epochs: int = 10,  # KL 退火周期
        
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
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            **kwargs
        )
        
        # 混合内容损失
        self.criterion_content = HybridLoss(
            lambda_mae=lambda_mae, 
            lambda_csi=lambda_csi, 
            lambda_corr=lambda_corr
        )
        
        # 指标收集
        self.train_metrics = MetMetricCollection(prefix="train_")
        self.val_metrics = MetMetricCollection(prefix="val_")

    def forward(self, x: torch.Tensor, y_target=None):
        """
        推理接口。
        训练时传入 y_target 以启用 CVAE Posterior;
        推理时 y_target=None，启用 CVAE Prior。
        """
        return self.model(x, y_target=y_target)

    def training_step(self, batch, batch_idx):
        meta, x, y, x_mask, y_mask = batch
        
        # 1. Forward (Train Mode: Posterior Guided)
        y_pred, flows, kl_loss = self.model(x, y_target=y)
        
        # 2. Content Loss
        content_loss, loss_dict = self.criterion_content(
            pred=y_pred, target=y, mask=y_mask, current_epoch=self.current_epoch
        )
        
        # 3. KL Annealing
        # 计算当前 KL 权重: 初期为 0，逐渐线性增加到 kl_weight_max
        anneal_ratio = min(1.0, self.current_epoch / self.hparams.kl_anneal_epochs)
        kl_weight = self.hparams.kl_weight_max * anneal_ratio
        
        # 4. Total Loss
        total_loss = content_loss + kl_weight * kl_loss
        
        # 5. Logging
        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        loss_dict['kl_loss'] = kl_loss
        loss_dict['kl_weight'] = kl_weight
        self.log_dict(loss_dict, prog_bar=False, on_step=False, on_epoch=True)
        
        self.train_metrics(y_pred.detach(), y, y_mask)
        return total_loss

    def validation_step(self, batch, batch_idx):
        meta, x, y, x_mask, y_mask = batch
        
        # 1. Inference (Val Mode: Prior Guided)
        # 验证时不传 y_target，测试模型真实的生成能力
        y_hat, flows = self.model(x, y_target=None)
        
        # 2. Loss & Metrics
        loss, _ = self.criterion_content(y_hat, y, mask=y_mask)
        
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.val_metrics(y_hat, y, y_mask)
        return loss

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        # 将 tensor 移动到设备以防万一
        metrics = {k: v.to(self.device) for k, v in metrics.items()}
        scalar_metrics = {k: v for k, v in metrics.items() if v.numel() == 1}
        
        val_score = scalar_metrics.get('val_total_score', None)
        if val_score is not None:
            self.log('val_score', val_score, prog_bar=True, logger=True, sync_dist=True)

        self.log_dict(scalar_metrics, prog_bar=False, logger=True, sync_dist=True)
        self.val_metrics.reset()

    def configure_optimizers(self):
        # 优化器配置
        params = self.model.parameters()
        opt = AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = CosineAnnealingLR(opt, T_max=self.hparams.max_epochs, eta_min=1e-6)
        
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "interval": "epoch",
                "frequency": 1
            }
        }