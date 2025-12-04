# metai/model/met_mamba/trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List, Union
from lightning.pytorch import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# 引入项目模块
from metai.model.met_mamba.model import MeteoMamba
from metai.model.met_mamba.loss import HybridLoss, GANLoss
from metai.model.met_mamba.metrices import MetMetricCollection
from metai.utils import MLOGI

class MeteoMambaModule(LightningModule):
    """
    [MeteoMamba Lightning Module v2.2 - Fix DDP Sync Error]
    
    修复内容：
    1. on_validation/test_epoch_end: 在 log(sync_dist=True) 之前将 CPU 指标移动回 GPU。
       这是因为 NCCL 后端不支持 CPU Tensor 的集合通信。
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
        
        # --- GAN 配置 ---
        use_gan: bool = True,
        gan_start_epoch: int = 0,
        gan_weight: float = 0.01,
        discriminator_lr: float = 2e-4,
        
        # --- Loss 配置 ---
        use_temporal_weight: bool = True,
        
        # --- 其他 kwargs ---
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
        
        self.criterion_content = HybridLoss(use_temporal_weight=use_temporal_weight)
        self.criterion_gan = GANLoss(mode='hinge')
        
        self.train_metrics = MetMetricCollection(prefix="train_")
        self.val_metrics = MetMetricCollection(prefix="val_")
        self.test_metrics = MetMetricCollection(prefix="test_")

        self.automatic_optimization = False

    def forward(self, x: torch.Tensor, current_epoch: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(x, current_epoch=current_epoch)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        meta, x, y, x_mask, y_mask = batch
        prev_frame = x[:, -1:, :self.model.out_channels, ...]
        do_gan = self.hparams.use_gan and (self.current_epoch >= self.hparams.gan_start_epoch)

        # --- Phase 1: Discriminator ---
        if do_gan:
            with torch.no_grad():
                y_fake_detached, _ = self.model(x, current_epoch=self.current_epoch)
            
            logits_real = self.model.discriminator(y)
            logits_fake = self.model.discriminator(y_fake_detached)
            loss_d = self.criterion_gan.get_disc_loss(logits_real, logits_fake)
            
            opt_d.zero_grad()
            self.manual_backward(loss_d)
            opt_d.step()
            self.log("train_loss_d", loss_d, prog_bar=True, on_step=True, on_epoch=True)

        # --- Phase 2: Generator ---
        y_fake, flows = self.model(x, current_epoch=self.current_epoch)
        
        loss_content, loss_dict = self.criterion_content(
            pred=y_fake, target=y, flow=flows, prev_frame=prev_frame, mask=y_mask
        )
        
        loss_adv = torch.tensor(0.0, device=self.device)
        current_gan_weight = 0.0
        if do_gan:
            logits_fake_g = self.model.discriminator(y_fake)
            loss_adv = self.criterion_gan.get_gen_loss(logits_fake_g)
            current_gan_weight = self.hparams.gan_weight
        
        total_loss_g = loss_content + current_gan_weight * loss_adv
        
        opt_g.zero_grad()
        self.manual_backward(total_loss_g)
        opt_g.step()
        
        self.log("train_loss_g", total_loss_g, prog_bar=True, on_step=True, on_epoch=True)
        if do_gan:
            self.log("train_loss_adv", loss_adv, prog_bar=False, on_step=False, on_epoch=True)
        
        self.log_dict(loss_dict, prog_bar=False, on_step=False, on_epoch=True)
        self.train_metrics(y_fake.detach(), y, y_mask)

    def validation_step(self, batch, batch_idx):
        meta, x, y, x_mask, y_mask = batch
        prev_frame = x[:, -1:, :self.model.out_channels, ...]
        y_hat, flows = self(x, current_epoch=self.current_epoch)
        loss, _ = self.criterion_content(y_hat, y, flow=flows, prev_frame=prev_frame, mask=y_mask)
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
        y_hat, _ = self(x, current_epoch=self.hparams.max_epochs)
        return y_hat

    def on_train_epoch_end(self):
        train_metrics = self.train_metrics.compute()
        scalar_metrics = {k: v for k, v in train_metrics.items() if v.numel() == 1}
        self.log_dict(scalar_metrics, prog_bar=False)
        self.train_metrics.reset()
        
        sch_g, sch_d = self.lr_schedulers()
        sch_g.step()
        sch_d.step()

    def on_validation_epoch_end(self):
        # 1. 计算指标 (此时 Tensor 在 CPU 上)
        metrics = self.val_metrics.compute()
        
        # 2. [关键修复] 将所有 Tensor 移动回 GPU (self.device)
        # NCCL 后端要求 sync_dist=True 时 Tensor 必须在 GPU 上
        metrics = {k: v.to(self.device) for k, v in metrics.items()}
        
        # 3. 分离标量 (Scalars) 和非标量 (Tensors)
        scalar_metrics = {}
        tensor_metrics = {}
        
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor) and v.numel() > 1:
                tensor_metrics[k] = v
            else:
                scalar_metrics[k] = v
        
        # 4. 记录关键指标 val_score
        val_score = scalar_metrics.get('val_total_score', None)
        if val_score is None and 'val_ts_levels' in tensor_metrics:
             val_score = tensor_metrics['val_ts_levels'].mean()
        
        if val_score is not None:
            self.log('val_score', val_score, prog_bar=True, logger=True, sync_dist=True)

        # 5. 记录标量
        self.log_dict(scalar_metrics, prog_bar=False, logger=True, sync_dist=True)
        
        # 6. 记录非标量的均值
        for k, v in tensor_metrics.items():
            self.log(f"{k}_mean", v.mean(), prog_bar=False, logger=True, sync_dist=True)
        
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        # [修复] 测试时同样需要移动到 GPU 以支持多卡测试
        metrics = {k: v.to(self.device) for k, v in metrics.items()}
        
        scalar_metrics = {k: v for k, v in metrics.items() if v.numel() == 1}
        self.log_dict(scalar_metrics, prog_bar=False, sync_dist=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        g_params = [p for n, p in self.model.named_parameters() if "discriminator" not in n]
        d_params = self.model.discriminator.parameters()
        opt_g = AdamW(g_params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        opt_d = AdamW(d_params, lr=self.hparams.discriminator_lr, weight_decay=self.hparams.weight_decay)
        sch_g = CosineAnnealingLR(opt_g, T_max=self.hparams.max_epochs, eta_min=1e-6)
        sch_d = CosineAnnealingLR(opt_d, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return [opt_g, opt_d], [sch_g, sch_d]