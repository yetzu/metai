# metai/model/st_mamba/trainer.py

import lightning as l
import torch
import torch.nn.functional as F
from typing import Any
from .model import MeteoMamba
from .loss import HybridLoss
from metai.model.core import get_optim_scheduler, timm_schedulers

class MeteoMambaModule(l.LightningModule):
    def __init__(
        self,
        # --- 模型参数 ---
        in_shape: tuple = (10, 31, 256, 256),
        hid_S: int = 64,
        hid_T: int = 256,
        N_S: int = 4,
        N_T: int = 8,
        aft_seq_length: int = 20,
        
        # --- 训练与优化参数 ---
        max_epochs: int = 50,
        opt: str = "adamw",
        lr: float = 1e-3,
        min_lr: float = 1e-5,
        warmup_lr: float = 1e-5,
        warmup_epoch: int = 5,
        weight_decay: float = 0.05,
        filter_bias_and_bn: bool = True,  # 显式定义
        momentum: float = 0.9,
        sched: str = "cosine",
        decay_epoch: int = 30,
        decay_rate: float = 0.1,
        
        # --- 损失函数权重 ---
        loss_weight_focal: float = 0.0,
        loss_weight_evo: float = 0.5,
        loss_weight_ssim: float = 0.5,
        loss_weight_csi: float = 1.0,
        
        # --- 策略 ---
        use_curriculum_learning: bool = True,
        
        # --- 其他 ---
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = MeteoMamba(
            in_shape=in_shape,
            hid_S=hid_S,
            hid_T=hid_T,
            N_S=N_S,
            N_T=N_T,
            aft_seq_length=aft_seq_length,
            out_channels=1
        )
        
        self.criterion = HybridLoss(
            focal_weight=loss_weight_focal,
            evo_weight=loss_weight_evo,
            ssim_weight=loss_weight_ssim,
            csi_weight=loss_weight_csi
        )
        
        self.resize_shape = (in_shape[2], in_shape[3])
        self.MM_MAX = 30.0
        self.val_thresholds = [0.1, 1.0, 2.0, 5.0, 8.0] 
        weights_raw = [0.1, 0.1, 0.2, 0.25, 0.35]
        total_w = sum(weights_raw)
        self.val_weights = [w / total_w for w in weights_raw]

    def configure_optimizers(self):
        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.hparams, self.hparams.max_epochs, self.model
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch" if by_epoch else "step"
            }
        }

    # [关键修复] 必须重写此方法以兼容 timm 的 CosineLRScheduler
    def lr_scheduler_step(self, scheduler: Any, metric: Any):
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            # timm 调度器需要传入 epoch 参数
            scheduler.step(epoch=self.current_epoch)
        else:
            # PyTorch 原生调度器
            scheduler.step(metric) if metric is not None else scheduler.step()

    def on_train_epoch_start(self):
        if not self.hparams.use_curriculum_learning: return
        epoch = self.current_epoch
        max_epochs = self.hparams.max_epochs
        phase_1_end = int(0.2 * max_epochs)
        phase_2_end = int(0.6 * max_epochs)
        weights = {}
        
        if epoch < phase_1_end:
            weights = {'focal': 10.0, 'evo': 0.0, 'ssim': 0.0, 'csi': 0.0}
        elif epoch < phase_2_end:
            p = (epoch - phase_1_end) / (phase_2_end - phase_1_end)
            weights = {'focal': 10.0 - p * 5.0, 'evo': p * 1.0, 'ssim': p * 0.5, 'csi': 0.0}
        else:
            p = (epoch - phase_2_end) / (max_epochs - phase_2_end)
            weights = {'focal': 5.0 - p * 4.0, 'evo': 1.0 - p * 0.5, 'ssim': 0.5, 'csi': p * 5.0}

        self.criterion.weights.update(weights)
        for k, v in weights.items():
            self.log(f"train/w_{k}", v, on_epoch=True, sync_dist=True)

    def _interpolate_batch(self, x, mode='bilinear'):
        if self.resize_shape is None: return x
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        if mode == 'nearest':
            x = F.interpolate(x, size=self.resize_shape, mode='nearest')
        else:
            x = F.interpolate(x, size=self.resize_shape, mode='bilinear', align_corners=False)
        return x.view(B, T, C, *self.resize_shape)

    def training_step(self, batch, batch_idx):
        _, x, y, t_mask, _ = batch
        x = self._interpolate_batch(x, mode='bilinear')
        y = self._interpolate_batch(y, mode='bilinear')
        t_mask = self._interpolate_batch(t_mask.float(), mode='nearest')
        logits = self.model(x)
        loss, loss_dict = self.criterion(logits, y, mask=t_mask)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, x, y, t_mask, _ = batch
        x = self._interpolate_batch(x, mode='bilinear')
        y = self._interpolate_batch(y, mode='bilinear')
        t_mask = self._interpolate_batch(t_mask.float(), mode='nearest')
        logits = self.model(x)
        loss, _ = self.criterion(logits, y, mask=t_mask)
        pred = torch.clamp(torch.sigmoid(logits), 0.0, 1.0)
        
        if t_mask.sum() > 0:
            mae = (torch.abs(pred - y) * t_mask).sum() / (t_mask.sum() + 1e-6)
        else: mae = 0.0
            
        pred_mm = pred * self.MM_MAX
        target_mm = y * self.MM_MAX
        weighted_csi_sum = 0.0
        valid_mask = t_mask > 0.5
        
        for i, threshold in enumerate(self.val_thresholds):
            hits = ((pred_mm >= threshold) & (target_mm >= threshold) & valid_mask).float().sum()
            union = (((pred_mm >= threshold) | (target_mm >= threshold)) & valid_mask).float().sum()
            csi = hits / (union + 1e-6)
            weighted_csi_sum += csi * self.val_weights[i]

        self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        self.log('val_mae', mae, on_epoch=True, sync_dist=True)
        self.log('val_score', weighted_csi_sum, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        _, x, y, t_mask, _ = batch
        x = self._interpolate_batch(x, mode='bilinear')
        y = self._interpolate_batch(y, mode='bilinear')
        # t_mask = self._interpolate_batch(t_mask.float(), mode='nearest')

        logits = self.model(x)
        pred = torch.clamp(torch.sigmoid(logits), 0.0, 1.0)
        
        return {
            'inputs': x,
            'trues': y,
            'preds': pred
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        _, x, _ = batch
        x = self._interpolate_batch(x, mode='bilinear')
        return torch.clamp(torch.sigmoid(self.model(x)), 0.0, 1.0)