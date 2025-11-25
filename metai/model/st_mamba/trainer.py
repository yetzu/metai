import lightning as l
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from metai.model.core import get_optim_scheduler, timm_schedulers
from .model import MeteoMamba
from .loss import HybridLoss

class MeteoMambaModule(l.LightningModule):
    def __init__(self, in_shape: Tuple[int, int, int, int] = (10, 31, 256, 256),
                 hid_S: int = 64, hid_T: int = 256, N_S: int = 4, N_T: int = 8,
                 aft_seq_length: int = 20, lr: float = 5e-4, min_lr: float = 1e-5,
                 warmup_lr: float = 1e-5, warmup_epoch: int = 10, weight_decay: float = 0.05,
                 momentum: float = 0.9, opt: str = 'adamw', sched: str = 'cosine',
                 max_epochs: int = 100, use_curriculum_learning: bool = True, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = MeteoMamba(in_shape, hid_S, hid_T, N_S, N_T, spatio_kernel_enc=3, spatio_kernel_dec=3,
                                out_channels=1, aft_seq_length=aft_seq_length)
        self.criterion = HybridLoss(l1_weight=10.0, ssim_weight=1.0, csi_weight=0.0, spectral_weight=0.0, evo_weight=0.0)
        self.resize_shape = (in_shape[2], in_shape[3])

    def configure_optimizers(self):
        class Args:
            def __init__(self, hparams):
                self.__dict__.update(hparams)
                self.filter_bias_and_bn = True
                self.decay_epoch = 30
                self.decay_rate = 0.1
        return {"optimizer": get_optim_scheduler(Args(self.hparams), self.hparams.max_epochs, self.model)[0],
                "lr_scheduler": {"scheduler": get_optim_scheduler(Args(self.hparams), self.hparams.max_epochs, self.model)[1],
                                 "interval": "epoch"}}

    def lr_scheduler_step(self, scheduler, metric):
        if any(isinstance(scheduler, sch) for sch in timm_schedulers): scheduler.step(epoch=self.current_epoch)
        else: scheduler.step(metric)

    def on_train_epoch_start(self):
        if not self.hparams.use_curriculum_learning: return
        epoch, max_epochs = self.current_epoch, self.hparams.max_epochs
        phase_1_end = int(0.2 * max_epochs)
        phase_2_end = int(0.6 * max_epochs)
        
        weights = {}
        if epoch < phase_1_end:
            # Phase 1: Structure Warmup
            weights = {'l1': 10.0, 'ssim': 1.0, 'evo': 0.0, 'spec': 0.0, 'csi': 0.0}
        elif epoch < phase_2_end:
            # Phase 2: Safe Mode Physics
            p = (epoch - phase_1_end) / (phase_2_end - phase_1_end)
            # Safe Mode: Keep L1 high (10->5), introduce Evo slowly (0->0.2)
            weights = {'l1': 10.0 - p * 5.0, 'ssim': 1.0, 'evo': p * 0.2, 'spec': p * 0.05, 'csi': p * 0.2}
        else:
            # Phase 3: Metric Sprint
            p = (epoch - phase_2_end) / (max_epochs - phase_2_end)
            weights = {'l1': 5.0 - p * 4.0, 'ssim': 1.0 - p * 0.5, 'evo': 0.2 + p * 0.3, 'spec': 0.05 + p * 0.15, 'csi': 0.2 + (5.0 - 0.2) * (p**2)}
            
        self.criterion.weights.update(weights)
        for k, v in weights.items(): self.log(f"train/weight_{k}", v, on_epoch=True, sync_dist=True)

    def _interpolate_batch(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        x = F.interpolate(x, size=self.resize_shape, mode='bilinear', align_corners=False)
        return x.view(B, T, C, *self.resize_shape)

    def training_step(self, batch, batch_idx):
        _, x, y, mask, _ = batch
        x, y, mask = self._interpolate_batch(x), self._interpolate_batch(y), self._interpolate_batch(mask.float()).bool()
        pred = self.model(x)
        loss, loss_dict = self.criterion(pred, y, mask)
        self.log("train_loss", loss, prog_bar=True)
        for k, v in loss_dict.items():
            if k != 'total': self.log(f"train_loss_{k}", v)
        return loss

    def validation_step(self, batch, batch_idx):
        _, x, y, mask, _ = batch
        x, y, mask = self._interpolate_batch(x), self._interpolate_batch(y), self._interpolate_batch(mask.float()).bool()
        logits = self.model(x)
        loss, _ = self.criterion(logits, y, mask)
        pred = torch.sigmoid(logits)
        pred_mm, target_mm = pred * 30.0, y * 30.0
        hits = ((pred_mm > 1.0) & (target_mm > 1.0)).float().sum()
        union = ((pred_mm > 1.0) | (target_mm > 1.0)).float().sum()
        csi = hits / (union + 1e-6)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        self.log("val_csi", csi, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        _, x, y, target_mask, input_mask = batch
        x, y, target_mask = self._interpolate_batch(x), self._interpolate_batch(y), self._interpolate_batch(target_mask.float()).bool()
        logits = self.model(x)
        y_pred = torch.clamp(torch.sigmoid(logits), 0.0, 1.0)
        loss, _ = self.criterion(logits, y, mask=target_mask)
        self.log('test_loss', loss, on_epoch=True)
        return {'inputs': x[0, :, 0].cpu().float().numpy(), 'preds': y_pred[0].squeeze().cpu().float().numpy(), 'trues': y[0].squeeze().cpu().float().numpy()}

    def infer_step(self, batch, batch_idx):
        metadata, x, input_mask = batch
        x = self._interpolate_batch(x)
        return torch.clamp(torch.sigmoid(self.model(x)), 0.0, 1.0)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0): return self.infer_step(batch, batch_idx)