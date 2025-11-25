import lightning as l
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from metai.model.core import get_optim_scheduler, timm_schedulers
from .model import MeteoMamba
from .loss import HybridLoss

class MeteoMambaModule(l.LightningModule):
    def __init__(
        self, 
        in_shape: Tuple[int, int, int, int] = (10, 31, 256, 256),
        hid_S: int = 64,
        hid_T: int = 256,
        N_S: int = 4,
        N_T: int = 8,
        aft_seq_length: int = 20,
        # 优化器参数
        lr: float = 5e-4,
        min_lr: float = 1e-5,
        warmup_lr: float = 1e-5,
        warmup_epoch: int = 10,
        weight_decay: float = 0.05,
        momentum: float = 0.9,
        opt: str = 'adamw',
        sched: str = 'cosine',
        max_epochs: int = 100,
        # 策略参数
        use_curriculum_learning: bool = True,
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
            out_channels=1  # 强制单通道输出
        )
        
        self.criterion = HybridLoss(
            l1_weight=10.0, 
            ssim_weight=1.0,
            csi_weight=0.0,
            spectral_weight=0.0,
            evo_weight=0.0
        )
        
        self.resize_shape = (in_shape[2], in_shape[3])

    def configure_optimizers(self):
        class Args:
            def __init__(self, hparams):
                self.opt = hparams.opt
                self.lr = hparams.lr
                self.weight_decay = hparams.weight_decay
                self.momentum = hparams.momentum
                self.filter_bias_and_bn = True
                self.sched = hparams.sched
                self.min_lr = hparams.min_lr
                self.warmup_lr = hparams.warmup_lr
                self.warmup_epoch = hparams.warmup_epoch
                self.decay_epoch = 30
                self.decay_rate = 0.1

        args = Args(self.hparams)
        optimizer, scheduler, by_epoch = get_optim_scheduler(args, self.hparams.max_epochs, self.model)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch" if by_epoch else "step"
            }
        }

    def lr_scheduler_step(self, scheduler: Any, metric: Any):
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            scheduler.step(metric) if metric is not None else scheduler.step()

    def on_train_epoch_start(self):
        if not self.hparams.use_curriculum_learning: return
        epoch = self.current_epoch
        max_epochs = self.hparams.max_epochs
        
        phase_1_end = int(0.2 * max_epochs)
        phase_2_end = int(0.6 * max_epochs)
        
        weights = {}
        if epoch < phase_1_end:
            weights = {'l1': 10.0, 'ssim': 1.0, 'evo': 0.0, 'spec': 0.0, 'csi': 0.0}
        elif epoch < phase_2_end:
            p = (epoch - phase_1_end) / (phase_2_end - phase_1_end)
            weights = {
                'l1': 10.0 - p * 5.0, 'ssim': 1.0,
                'evo': p * 0.1, 'spec': p * 0.05, 'csi': p * 0.5
            }
        else:
            p = (epoch - phase_2_end) / (max_epochs - phase_2_end)
            weights = {
                'l1': 5.0 - p * 4.0, 'ssim': 1.0 - p * 0.5,
                'evo': 0.1 + p * 0.4, 'spec': 0.05 + p * 0.15,
                'csi': 0.5 + (5.0 - 0.5) * (p**2)
            }
        self.criterion.weights.update(weights)
        for k, v in weights.items(): self.log(f"train/weight_{k}", v, on_epoch=True, sync_dist=True)

    def _interpolate_batch(self, batch_tensor: torch.Tensor, mode: str = 'max_pool') -> torch.Tensor:
        """
        [Strictly aligned with SimVP logic]
        处理 (B, T, C, H, W) 格式的 Tensor 插值
        """
        if self.resize_shape is None: return batch_tensor
        T, C, H, W = batch_tensor.shape[1:]
        target_H, target_W = self.resize_shape
        if H == target_H and W == target_W: return batch_tensor
        
        # Handle bool masks
        is_bool = batch_tensor.dtype == torch.bool
        if is_bool: batch_tensor = batch_tensor.float()
        
        B = batch_tensor.shape[0]
        batch_tensor = batch_tensor.view(B * T, C, H, W)
        
        if mode == 'max_pool':
            # 如果是降采样，优先使用 max_pool 保留极值（如强降水中心）
            if target_H < H or target_W < W:
                processed_tensor = F.adaptive_max_pool2d(batch_tensor, output_size=self.resize_shape)
            else:
                processed_tensor = F.interpolate(batch_tensor, size=self.resize_shape, mode='bilinear', align_corners=False)
        elif mode in ['nearest', 'bilinear']:
            align = False if mode == 'bilinear' else None
            processed_tensor = F.interpolate(batch_tensor, size=self.resize_shape, mode=mode, align_corners=align)
        else:
            raise ValueError(f"Unsupported interpolation mode: {mode}")

        processed_tensor = processed_tensor.view(B, T, C, target_H, target_W)
        if is_bool: processed_tensor = processed_tensor.bool()
        return processed_tensor

    def training_step(self, batch, batch_idx):
        _, x, y, mask, _ = batch
        # 对齐 SimVP: 数据使用 max_pool 保留特征，Mask 使用 nearest 防止插值产生小数
        x = self._interpolate_batch(x, mode='max_pool')
        y = self._interpolate_batch(y, mode='max_pool')
        mask = self._interpolate_batch(mask, mode='nearest')
        
        pred = self.model(x)
        loss, loss_dict = self.criterion(pred, y, mask)
        
        self.log("train_loss", loss, prog_bar=True)
        for k, v in loss_dict.items():
            if k != 'total': self.log(f"train_loss_{k}", v)
        return loss

    def validation_step(self, batch, batch_idx):
        _, x, y, mask, _ = batch
        x = self._interpolate_batch(x, mode='max_pool')
        y = self._interpolate_batch(y, mode='max_pool')
        mask = self._interpolate_batch(mask, mode='nearest')
        
        logits = self.model(x)
        loss, _ = self.criterion(logits, y, mask)
        
        pred = torch.sigmoid(logits)
        y_pred_clamped = torch.clamp(pred, 0.0, 1.0)
        
        # Ensure 4D for metric calculation: [B, T, H, W]
        if y_pred_clamped.dim() == 5: y_pred_clamped = y_pred_clamped.squeeze(2)
        if y.dim() == 5: y = y.squeeze(2)

        # [Strictly aligned with SimVP metrics]
        mae = F.l1_loss(y_pred_clamped, y)
        
        MM_MAX = 30.0
        pred_mm = y_pred_clamped * MM_MAX
        target_mm = y * MM_MAX
        thresholds = [0.01, 0.1, 1.0, 2.0, 5.0, 8.0] 
        weights =    [0.1,  0.1, 0.1, 0.2, 0.2, 0.3]
        ts_sum = 0.0
        
        for t, w in zip(thresholds, weights):
            hits = ((pred_mm >= t) & (target_mm >= t)).float().sum()
            misses = ((pred_mm < t) & (target_mm >= t)).float().sum()
            false_alarms = ((pred_mm >= t) & (target_mm < t)).float().sum()
            ts = hits / (hits + misses + false_alarms + 1e-6)
            ts_sum += ts * w
            
        val_score = ts_sum / sum(weights)
        
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        self.log("val_mae", mae, on_epoch=True, sync_dist=True)
        self.log("val_score", val_score, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        _, x, y, target_mask, input_mask = batch
        x = self._interpolate_batch(x, mode='max_pool')
        y = self._interpolate_batch(y, mode='max_pool')
        target_mask = self._interpolate_batch(target_mask, mode='nearest')
        
        logits = self.model(x)
        y_pred = torch.clamp(torch.sigmoid(logits), 0.0, 1.0)
        
        loss, _ = self.criterion(logits, y, mask=target_mask)
        self.log('test_loss', loss, on_epoch=True)
        
        return {
            'inputs': x[0, :, 0].cpu().float().numpy(), 
            'preds': y_pred[0].squeeze().cpu().float().numpy(), 
            'trues': y[0].squeeze().cpu().float().numpy() 
        }
    
    def infer_step(self, batch, batch_idx):
        metadata, x, input_mask = batch 
        x = self._interpolate_batch(x, mode='max_pool')
        logits_pred = self(x)
        y_pred = torch.sigmoid(logits_pred)
        return torch.clamp(y_pred, 0.0, 1.0)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.infer_step(batch, batch_idx)