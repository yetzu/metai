# metai/model/met_mamba/trainer.py

import lightning as l
import torch
from .model import MeteoMamba
from .loss import HybridLoss
from metai.model.core import get_optim_scheduler, timm_schedulers

class MeteoMambaModule(l.LightningModule):
    def __init__(
        self,
        # --- 模型结构参数 ---
        in_shape: tuple = (10, 31, 256, 256),
        hid_S: int = 64,
        hid_T: int = 256,
        N_S: int = 4,
        N_T: int = 8,
        aft_seq_length: int = 20,
        
        # --- Mamba 核心参数 ---
        mamba_d_state: int = 16, 
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        
        # --- 训练策略 ---
        use_curriculum_learning: bool = False,
        max_epochs: int = 50,
        
        # --- 优化器与调度器参数 (完整版) ---
        opt: str = "adamw",
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        momentum: float = 0.9,
        sched: str = "cosine",
        min_lr: float = 1e-5,
        warmup_lr: float = 1e-5,
        warmup_epoch: int = 5,
        decay_epoch: int = 30,
        decay_rate: float = 0.1,
        
        # [修复点] 必须添加此参数，optim_scheduler.py 强依赖它
        # False: 对 bias 和 bn 层也使用 weight_decay (通常选 False 或 True 均可，这里给个默认值防止报错)
        filter_bias_and_bn: bool = False, 
        
        # --- Loss 权重 ---
        loss_weight_l1: float = 1.0,
        loss_weight_gdl: float = 1.0,
        
        **kwargs
    ):
        super().__init__()
        # 保存所有超参，供 get_optim_scheduler 使用
        self.save_hyperparameters()
        
        self.model = MeteoMamba(
            in_shape=in_shape,
            hid_S=hid_S,
            hid_T=hid_T,
            N_S=N_S,
            N_T=N_T,
            aft_seq_length=aft_seq_length,
            out_channels=1,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand
        )
        
        self.criterion = HybridLoss(
            l1_weight=loss_weight_l1, 
            gdl_weight=loss_weight_gdl
        )
        
        self.resize_shape = (in_shape[2], in_shape[3])
        self.MM_MAX = 30.0
        # 验证指标相关
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

    def lr_scheduler_step(self, scheduler, metric):
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            scheduler.step(metric) if metric is not None else scheduler.step()

    def _interpolate_batch(self, x, mode='bilinear'):
        if self.resize_shape is None: return x
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = torch.nn.functional.interpolate(x, size=self.resize_shape, mode='bilinear', align_corners=False)
        return x.view(B, T, C, *self.resize_shape)

    def training_step(self, batch, batch_idx):
        _, x, y, t_mask, _ = batch
        x = self._interpolate_batch(x, mode='bilinear')
        y = self._interpolate_batch(y, mode='bilinear')
        t_mask = self._interpolate_batch(t_mask.float(), mode='nearest')
        
        pred_raw = self.model(x)
        pred = torch.clamp(pred_raw, 0.0, 1.0)
        
        loss, loss_dict = self.criterion(pred, y, mask=t_mask)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('loss_l1', loss_dict.get('l1', 0), on_step=False, on_epoch=True, prog_bar=False)
        self.log('loss_gdl', loss_dict.get('gdl', 0), on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        _, x, y, t_mask, _ = batch
        x = self._interpolate_batch(x, mode='bilinear')
        y = self._interpolate_batch(y, mode='bilinear')
        t_mask = self._interpolate_batch(t_mask.float(), mode='nearest')
        
        pred_raw = self.model(x)
        pred = torch.clamp(pred_raw, 0.0, 1.0)
        
        loss, _ = self.criterion(pred, y, mask=t_mask)
        
        pred_mm = pred * self.MM_MAX
        target_mm = y * self.MM_MAX
        weighted_csi_sum = 0.0
        valid_mask = t_mask > 0.5
        
        for i, threshold in enumerate(self.val_thresholds):
            hits = ((pred_mm >= threshold) & (target_mm >= threshold) & valid_mask).float().sum()
            union = (((pred_mm >= threshold) | (target_mm >= threshold)) & valid_mask).float().sum()
            csi = hits / (union + 1e-6)
            weighted_csi_sum += csi * self.val_weights[i]

        self.log('val_loss', loss, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('val_score', weighted_csi_sum, on_epoch=True, sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        _, x, y, t_mask, _ = batch
        x = self._interpolate_batch(x, mode='bilinear')
        y = self._interpolate_batch(y, mode='bilinear')
        
        pred_raw = self.model(x)
        pred = torch.clamp(pred_raw, 0.0, 1.0)
        return {'inputs': x, 'trues': y, 'preds': pred}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        _, x, _ = batch
        x = self._interpolate_batch(x, mode='bilinear')
        return torch.clamp(self.model(x), 0.0, 1.0)