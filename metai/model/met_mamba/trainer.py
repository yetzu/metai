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
    def __init__(self, config: Optional[Union[ModelConfig, dict]] = None, **kwargs):
        super().__init__()
        
        # 配置初始化
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

        hparams = self.config.to_dict()
        for key in ['batch_size', 'data_path']:
            hparams.pop(key, None)
        self.save_hyperparameters(hparams)
        
        # 初始化模型
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
        
        # 初始化损失 (包含 MS-SSIM)
        self.criterion = HybridLoss(
            weight_focal=self.config.weight_focal, 
            weight_msssim=self.config.weight_msssim, # New
            weight_corr=self.config.weight_corr,
            weight_dice=self.config.weight_dice,
            intensity_weights=self.config.intensity_weights,
            focal_alpha=self.config.focal_alpha,
            focal_gamma=self.config.focal_gamma,
            false_alarm_penalty=self.config.false_alarm_penalty,
            corr_smooth_eps=self.config.corr_smooth_eps
        )
        
        self.valid_scorer = MetScore()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self) -> Dict[str, Any]:
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
        optimizer.zero_grad(set_to_none=True)

    def lr_scheduler_step(self, scheduler: Any, metric: Any):
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            scheduler.step(metric) if metric is not None else scheduler.step()

    def on_train_epoch_start(self):
        """课程学习策略：动态调整 Loss 权重"""
        if not self.config.use_curriculum_learning:
            return

        epoch = self.current_epoch
        target_msssim = self.config.weight_msssim
        target_dice = self.config.weight_dice
        target_corr = self.config.weight_corr
        
        total_warmup = self.config.warmup_epoch
        phase_1_end = int(total_warmup * 0.4) 
        phase_2_end = total_warmup
        
        if epoch < phase_1_end:
            alpha = 0.0
        elif epoch < phase_2_end:
            progress = (epoch - phase_1_end) / float(phase_2_end - phase_1_end)
            alpha = progress
        else:
            alpha = 1.0
            
        if hasattr(self.criterion, 'weights'):
            self.criterion.weights['msssim'] = target_msssim * alpha
            self.criterion.weights['dice'] = target_dice * alpha
            self.criterion.weights['corr'] = target_corr * alpha

        self.log_dict({
            'curriculum/alpha': alpha,
            'curriculum/phase': 1.0 if epoch < phase_1_end else (2.0 if epoch < phase_2_end else 3.0)
        }, on_step=False, on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        _, x, y, _, t_mask = batch
        if t_mask.dtype != torch.float32: t_mask = t_mask.float()
        
        # ====================================================
        # [时间维度课程学习] (Temporal Curriculum)
        # 前 20% Epoch: 仅计算前 10 帧 (0-60min)，专注短临
        # 之后: 计算全部 20 帧 (0-120min)，解锁长时效
        # ====================================================
        if self.config.use_curriculum_learning:
            progress = self.current_epoch / float(self.config.max_epochs)
            if progress < 0.2:
                valid_len = 10
                # t_mask: [B, T, H, W]
                # 将 T > 10 的部分 mask 置 0
                time_mask = torch.ones((1, y.shape[1], 1, 1), device=self.device)
                time_mask[:, valid_len:, :, :] = 0.0
                t_mask = t_mask * time_mask

        pred = torch.clamp(self.model(x), 0.0, 1.0)
        
        loss, loss_dict = self.criterion(pred, y, mask=t_mask)
        
        bs = x.size(0)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        
        for k, v in loss_dict.items():
            if k != 'total':
                self.log(f'loss_{k}', v, on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
                
        return loss

    def validation_step(self, batch, batch_idx):
        _, x, y, _, t_mask = batch
        if t_mask.dtype != torch.float32: t_mask = t_mask.float()
        
        pred = torch.clamp(self.model(x), 0.0, 1.0)
        
        loss, _ = self.criterion(pred, y, mask=t_mask)
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