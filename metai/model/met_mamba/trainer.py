import lightning as l
import torch
from typing import Optional, Union, Any, Dict

from metai.model.core import get_optim_scheduler
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

        # 保存超参数
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
        
        # 初始化损失 (HybridLoss)
        self.criterion = HybridLoss(
            weight_focal=self.config.weight_focal, 
            weight_grad=self.config.weight_grad,
            weight_corr=self.config.weight_corr,
            weight_dice=self.config.weight_dice,
            intensity_weights=self.config.intensity_weights,
            focal_alpha=self.config.focal_alpha,
            focal_gamma=self.config.focal_gamma,
            false_alarm_penalty=self.config.false_alarm_penalty,
            corr_smooth_eps=self.config.corr_smooth_eps
        )
        
        # 评估指标
        self.valid_scorer = MetScore()

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

    def training_step(self, batch, batch_idx):
        _, x, y, _, t_mask = batch
        if t_mask.dtype != torch.float32: t_mask = t_mask.float()
        
        pred = torch.clamp(self.model(x), 0.0, 1.0)
        
        loss, loss_dict = self.criterion(pred, y, mask=t_mask)
        
        bs = x.size(0)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        
        # 记录分项损失 (loss_focal, loss_grad, loss_corr, loss_dice)
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
        val_score = metric_results['total_score']

        bs = x.size(0)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        self.log('val_score', val_score, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)

    def test_step(self, batch, batch_idx):
        _, x, y, _, t_mask = batch
        pred = torch.clamp(self.model(x), 0.0, 1.0)
        return {'inputs': x, 'trues': y, 'preds': pred}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        _, x, _ = batch
        return torch.clamp(self.model(x), 0.0, 1.0)