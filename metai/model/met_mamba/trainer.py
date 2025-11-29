import lightning as l
import torch
from typing import Optional, Union, Any

# 引入项目配置与工具
from metai.utils import MetLabel
from metai.model.core import get_optim_scheduler, timm_schedulers
from .model import MeteoMamba
from .loss import HybridLoss
from .metrices import MetScore
from .config import ModelConfig

class MeteoMambaModule(l.LightningModule):
    def __init__(
        self, 
        config: Optional[Union[ModelConfig, dict]] = None,
        **kwargs 
    ):
        super().__init__()
        
        # 1. 配置对象初始化
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

        # 2. 保存超参数 [关键修改]
        # 移除与 DataModule 冲突的参数 (batch_size, data_path)
        # 这样 Lightning 就不会因为参数值不一致而报错
        hparams = self.config.to_dict()
        hparams.pop('batch_size', None)
        hparams.pop('data_path', None)
        self.save_hyperparameters(hparams)
        
        # 3. 初始化模型
        self.model = MeteoMamba(
            in_shape=self.config.in_shape,      # (C, H, W)
            in_seq_len=self.config.obs_seq_len, # T_in
            out_seq_len=self.config.pred_seq_len, # T_out
            out_channels=1,
            hid_S=self.config.hid_S,
            hid_T=self.config.hid_T,
            N_S=self.config.N_S,
            N_T=self.config.N_T,
            mamba_d_state=self.config.mamba_d_state,
            mamba_d_conv=self.config.mamba_d_conv,
            mamba_expand=self.config.mamba_expand
        )
        
        # 4. 初始化 Loss
        self.criterion = HybridLoss(
            l1_weight=self.config.loss_weight_l1, 
            gdl_weight=self.config.loss_weight_gdl,
            corr_weight=self.config.loss_weight_corr,
            dice_weight=self.config.loss_weight_dice
        )
        
        # 5. 辅助属性
        self.resize_shape = (self.config.in_shape[1], self.config.in_shape[2])
        self.valid_scorer = MetScore()

    def configure_optimizers(self) -> Any:
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

    def lr_scheduler_step(self, scheduler, metric):
        # 兼容 timm scheduler 和标准 PyTorch scheduler
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch, metric=metric) # type: ignore
        else:
            if metric is None:
                scheduler.step() # type: ignore
            else:
                try:
                    scheduler.step(metrics=metric) # type: ignore
                except TypeError:
                    scheduler.step() # type: ignore

    def training_step(self, batch, batch_idx):
        _, x, y, _, t_mask = batch
        
        if t_mask.dtype != torch.float32: t_mask = t_mask.float()
        
        pred_raw = self.model(x)
        pred = torch.clamp(pred_raw, 0.0, 1.0)
        
        loss, loss_dict = self.criterion(pred, y, mask=t_mask)
        
        bs = x.size(0)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        self.log('loss_l1', loss_dict.get('l1', 0), on_step=False, on_epoch=True, prog_bar=False, batch_size=bs, sync_dist=True)
        self.log('loss_gdl', loss_dict.get('gdl', 0), on_step=False, on_epoch=True, prog_bar=False, batch_size=bs, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, x, y, _, t_mask = batch
        
        if t_mask.dtype != torch.float32: t_mask = t_mask.float()
        
        pred_raw = self.model(x)
        pred = torch.clamp(pred_raw, 0.0, 1.0)
        
        loss, _ = self.criterion(pred, y, mask=t_mask)
        
        metric_results = self.valid_scorer(pred, y, mask=t_mask)
        val_score = metric_results['total_score']

        bs = x.size(0)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        self.log('val_score', val_score, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)

    def test_step(self, batch, batch_idx):
        _, x, y, _, t_mask = batch
        pred_raw = self.model(x)
        pred = torch.clamp(pred_raw, 0.0, 1.0)
        return {'inputs': x, 'trues': y, 'preds': pred}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        _, x, _ = batch
        return torch.clamp(self.model(x), 0.0, 1.0)