import lightning as l
import torch

# 引入项目配置与工具
from metai.utils import MetLabel
from metai.model.core import get_optim_scheduler, timm_schedulers
from .model import MeteoMamba
from .loss import HybridLoss
from .metrices import MetScore  # [新增] 引入评分计算模块

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
        
        # --- 优化器与调度器参数 ---
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
        filter_bias_and_bn: bool = False, 
        
        # --- Loss 权重 ---
        loss_weight_l1: float = 1.0,
        loss_weight_gdl: float = 1.0,
        
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # [Explicit Physics Management]
        # 显式定义数据缩放因子与物理极值
        # 说明：原始数据 RA 存储值放大了 10 倍 (例如物理量 30mm 存储为 300)
        # Dataset 使用 MetLabel.RA.max (300) 进行归一化
        # 因此，归一化数值 1.0 对应的物理值为 300 / 10 = 30.0 mm
        self.ra_storage_factor = 10.0
        self.mm_max = float(MetLabel.RA.max) / self.ra_storage_factor  # 结果为 30.0
        
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
        
        # [修改] 使用 MetScore 进行验证指标计算
        # MetScore 默认 data_max=30.0，与 self.mm_max 一致，直接初始化即可
        self.valid_scorer = MetScore()

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

    def training_step(self, batch, batch_idx):
        # [Dataset 顺序]: Meta, Input, Target, Input_Mask, Target_Mask
        # 解包顺序修正：Target Mask 在第 5 位 (index 4)
        _, x, y, _, t_mask = batch
        
        # 确保 mask 类型正确
        if t_mask.dtype != torch.float32:
             t_mask = t_mask.float()
        
        pred_raw = self.model(x)
        pred = torch.clamp(pred_raw, 0.0, 1.0)
        
        loss, loss_dict = self.criterion(pred, y, mask=t_mask)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        self.log('loss_l1', loss_dict.get('l1', 0), on_step=False, on_epoch=True, prog_bar=False, batch_size=x.size(0))
        self.log('loss_gdl', loss_dict.get('gdl', 0), on_step=False, on_epoch=True, prog_bar=False, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        _, x, y, _, t_mask = batch
        
        if t_mask.dtype != torch.float32:
             t_mask = t_mask.float()
        
        pred_raw = self.model(x)
        pred = torch.clamp(pred_raw, 0.0, 1.0)
        
        loss, _ = self.criterion(pred, y, mask=t_mask)
        
        # [修改] 调用 MetScore 计算综合评分
        # MetScore.forward 返回字典 {'total_score': ..., 'score_time': ..., ...}
        metric_results = self.valid_scorer(pred, y, mask=t_mask)
        val_score = metric_results['total_score']

        self.log('val_loss', loss, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=x.size(0))
        self.log('val_score', val_score, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=x.size(0))

    def test_step(self, batch, batch_idx):
        _, x, y, _, _ = batch
        
        pred_raw = self.model(x)
        pred = torch.clamp(pred_raw, 0.0, 1.0)
        return {'inputs': x, 'trues': y, 'preds': pred}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        _, x, _ = batch
        return torch.clamp(self.model(x), 0.0, 1.0)