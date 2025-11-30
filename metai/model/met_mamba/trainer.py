import lightning as l
import torch
from typing import Optional, Union, Any, Dict

# 新增导入：用于类型检查
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

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        # 使用 set_to_none=True 代替默认的 zero_grad()
        # 这可以重置梯度的内存布局，通常能消除 stride 不匹配的警告
        optimizer.zero_grad(set_to_none=True)

    # =========================================================
    # [关键修复] 重写 LR Scheduler Step 逻辑以兼容 timm
    # =========================================================
    def lr_scheduler_step(self, scheduler: Any, metric: Any):
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            scheduler.step(metric) if metric is not None else scheduler.step()

    # =========================================================
    # [修复版] 课程学习调度器 (动态读取 Config)
    # =========================================================
    def on_train_epoch_start(self):
        """
        动态调整 Loss 权重：
        - 阶段1 (L1 Only): 前 1/3 的 warmup 周期，仅使用 Focal Loss 学习基础分布。
        - 阶段2 (Ramp-up): 剩余 warmup 周期，线性增加 Grad/Dice/Corr 权重。
        - 阶段3 (Full): warmup 结束后，使用全权重精调。
        """
        if not self.config.use_curriculum_learning:
            return

        epoch = self.current_epoch
        
        # 1. 获取配置中的最终目标权重
        target_grad = self.config.weight_grad
        target_dice = self.config.weight_dice
        target_corr = self.config.weight_corr
        
        # 2. 动态计算阶段阈值 (基于 config.warmup_epoch)
        # 策略：前 30% 的时间只学 L1，剩下 70% 的时间线性增加难度
        total_warmup = self.config.warmup_epoch
        phase_1_end = int(total_warmup * 0.4) 
        phase_2_end = total_warmup
        
        # 3. 计算当前权重系数 (0.0 -> 1.0)
        if epoch < phase_1_end:
            # 阶段1: 纯净 L1 模式，让模型先学会“哪里有雨”
            alpha = 0.0
        elif epoch < phase_2_end:
            # 阶段2: 线性增长，逐渐引入纹理(Grad)和评分(Dice)约束
            progress = (epoch - phase_1_end) / float(phase_2_end - phase_1_end)
            alpha = progress
        else:
            # 阶段3: 全开模式
            alpha = 1.0
            
        # 4. 更新 Loss 模块内部的权重
        if hasattr(self.criterion, 'weights'):
            self.criterion.weights['grad'] = target_grad * alpha
            self.criterion.weights['dice'] = target_dice * alpha
            self.criterion.weights['corr'] = target_corr * alpha
            
            # Focal 权重保持不变 (作为基础引导)
            # self.criterion.weights['focal'] = self.config.weight_focal 

        # 5. 记录日志 (这对监控是否过早引入复杂 Loss 非常重要)
        self.log_dict({
            'curriculum/alpha': alpha,
            'curriculum/phase': 1.0 if epoch < phase_1_end else (2.0 if epoch < phase_2_end else 3.0)
        }, on_step=False, on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        _, x, y, _, t_mask = batch
        if t_mask.dtype != torch.float32: t_mask = t_mask.float()
        
        pred = torch.clamp(self.model(x), 0.0, 1.0)
        
        loss, loss_dict = self.criterion(pred, y, mask=t_mask)
        
        bs = x.size(0)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        
        # 记录分项损失
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