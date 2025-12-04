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
    [MeteoMamba Lightning Module v2.1 - Stable Training Edition]
    
    核心特性：
    1. 三阶段课程学习：
       - Phase 1 (Warmup): 仅物理/内容 Loss，稀疏率=0，GAN=OFF
       - Phase 2 (Sparsity): 开启稀疏化退火，Router 学习重点区域
       - Phase 3 (Texture): 开启 GAN (带权重预热)，精修高频纹理
    2. 训练稳定性增强：
       - GAN Weight Linear Warmup
       - Manual Optimization Control
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
        gan_start_epoch: int = 30, # 建议设为 15-20，在稀疏化稳定后再开启
        gan_weight: float = 0.01,
        discriminator_lr: float = 2e-4,
        
        # --- Loss 配置 ---
        use_temporal_weight: bool = True,
        
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. 模型初始化
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
        
        # 2. Loss 初始化
        self.criterion_content = HybridLoss(use_temporal_weight=use_temporal_weight)
        self.criterion_gan = GANLoss(mode='hinge')
        
        # 3. Metrics 初始化
        self.train_metrics = MetMetricCollection(prefix="train_")
        self.val_metrics = MetMetricCollection(prefix="val_")
        self.test_metrics = MetMetricCollection(prefix="test_")

        # 4. 开启手动优化 (Manual Optimization)
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor, current_epoch: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(x, current_epoch=current_epoch)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        meta, x, y, x_mask, y_mask = batch
        prev_frame = x[:, -1:, :self.model.out_channels, ...]
        
        # 策略检查：是否处于 GAN 阶段
        # 必须同时满足：配置启用 + 当前 Epoch 达到起始点
        do_gan = self.hparams.use_gan and (self.current_epoch >= self.hparams.gan_start_epoch)

        # ======================================================================
        # Phase 1: 判别器训练 (Discriminator Step)
        # ======================================================================
        if do_gan:
            # 生成假样本 (无梯度)
            with torch.no_grad():
                y_fake_detached, _ = self.model(x, current_epoch=self.current_epoch)
            
            logits_real = self.model.discriminator(y)
            logits_fake = self.model.discriminator(y_fake_detached)
            
            loss_d = self.criterion_gan.get_disc_loss(logits_real, logits_fake)
            
            opt_d.zero_grad()
            self.manual_backward(loss_d)
            opt_d.step()
            
            self.log("train_loss_d", loss_d, prog_bar=True, on_step=True, on_epoch=True)

        # ======================================================================
        # Phase 2: 生成器训练 (Generator Step)
        # ======================================================================
        # 前向传播 (带梯度)
        y_fake, flows = self.model(x, current_epoch=self.current_epoch)
        
        # 1. 内容损失 (Physics + MSE + CSI)
        loss_content, loss_dict = self.criterion_content(
            pred=y_fake, 
            target=y, 
            flow=flows, 
            prev_frame=prev_frame, 
            mask=y_mask
        )
        
        # 2. 对抗损失 (Adversarial)
        loss_adv = torch.tensor(0.0, device=self.device)
        current_gan_weight = 0.0
        
        if do_gan:
            logits_fake_g = self.model.discriminator(y_fake)
            loss_adv = self.criterion_gan.get_gen_loss(logits_fake_g)
            
            # [策略优化] GAN 权重线性预热 (Linear Warmup)
            # 防止 GAN 刚介入时 Loss 剧变导致模型崩塌
            # 预热周期: 5 Epochs
            warmup_epochs = 5
            steps = self.current_epoch - self.hparams.gan_start_epoch
            target_weight = self.hparams.gan_weight
            
            if steps < warmup_epochs:
                current_gan_weight = target_weight * (steps / warmup_epochs)
            else:
                current_gan_weight = target_weight
        
        # 3. 总损失
        total_loss_g = loss_content + current_gan_weight * loss_adv
        
        opt_g.zero_grad()
        self.manual_backward(total_loss_g)
        # 梯度裁剪 (防止梯度爆炸，这对 Physics+GAN 很重要)
        self.clip_gradients(opt_g, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        opt_g.step()
        
        # 4. 日志记录
        self.log("train_loss_g", total_loss_g, prog_bar=True, on_step=True, on_epoch=True)
        if do_gan:
            self.log("train_loss_adv", loss_adv, prog_bar=False, on_step=False, on_epoch=True)
            self.log("gan_w", current_gan_weight, prog_bar=False, on_step=False, on_epoch=True)
            
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
        # 推理时使用最大 Epoch，确保使用最终的稀疏率
        y_hat, _ = self(x, current_epoch=self.hparams.max_epochs)
        return y_hat

    def on_train_epoch_end(self):
        # 记录训练指标
        self.log_dict(self.train_metrics.compute(), prog_bar=False)
        self.train_metrics.reset()
        
        # 手动更新学习率调度器
        sch_g, sch_d = self.lr_schedulers()
        
        # 确保在主进程操作
        sch_g.step()
        sch_d.step()
        
        # 监控稀疏率变化 (Debug)
        if hasattr(self.model.evolution.layers[0], '_get_curr_sparse_ratio'):
            curr_ratio = self.model.evolution.layers[0]._get_curr_sparse_ratio(self.current_epoch)
            self.log("curr_sparse_ratio", curr_ratio, prog_bar=False, on_epoch=True)

    def on_validation_epoch_end(self):
        """
        验证结束：计算指标并记录 Checkpoint 依据。
        """
        metrics = self.val_metrics.compute()
        
        # [修复] 显式记录 val_score 用于 ModelCheckpoint
        # 逻辑：优先用 total_score，没有则用 csi_avg
        score = metrics.get('val_total_score', metrics.get('val_csi_avg', 0.0))
        self.log('val_score', score, prog_bar=True, logger=True, sync_dist=True)
        
        # 记录 CSI 均值 (业务最关注指标)
        if 'val_csi_avg' in metrics:
            self.log('val_csi', metrics['val_csi_avg'], prog_bar=True, sync_dist=True)
            
        self.log_dict(metrics, prog_bar=False, logger=True, sync_dist=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), prog_bar=False, sync_dist=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        # 过滤 Generator 参数 (排除 Discriminator)
        g_params = [p for n, p in self.model.named_parameters() if "discriminator" not in n]
        d_params = self.model.discriminator.parameters()
        
        opt_g = AdamW(g_params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        opt_d = AdamW(d_params, lr=self.hparams.discriminator_lr, weight_decay=self.hparams.weight_decay)
        
        # 使用 Cosine 调度
        sch_g = CosineAnnealingLR(opt_g, T_max=self.hparams.max_epochs, eta_min=1e-6)
        sch_d = CosineAnnealingLR(opt_d, T_max=self.hparams.max_epochs, eta_min=1e-6)
        
        return [opt_g, opt_d], [sch_g, sch_d]