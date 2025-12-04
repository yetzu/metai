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
    [MeteoMamba Lightning Module v2.0]
    
    功能职责：
    1. 生命周期管理：负责模型的初始化、训练、验证、测试流程。
    2. GAN 训练编排：管理生成器 (Generator) 和判别器 (Discriminator) 的交替优化。
    3. 稀疏控制：将 current_epoch 注入模型，驱动稀疏率退火。
    4. 自动/手动优化：关闭 Lightning 自动优化，手动控制 backward 和 step。
    """
    
    def __init__(
        self, 
        # --- 模型架构超参 (需与 MeteoMamba __init__ 对应) ---
        in_shape: Tuple[int, int, int] = (19, 256, 256), # (C, H, W)
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
        gan_start_epoch: int = 0,
        gan_weight: float = 0.01,
        discriminator_lr: float = 2e-4,
        
        # --- Loss 配置 ---
        use_temporal_weight: bool = True,
        
        # --- 其他 kwargs ---
        **kwargs
    ):
        super().__init__()
        # 1. 保存超参 (Checkpoints 恢复需要)
        self.save_hyperparameters()
        
        # 2. 核心模型初始化 (生成器 + 判别器)
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
        
        # 3. 损失函数初始化
        # 内容损失：MSE, CSI, Spectral, Conservation, Warp
        self.criterion_content = HybridLoss(use_temporal_weight=use_temporal_weight)
        # 对抗损失：Hinge Loss
        self.criterion_gan = GANLoss(mode='hinge')
        
        # 4. 评估指标集合
        self.train_metrics = MetMetricCollection(prefix="train_")
        self.val_metrics = MetMetricCollection(prefix="val_")
        self.test_metrics = MetMetricCollection(prefix="test_")

        # 5. [关键] 开启手动优化模式 (Manual Optimization)
        # GAN 需要交替训练，PyTorch Lightning 的自动模式不再适用
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor, current_epoch: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播包装函数。
        Returns:
            y_hat: 降水预测
            flows: 潜在空间流场
        """
        return self.model(x, current_epoch=current_epoch)

    def training_step(self, batch, batch_idx):
        """
        训练步 (Manual Optimization Loop)
        包含 Discriminator 更新 和 Generator 更新。
        """
        # 获取优化器
        opt_g, opt_d = self.optimizers()
        
        # 解包数据
        meta, x, y, x_mask, y_mask = batch
        
        # 准备物理约束所需的上下文
        prev_frame = x[:, -1:, :self.model.out_channels, ...]
        
        # 判断是否启用 GAN (预热期过后且配置开启)
        do_gan = self.hparams.use_gan and (self.current_epoch >= self.hparams.gan_start_epoch)

        # ======================================================================
        # Phase 1: 判别器训练 (Train Discriminator)
        # ======================================================================
        # 仅当启用 GAN 时才训练 D
        if do_gan:
            # 1. 生成假数据 (Detach: 不传梯度给生成器)
            with torch.no_grad():
                y_fake_detached, _ = self.model(x, current_epoch=self.current_epoch)
            
            # 2. 判别器前向
            # 真实样本得分
            logits_real = self.model.discriminator(y)
            # 假样本得分
            logits_fake = self.model.discriminator(y_fake_detached)
            
            # 3. 计算 D Loss
            loss_d = self.criterion_gan.get_disc_loss(logits_real, logits_fake)
            
            # 4. 反向传播与更新
            opt_d.zero_grad()
            self.manual_backward(loss_d)
            opt_d.step()
            
            # 记录日志
            self.log("train_loss_d", loss_d, prog_bar=True, on_step=True, on_epoch=True)

        # ======================================================================
        # Phase 2: 生成器训练 (Train Generator)
        # ======================================================================
        
        # 1. 生成数据 (需要梯度)
        # 注意：这里重新执行一次 forward 以构建计算图，或者如果显存够大，
        # 也可以在 Phase 1 前执行一次 forward 并 retain_graph=True
        y_fake, flows = self.model(x, current_epoch=self.current_epoch)
        
        # 2. 计算内容损失 (MSE, CSI, Warp...)
        loss_content, loss_dict = self.criterion_content(
            pred=y_fake, 
            target=y, 
            flow=flows, 
            prev_frame=prev_frame, 
            mask=y_mask
        )
        
        # 3. 计算对抗损失 (Adversarial Loss)
        loss_adv = torch.tensor(0.0, device=self.device)
        if do_gan:
            # 判别器对生成样本的评分
            logits_fake_g = self.model.discriminator(y_fake)
            # 生成器的目标是让 D 认为这是真的
            loss_adv = self.criterion_gan.get_gen_loss(logits_fake_g)
        
        # 4. 动态调整 GAN 权重 (可选：随时间线性增加权重)
        # 策略：在 gan_start_epoch 后，权重从 0 逐渐增加到 gan_weight
        current_gan_weight = 0.0
        if do_gan:
            # 简单的恒定权重或基于 Epoch 的调度
            current_gan_weight = self.hparams.gan_weight
        
        # 5. 总损失
        total_loss_g = loss_content + current_gan_weight * loss_adv
        
        # 6. 反向传播与更新
        opt_g.zero_grad()
        self.manual_backward(total_loss_g)
        opt_g.step()
        
        # 7. 记录日志与指标
        self.log("train_loss_g", total_loss_g, prog_bar=True, on_step=True, on_epoch=True)
        if do_gan:
            self.log("train_loss_adv", loss_adv, prog_bar=False, on_step=False, on_epoch=True)
        
        # 记录详细的内容损失分量
        self.log_dict(loss_dict, prog_bar=False, on_step=False, on_epoch=True)
        
        # 计算业务指标 (CSI, MAE...)
        self.train_metrics(y_fake.detach(), y, y_mask)

    def validation_step(self, batch, batch_idx):
        """
        验证步 (标准推理)
        """
        meta, x, y, x_mask, y_mask = batch
        prev_frame = x[:, -1:, :self.model.out_channels, ...]
        
        # 推理
        y_hat, flows = self(x, current_epoch=self.current_epoch)
        
        # 计算 Loss (仅监控 Content Loss)
        loss, _ = self.criterion_content(y_hat, y, flow=flows, prev_frame=prev_frame, mask=y_mask)
        
        # 计算指标
        self.val_metrics(y_hat, y, y_mask)
        
        # 记录
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        测试步
        """
        meta, x, y, x_mask, y_mask = batch
        y_hat, flows = self(x, current_epoch=self.current_epoch)
        
        loss, _ = self.criterion_content(y_hat, y, mask=y_mask) # 测试时通常不需要 flow loss
        self.test_metrics(y_hat, y, y_mask)
        self.log('test_loss', loss, on_epoch=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        生产环境推理接口
        """
        if len(batch) >= 2: x = batch[1]
        else: x = batch[0]
        
        # 强制使用完全稀疏状态进行推理
        y_hat, _ = self(x, current_epoch=self.hparams.max_epochs)
        return y_hat

    def on_train_epoch_end(self):
        """
        Epoch 结束时的处理：
        1. 记录 Train Metrics
        2. 手动更新 LR Scheduler (因为 automatic_optimization=False)
        """
        # 1. Metrics
        self.log_dict(self.train_metrics.compute(), prog_bar=False)
        self.train_metrics.reset()
        
        # 2. Schedulers Step
        sch_g, sch_d = self.lr_schedulers()
        
        # Lightning 2.0+ 在手动优化模式下可能不会自动 step scheduler，
        # 如果配置为 interval='epoch'，最好手动调用
        if self.trainer.is_global_zero:
            # 仅在主进程打印一次 Debug 信息
            pass
            
        sch_g.step()
        sch_d.step()
        
        # 3. 监控稀疏率 (Debug)
        if hasattr(self.model.evolution.layers[0], '_get_curr_sparse_ratio'):
            curr_ratio = self.model.evolution.layers[0]._get_curr_sparse_ratio(self.current_epoch)
            # MLOGI(f"Epoch {self.current_epoch} | Sparse Ratio: {curr_ratio:.2f}")

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        
        # 记录关键指标
        if 'val_csi_avg' in metrics: # 假设 metrics 中有这个 key，具体取决于 MetMetricCollection 实现
             self.log('val_csi', metrics['val_csi_avg'], prog_bar=True, sync_dist=True)
        
        self.log_dict(metrics, prog_bar=False, sync_dist=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), prog_bar=False, sync_dist=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        """
        配置优化器与调度器。
        GAN 模式下需要返回两个独立的优化器 (G 和 D)。
        """
        # 1. 拆分参数
        # Generator: 主模型 + 演变网络 + 投影层 (排除 discriminator)
        # 注意：使用 named_parameters 过滤更安全
        g_params = [p for n, p in self.model.named_parameters() if "discriminator" not in n]
        
        # Discriminator: 仅判别器参数
        d_params = self.model.discriminator.parameters()
        
        # 2. 初始化优化器
        opt_g = AdamW(g_params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        opt_d = AdamW(d_params, lr=self.hparams.discriminator_lr, weight_decay=self.hparams.weight_decay)
        
        # 3. 初始化调度器 (Cosine Annealing)
        sch_g = CosineAnnealingLR(opt_g, T_max=self.hparams.max_epochs, eta_min=1e-6)
        sch_d = CosineAnnealingLR(opt_d, T_max=self.hparams.max_epochs, eta_min=1e-6)
        
        # 返回格式: (optimizers_list, schedulers_list)
        return [opt_g, opt_d], [sch_g, sch_d]# metai/model/met_mamba/trainer.py

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
    [MeteoMamba Lightning Module v2.0]
    
    功能职责：
    1. 生命周期管理：负责模型的初始化、训练、验证、测试流程。
    2. GAN 训练编排：管理生成器 (Generator) 和判别器 (Discriminator) 的交替优化。
    3. 稀疏控制：将 current_epoch 注入模型，驱动稀疏率退火。
    4. 自动/手动优化：关闭 Lightning 自动优化，手动控制 backward 和 step。
    """
    
    def __init__(
        self, 
        # --- 模型架构超参 (需与 MeteoMamba __init__ 对应) ---
        in_shape: Tuple[int, int, int] = (19, 256, 256), # (C, H, W)
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
        gan_start_epoch: int = 0,
        gan_weight: float = 0.01,
        discriminator_lr: float = 2e-4,
        
        # --- Loss 配置 ---
        use_temporal_weight: bool = True,
        
        # --- 其他 kwargs ---
        **kwargs
    ):
        super().__init__()
        # 1. 保存超参 (Checkpoints 恢复需要)
        self.save_hyperparameters()
        
        # 2. 核心模型初始化 (生成器 + 判别器)
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
        
        # 3. 损失函数初始化
        # 内容损失：MSE, CSI, Spectral, Conservation, Warp
        self.criterion_content = HybridLoss(use_temporal_weight=use_temporal_weight)
        # 对抗损失：Hinge Loss
        self.criterion_gan = GANLoss(mode='hinge')
        
        # 4. 评估指标集合
        self.train_metrics = MetMetricCollection(prefix="train_")
        self.val_metrics = MetMetricCollection(prefix="val_")
        self.test_metrics = MetMetricCollection(prefix="test_")

        # 5. [关键] 开启手动优化模式 (Manual Optimization)
        # GAN 需要交替训练，PyTorch Lightning 的自动模式不再适用
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor, current_epoch: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播包装函数。
        Returns:
            y_hat: 降水预测
            flows: 潜在空间流场
        """
        return self.model(x, current_epoch=current_epoch)

    def training_step(self, batch, batch_idx):
        """
        训练步 (Manual Optimization Loop)
        包含 Discriminator 更新 和 Generator 更新。
        """
        # 获取优化器
        opt_g, opt_d = self.optimizers()
        
        # 解包数据
        meta, x, y, x_mask, y_mask = batch
        
        # 准备物理约束所需的上下文
        prev_frame = x[:, -1:, :self.model.out_channels, ...]
        
        # 判断是否启用 GAN (预热期过后且配置开启)
        do_gan = self.hparams.use_gan and (self.current_epoch >= self.hparams.gan_start_epoch)

        # ======================================================================
        # Phase 1: 判别器训练 (Train Discriminator)
        # ======================================================================
        # 仅当启用 GAN 时才训练 D
        if do_gan:
            # 1. 生成假数据 (Detach: 不传梯度给生成器)
            with torch.no_grad():
                y_fake_detached, _ = self.model(x, current_epoch=self.current_epoch)
            
            # 2. 判别器前向
            # 真实样本得分
            logits_real = self.model.discriminator(y)
            # 假样本得分
            logits_fake = self.model.discriminator(y_fake_detached)
            
            # 3. 计算 D Loss
            loss_d = self.criterion_gan.get_disc_loss(logits_real, logits_fake)
            
            # 4. 反向传播与更新
            opt_d.zero_grad()
            self.manual_backward(loss_d)
            opt_d.step()
            
            # 记录日志
            self.log("train_loss_d", loss_d, prog_bar=True, on_step=True, on_epoch=True)

        # ======================================================================
        # Phase 2: 生成器训练 (Train Generator)
        # ======================================================================
        
        # 1. 生成数据 (需要梯度)
        # 注意：这里重新执行一次 forward 以构建计算图，或者如果显存够大，
        # 也可以在 Phase 1 前执行一次 forward 并 retain_graph=True
        y_fake, flows = self.model(x, current_epoch=self.current_epoch)
        
        # 2. 计算内容损失 (MSE, CSI, Warp...)
        loss_content, loss_dict = self.criterion_content(
            pred=y_fake, 
            target=y, 
            flow=flows, 
            prev_frame=prev_frame, 
            mask=y_mask
        )
        
        # 3. 计算对抗损失 (Adversarial Loss)
        loss_adv = torch.tensor(0.0, device=self.device)
        if do_gan:
            # 判别器对生成样本的评分
            logits_fake_g = self.model.discriminator(y_fake)
            # 生成器的目标是让 D 认为这是真的
            loss_adv = self.criterion_gan.get_gen_loss(logits_fake_g)
        
        # 4. 动态调整 GAN 权重 (可选：随时间线性增加权重)
        # 策略：在 gan_start_epoch 后，权重从 0 逐渐增加到 gan_weight
        current_gan_weight = 0.0
        if do_gan:
            # 简单的恒定权重或基于 Epoch 的调度
            current_gan_weight = self.hparams.gan_weight
        
        # 5. 总损失
        total_loss_g = loss_content + current_gan_weight * loss_adv
        
        # 6. 反向传播与更新
        opt_g.zero_grad()
        self.manual_backward(total_loss_g)
        opt_g.step()
        
        # 7. 记录日志与指标
        self.log("train_loss_g", total_loss_g, prog_bar=True, on_step=True, on_epoch=True)
        if do_gan:
            self.log("train_loss_adv", loss_adv, prog_bar=False, on_step=False, on_epoch=True)
        
        # 记录详细的内容损失分量
        self.log_dict(loss_dict, prog_bar=False, on_step=False, on_epoch=True)
        
        # 计算业务指标 (CSI, MAE...)
        self.train_metrics(y_fake.detach(), y, y_mask)

    def validation_step(self, batch, batch_idx):
        """
        验证步 (标准推理)
        """
        meta, x, y, x_mask, y_mask = batch
        prev_frame = x[:, -1:, :self.model.out_channels, ...]
        
        # 推理
        y_hat, flows = self(x, current_epoch=self.current_epoch)
        
        # 计算 Loss (仅监控 Content Loss)
        loss, _ = self.criterion_content(y_hat, y, flow=flows, prev_frame=prev_frame, mask=y_mask)
        
        # 计算指标
        self.val_metrics(y_hat, y, y_mask)
        
        # 记录
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        测试步
        """
        meta, x, y, x_mask, y_mask = batch
        y_hat, flows = self(x, current_epoch=self.current_epoch)
        
        loss, _ = self.criterion_content(y_hat, y, mask=y_mask) # 测试时通常不需要 flow loss
        self.test_metrics(y_hat, y, y_mask)
        self.log('test_loss', loss, on_epoch=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        生产环境推理接口
        """
        if len(batch) >= 2: x = batch[1]
        else: x = batch[0]
        
        # 强制使用完全稀疏状态进行推理
        y_hat, _ = self(x, current_epoch=self.hparams.max_epochs)
        return y_hat

    def on_train_epoch_end(self):
        """
        Epoch 结束时的处理：
        1. 记录 Train Metrics
        2. 手动更新 LR Scheduler (因为 automatic_optimization=False)
        """
        # 1. Metrics
        self.log_dict(self.train_metrics.compute(), prog_bar=False)
        self.train_metrics.reset()
        
        # 2. Schedulers Step
        sch_g, sch_d = self.lr_schedulers()
        
        # Lightning 2.0+ 在手动优化模式下可能不会自动 step scheduler，
        # 如果配置为 interval='epoch'，最好手动调用
        if self.trainer.is_global_zero:
            # 仅在主进程打印一次 Debug 信息
            pass
            
        sch_g.step()
        sch_d.step()
        
        # 3. 监控稀疏率 (Debug)
        if hasattr(self.model.evolution.layers[0], '_get_curr_sparse_ratio'):
            curr_ratio = self.model.evolution.layers[0]._get_curr_sparse_ratio(self.current_epoch)
            # MLOGI(f"Epoch {self.current_epoch} | Sparse Ratio: {curr_ratio:.2f}")
        
    def on_validation_epoch_end(self):
        """
        Validation Epoch 结束时的处理。
        核心任务：
        1. 计算验证集指标。
        2. 显式记录 'val_score'，供 ModelCheckpoint 监控以保存最佳模型。
        """
        # 计算所有指标 (返回字典，key 包含 prefix "val_")
        metrics = self.val_metrics.compute()
        
        # -----------------------------------------------------------
        # [修复] 显式记录 val_score 用于 Checkpoint 监控
        # -----------------------------------------------------------
        # 优先使用 metrics 中计算好的综合得分 (通常是 CSI 和 HSS 的加权)
        if 'val_total_score' in metrics:
            self.log('val_score', metrics['val_total_score'], prog_bar=True, logger=True, sync_dist=True)
            
        # 如果 metrices.py 未定义 total_score，则退化使用 CSI 均值作为 val_score
        elif 'val_csi_avg' in metrics:
            self.log('val_score', metrics['val_csi_avg'], prog_bar=True, logger=True, sync_dist=True)
            
        # -----------------------------------------------------------
        # 记录其他详细物理指标 (如各个阈值的 CSI, MAE)
        # -----------------------------------------------------------
        # 记录关键的 CSI 均值，便于 Tensorboard 直观查看
        if 'val_csi_avg' in metrics:
             self.log('val_csi', metrics['val_csi_avg'], prog_bar=True, sync_dist=True)

        # 记录所有详细指标 (不显示在进度条，只记录到 Logger)
        self.log_dict(metrics, prog_bar=False, logger=True, sync_dist=True)
        
        # 重置状态，为下一个 Epoch 做准备
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), prog_bar=False, sync_dist=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        """
        配置优化器与调度器。
        GAN 模式下需要返回两个独立的优化器 (G 和 D)。
        """
        # 1. 拆分参数
        # Generator: 主模型 + 演变网络 + 投影层 (排除 discriminator)
        # 注意：使用 named_parameters 过滤更安全
        g_params = [p for n, p in self.model.named_parameters() if "discriminator" not in n]
        
        # Discriminator: 仅判别器参数
        d_params = self.model.discriminator.parameters()
        
        # 2. 初始化优化器
        opt_g = AdamW(g_params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        opt_d = AdamW(d_params, lr=self.hparams.discriminator_lr, weight_decay=self.hparams.weight_decay)
        
        # 3. 初始化调度器 (Cosine Annealing)
        sch_g = CosineAnnealingLR(opt_g, T_max=self.hparams.max_epochs, eta_min=1e-6)
        sch_d = CosineAnnealingLR(opt_d, T_max=self.hparams.max_epochs, eta_min=1e-6)
        
        # 返回格式: (optimizers_list, schedulers_list)
        return [opt_g, opt_d], [sch_g, sch_d]