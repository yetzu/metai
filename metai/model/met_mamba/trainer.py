# metai/model/met_mamba/trainer.py

import torch
from typing import Tuple, Optional, Any, List
from lightning.pytorch import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from metai.model.met_mamba.model import MeteoMamba
from metai.model.met_mamba.loss import HybridLoss
from metai.model.met_mamba.metrices import MetMetricCollection

class MeteoMambaModule(LightningModule):
    """
    MeteoMamba 训练模块 (LightningModule)。
    全生命周期管理：训练(Fit)、验证(Val)、测试(Test)、推理(Predict)。
    """
    def __init__(
        self, 
        # --- 模型架构参数 ---
        in_shape: Tuple[int, int, int] = (31, 256, 256),
        in_seq_len: int = 10,
        out_seq_len: int = 20,
        out_channels: int = 1,
        hid_S: int = 128,
        hid_T: int = 512,
        N_S: int = 4,
        N_T: int = 8,
        mamba_d_state: int = 32,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        
        # --- 优化参数 ---
        lr: float = 8e-4,
        weight_decay: float = 1e-2,
        max_epochs: int = 50,
        
        # --- Loss 与 策略参数 ---
        weight_mae: float = 10.0,
        weight_csi: float = 1.0,
        weight_corr: float = 1.0,
        kl_weight_max: float = 0.01, # KL Loss 的最大权重系数
        kl_anneal_epochs: int = 10,  # KL 权重退火周期 (Epochs)
        
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = MeteoMamba(
            in_shape=in_shape,
            in_seq_len=in_seq_len,
            out_seq_len=out_seq_len,
            out_channels=out_channels,
            hid_S=hid_S,
            hid_T=hid_T,
            N_S=N_S,
            N_T=N_T,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            **kwargs
        )
        
        # 混合内容损失
        self.criterion_content = HybridLoss(
            weight_mae=weight_mae, 
            weight_csi=weight_csi, 
            weight_corr=weight_corr
        )
        
        # 指标收集器
        self.train_metrics = MetMetricCollection(prefix="train_")
        self.val_metrics = MetMetricCollection(prefix="val_")
        self.test_metrics = MetMetricCollection(prefix="test_")

    def forward(self, x: torch.Tensor, y_target=None):
        """
        统一推理接口。
        """
        return self.model(x, y_target=y_target)

    # ==========================================================================
    # 训练循环 (Training Loop)
    # ==========================================================================
    def training_step(self, batch, batch_idx):
        meta, x, y, x_mask, y_mask = batch
        
        # 1. Forward (Train Mode: Posterior Guided)
        # 训练时传入 y，模型利用 Q(z|X,Y) 学习隐变量分布
        y_pred, flows, kl_loss = self.model(x, y_target=y)
        
        # 2. Content Loss
        content_loss, loss_dict = self.criterion_content(
            pred=y_pred, target=y, mask=y_mask
        )
        
        # 3. KL Annealing (退火)
        anneal_ratio = min(1.0, self.current_epoch / self.hparams.kl_anneal_epochs)
        kl_weight = self.hparams.kl_weight_max * anneal_ratio
        
        # 4. Total Loss
        total_loss = content_loss + kl_weight * kl_loss
        
        # 5. Logging
        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        loss_dict['kl_loss'] = kl_loss
        loss_dict['kl_weight'] = kl_weight
        self.log_dict(loss_dict, prog_bar=False, on_step=False, on_epoch=True)
        
        # Metrics
        self.train_metrics(y_pred.detach(), y, y_mask)
        return total_loss

    # ==========================================================================
    # 验证循环 (Validation Loop)
    # ==========================================================================
    def validation_step(self, batch, batch_idx):
        meta, x, y, x_mask, y_mask = batch
        
        # 1. Forward (Val Mode: Prior Sampling)
        # 关键: y_target=None, 强制使用 P(z|X) 进行预测，评估真实生成能力
        y_hat, flows = self.model(x, y_target=None)
        
        # 2. Loss & Metrics
        loss, _ = self.criterion_content(y_hat, y, mask=y_mask)
        
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.val_metrics(y_hat, y, y_mask)
        return loss

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        metrics = {k: v.to(self.device) for k, v in metrics.items()}
        scalar_metrics = {k: v for k, v in metrics.items() if v.numel() == 1}
        
        val_score = scalar_metrics.get('val_total_score', None)
        if val_score is not None:
            self.log('val_score', val_score, prog_bar=True, logger=True, sync_dist=True)

        self.log_dict(scalar_metrics, prog_bar=False, logger=True, sync_dist=True)
        self.val_metrics.reset()

    # ==========================================================================
    # 测试循环 (Test Loop)
    # ==========================================================================
    def test_step(self, batch, batch_idx):
        """
        测试步: 执行严格的评估。
        """
        meta, x, y, x_mask, y_mask = batch
        
        # 1. Inference (Prior Sampling)
        # 确保使用 Prior 采样，且不泄露未来信息
        y_hat, flows = self.model(x, y_target=None)
        
        # 2. Metrics (CSI, HSS, RMSE等)
        # test_metrics 会累积所有 batch 的结果
        self.test_metrics(y_hat, y, y_mask)
        
        # 3. Log Test Loss
        loss, _ = self.criterion_content(y_hat, y, mask=y_mask)
        self.log('test_loss', loss, on_epoch=True, sync_dist=True)
        
        return loss

    def on_test_epoch_end(self):
        """测试结束: 输出最终指标"""
        metrics = self.test_metrics.compute()
        metrics = {k: v.to(self.device) for k, v in metrics.items()}
        
        # 打印或记录所有详细指标
        self.log_dict(metrics, prog_bar=True, logger=True, sync_dist=True)
        print("\n=== Final Test Metrics ===")
        for k, v in metrics.items():
            if v.numel() == 1:
                print(f"{k}: {v.item():.4f}")
        print("==========================\n")
        
        self.test_metrics.reset()

    # ==========================================================================
    # 推理循环 (Prediction Loop)
    # ==========================================================================
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        推理步: 用于生成提交文件或可视化。
        兼容无 Label 的数据加载器。
        """
        # 适配不同的 Collate fn 返回值
        if len(batch) == 5:
            meta, x, y, x_mask, y_mask = batch
        elif len(batch) == 3:
            meta, x, x_mask = batch
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}")
            
        # 1. Inference
        y_hat, flows = self.model(x, y_target=None)
        
        # 2. Return results
        # 返回元数据以便后续将 Tensor 对应回文件名
        return {
            "meta": meta,
            "pred": y_hat,
            "flows": flows # 可选：返回光流用于分析运动轨迹
        }

    # ==========================================================================
    # 优化器配置
    # ==========================================================================
    def configure_optimizers(self):
        params = self.model.parameters()
        opt = AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = CosineAnnealingLR(opt, T_max=self.hparams.max_epochs, eta_min=1e-6)
        
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "interval": "epoch",
                "frequency": 1
            }
        }