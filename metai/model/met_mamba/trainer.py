# metai/model/met_mamba/trainer.py

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, List, Union
from lightning.pytorch import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# 引入项目模块
from metai.model.met_mamba.model import MeteoMamba
from metai.model.met_mamba.loss import HybridLoss
from metai.model.met_mamba.metrices import MetMetricCollection
from metai.utils import MLOGI

class MeteoMambaModule(LightningModule):
    """
    [MeteoMamba Lightning Module]
    
    功能职责：
    1. 生命周期管理：负责模型的初始化、训练、验证、测试流程。
    2. 数据流转：处理 Batch 数据解包，构建物理约束所需的上下文（如 prev_frame）。
    3. 优化策略：配置 AdamW 优化器与 Cosine 退火调度器。
    4. 稀疏控制：将 Trainer 的 current_epoch 状态注入模型，驱动稀疏率动态变化。
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
        
        # --- Loss 配置 ---
        use_temporal_weight: bool = True,
        
        # --- 其他 kwargs (用于兼容 CLI) ---
        **kwargs
    ):
        super().__init__()
        # 保存超参到 self.hparams，便于 Checkpoint 恢复
        self.save_hyperparameters()
        
        # 1. 初始化核心模型 (MeteoMamba)
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
        
        # 2. 初始化损失函数 (Physics-Informed Hybrid Loss)
        # 包含：MSE, CSI, Spectral, Conservation, Warp
        self.criterion = HybridLoss(use_temporal_weight=use_temporal_weight)
        
        # 3. 初始化评估指标集合 (CSI, HSS, MAE 等)
        # 分别用于训练、验证、测试阶段
        self.train_metrics = MetMetricCollection(prefix="train_")
        self.val_metrics = MetMetricCollection(prefix="val_")
        self.test_metrics = MetMetricCollection(prefix="test_")

    def forward(self, x: torch.Tensor, current_epoch: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播包装函数。
        
        Args:
            x: 输入张量 [B, T_in, C, H, W]
            current_epoch: 当前 Epoch 索引，用于控制模型内部的稀疏率退火。
            
        Returns:
            y_hat: 降水预测 [B, T_out, C_out, H, W]
            flows: 潜在空间流场 [B, T_out, 2, H, W] (用于 Warp Loss)
        """
        return self.model(x, current_epoch=current_epoch)

    def _shared_step(self, batch, batch_idx, stage: str):
        """
        通用的 Step 逻辑 (Train/Val/Test 共用)。
        负责数据解包、前向计算、构建物理约束上下文、计算 Loss 及指标。
        """
        # 1. 解包数据 (来自 DataLoader)
        # meta: 元数据列表
        # x: 历史观测 [B, T_in, C_in, H, W]
        # y: 未来真值 [B, T_out, C_out, H, W]
        # x_mask/y_mask: 有效性掩码 (Bool)
        meta, x, y, x_mask, y_mask = batch
        
        # 2. 前向传播
        # [关键] 传入 self.current_epoch 以驱动 STMambaBlock 中的稀疏率退火
        # [关键] 获取 flows 用于后续的显式物理约束 (Warp Loss)
        y_hat, flows = self(x, current_epoch=self.current_epoch)
        
        # 3. 构建物理约束所需的 '前一帧' (Previous Frame / Source Frame)
        # Warp Loss 需要将 '前一帧' 根据 '流场' 推演到 '当前帧'。
        # 对于预测序列的第一帧，其 '前一帧' 是输入序列 x 的最后一帧。
        prev_frame = x[:, -1:].detach() # [B, 1, C_in, H, W]
        
        # 输入 x 可能包含多模态通道 (如 Radar, NWP, GIS)，通道数 C_in 可能 > C_out。
        # 我们只提取与预测目标 (y) 对应的通道 (例如只取前 out_channels 个通道作为降水数据)。
        if prev_frame.shape[2] > self.model.out_channels:
            prev_frame = prev_frame[:, :, :self.model.out_channels, ...]
            
        # 4. 计算 Loss (包含 MSE, CSI, Spectral, Conservation, Warp)
        loss, loss_dict = self.criterion(
            pred=y_hat, 
            target=y, 
            flow=flows,        # [新增] 传入流场
            prev_frame=prev_frame, # [新增] 传入推演源
            mask=y_mask
        )
        
        # 5. 计算并记录业务指标 (Metrics)
        # 注意：Metrics 计算通常仅基于预测值和真实值，不涉及流场
        if stage == 'train':
            self.train_metrics(y_hat, y, y_mask)
            # 训练阶段：记录详细的 Loss 分量，并在进度条显示
            self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            
        elif stage == 'val':
            self.val_metrics(y_hat, y, y_mask)
            # 验证阶段：只记录 Total Loss，避免日志过于杂乱
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
        elif stage == 'test':
            self.test_metrics(y_hat, y, y_mask)
            self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True)
            
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'test')
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        生产环境预测接口。
        
        [修正] 显式传入 max_epochs，强制模型处于'完全成熟'的稀疏状态，
        确保推理速度最快，且与测试集表现一致。
        """
        # 解包逻辑需适配你的 Dataset collate_fn 返回值
        # 假设 batch = (meta, x, y, x_mask, y_mask) 或 (meta, x, x_mask)
        if len(batch) >= 2:
            x = batch[1]
        else:
            x = batch[0]
            
        # 强制使用最终的稀疏率进行推理
        y_hat, flows = self(x, current_epoch=self.hparams.max_epochs)
        
        return y_hat

    def on_train_epoch_end(self):
        """
        每个 Training Epoch 结束时的回调。
        1. 计算并记录全量 Train Metrics。
        2. 重置 Metrics 状态。
        3. (可选) 打印当前稀疏率，确认退火状态。
        """
        # 计算并记录指标
        self.log_dict(self.train_metrics.compute(), prog_bar=False, logger=True)
        self.train_metrics.reset()
        
        # 监控稀疏率退火状态 (从模型第一层获取示例)
        # 这是一个很好的 Debug 信息，确认 Schedule 是否按预期工作
        if hasattr(self.model.evolution.layers[0], '_get_curr_sparse_ratio'):
            curr_ratio = self.model.evolution.layers[0]._get_curr_sparse_ratio(self.current_epoch)
            MLOGI(f"Epoch {self.current_epoch} | Current Sparse Ratio: {curr_ratio:.2f}")
        
    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        
        # 1. 核心指标映射 (用于 Checkpoint)
        if 'val_total_score' in metrics:
            self.log('val_score', metrics['val_total_score'], prog_bar=True, logger=True, sync_dist=True)
            
        # 2. [新增] 关键物理子指标显式记录 (便于 TensorBoard 分析)
        # 假设 MetScore 返回了 val_ts_levels (Tensor)，我们取平均值作为 val_csi
        if 'val_ts_levels' in metrics:
             # ts_levels 通常是 [0.1, 1.0, 5.0] 等阈值的 TS，取均值代表整体 CSI
            avg_csi = metrics['val_ts_levels'].mean()
            self.log('val_csi', avg_csi, prog_bar=True, logger=True, sync_dist=True)

        if 'val_mae_levels' in metrics:
            avg_mae = metrics['val_mae_levels'].mean()
            self.log('val_mae', avg_mae, prog_bar=True, logger=True, sync_dist=True)
        
        # 3. 记录所有原始指标
        self.log_dict(metrics, prog_bar=False, logger=True, sync_dist=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), prog_bar=False, logger=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        """
        配置优化器与学习率调度器。
        标准配置：AdamW + CosineAnnealingLR。
        """
        # 1. 优化器
        optimizer = AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        
        # 2. 调度器 (余弦退火)
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=self.hparams.max_epochs, 
            eta_min=1e-6 # 最小学习率
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch", # 每个 Epoch 更新一次 LR
                "frequency": 1,
            },
        }