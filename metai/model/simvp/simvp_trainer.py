# metai/model/simvp_trainer.py

from typing import Any, cast, Dict, Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as l
from lightning.pytorch.utilities.types import OptimizerLRScheduler

# 导入实际依赖 (假设这些类和函数都存在于项目中)
from metai.model.core import get_optim_scheduler, timm_schedulers
from .simvp_model import SimVP_Model
from .simvp_loss import SparsePrecipitationLoss


class SimVP(l.LightningModule):
    def __init__(self, **args):
        super(SimVP, self).__init__()
        
        self.save_hyperparameters()
        config: Dict[str, Any] = dict(args)
        
        # 1. 模型初始化 (SimVP_Model)
        self.model = self._build_model(config)
        
        # 2. Loss Configuration Setup
        l1_weight = config.get('l1_weight', 0.75)
        bce_weight = config.get('bce_weight', 8.0)
        loss_threshold = config.get('loss_threshold', 0.01)
        temporal_consistency_weight = config.get('temporal_consistency_weight', 0.5)
        ssim_weight = config.get('ssim_weight', 0.3)
        
        self.use_curriculum_learning = config.get('use_curriculum_learning', True)
        self.curriculum_warmup_epochs = config.get('curriculum_warmup_epochs', 5)
        self.curriculum_transition_epochs = config.get('curriculum_transition_epochs', 10)
        
        self.target_positive_weight = config.get('positive_weight', 100.0)
        self.target_sparsity_weight = config.get('sparsity_weight', 10.0)
        
        # Threshold weights for loss
        if config.get('use_threshold_weights', True):
            precipitation_thresholds = [0.1/30.0, 1.0/30.0, 2.0/30.0, 5.0/30.0, 8.0/30.0]
            precipitation_weights = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
        else:
             precipitation_thresholds = None
             precipitation_weights = None
        
        init_pos_weight = 1.0 if self.use_curriculum_learning else self.target_positive_weight
        init_spar_weight = 0.0 if self.use_curriculum_learning else self.target_sparsity_weight
        
        ssim_weight_final = ssim_weight if config.get('use_composite_loss', True) else None
        referee_weights_w_k = config.get('referee_weights_w_k', None)
        
        # Temporal weight configuration
        temporal_weight_enabled = config.get('temporal_weight_enabled', False)
        temporal_weight_max = config.get('temporal_weight_max', 2.0)

        # 3. 初始化 Loss 函数
        self.criterion = SparsePrecipitationLoss(
            positive_weight=init_pos_weight, sparsity_weight=init_spar_weight, l1_weight=l1_weight, 
            bce_weight=bce_weight, threshold=loss_threshold,
            precipitation_thresholds=precipitation_thresholds, precipitation_weights=precipitation_weights,
            temporal_weight_enabled=temporal_weight_enabled, 
            temporal_weight_max=temporal_weight_max,
            ssim_weight=ssim_weight_final,
            temporal_consistency_weight=temporal_consistency_weight,
            referee_weights_w_k=referee_weights_w_k
        )
        
        self.test_outputs = []
        rs = config.get('resize_shape', None)
        self.resize_shape = tuple(rs) if rs is not None else None
    
    def _build_model(self, config: Dict[str, Any]):
        """实例化 SimVP 模型，使用配置中的优化参数"""
        return SimVP_Model(
             in_shape=config.get('in_shape'), hid_S=config.get('hid_S', 128), 
             hid_T=config.get('hid_T', 512), N_S=config.get('N_S', 4), N_T=config.get('N_T', 12),
             model_type=config.get('model_type', 'tau'), out_channels=config.get('out_channels', 1),
             mlp_ratio=config.get('mlp_ratio', 8.0), drop=config.get('drop', 0.0), drop_path=config.get('drop_path', 0.1),
             spatio_kernel_enc=config.get('spatio_kernel_enc', 3), 
             spatio_kernel_dec=config.get('spatio_kernel_dec', 3)
        )
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        """配置优化器和学习率调度器，使用 metai.model.core"""
        
        max_epochs = getattr(self.hparams, 'max_epochs', 100)
        
        # 假设 get_optim_scheduler 存在且可用
        optimizer, scheduler, by_epoch = get_optim_scheduler(self.hparams, max_epochs, self.model)
        
        return cast(OptimizerLRScheduler, {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "epoch" if by_epoch else "step"
            },
        })
    
    def lr_scheduler_step(self, scheduler: Any, metric: Any):
        """处理 timm 调度器的步进"""
        # 假设 timm_schedulers 存在且可用
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            scheduler.step(metric) if metric is not None else scheduler.step()
    
    def on_train_epoch_start(self):
        """课程学习：动态调整损失函数权重"""
        if not self.use_curriculum_learning:
            return
        
        epoch = self.current_epoch
        warmup = self.curriculum_warmup_epochs
        transition = self.curriculum_transition_epochs
        
        if epoch < warmup:
            pos_w, sparse_w, stage = 1.0, 0.0, "Warmup"
        elif epoch < warmup + transition:
            progress = (epoch - warmup) / transition
            pos_w = 1.0 + progress * (self.target_positive_weight - 1.0)
            sparse_w = progress * self.target_sparsity_weight
            stage = "Transition"
        else:
            pos_w, sparse_w, stage = self.target_positive_weight, self.target_sparsity_weight, "Full"
        
        # 更新 criterion 属性
        self.criterion.positive_weight = pos_w
        self.criterion.sparsity_weight = sparse_w
        
        if self.trainer is not None and self.trainer.is_global_zero and (epoch % 5 == 0 or epoch in [0, warmup, warmup + transition]):
            print(f'[Curriculum] Epoch {epoch} ({stage}): PosW={pos_w:.2f}, SparseW={sparse_w:.2f}')
            
        self.log('loss/pos_weight', pos_w, on_step=False, on_epoch=True)
        self.log('loss/sparse_weight', sparse_w, on_step=False, on_epoch=True)
    
    def forward(self, x):
        return self.model(x)
    
    def _interpolate_batch_gpu(self, batch_tensor: torch.Tensor, mode: str = 'max_pool') -> torch.Tensor:
        """高效的 GPU 批量插值/降采样函数"""
        if self.resize_shape is None: return batch_tensor
        T, C, H, W = batch_tensor.shape[1:]
        target_H, target_W = self.resize_shape
        if H == target_H and W == target_W: return batch_tensor
        
        # 检查是否为布尔类型，如果是则先转换为浮点数
        is_bool = batch_tensor.dtype == torch.bool
        if is_bool:
            batch_tensor = batch_tensor.float()
        
        B = batch_tensor.shape[0]
        batch_tensor = batch_tensor.view(B * T, C, H, W)
        
        if mode == 'max_pool':
            processed_tensor = F.adaptive_max_pool2d(batch_tensor, output_size=self.resize_shape) if target_H < H or target_W < W else F.interpolate(batch_tensor, size=self.resize_shape, mode='bilinear', align_corners=False)
        elif mode in ['nearest', 'bilinear']:
            align = False if mode == 'bilinear' else None
            processed_tensor = F.interpolate(batch_tensor, size=self.resize_shape, mode=mode, align_corners=align)
        else:
            raise ValueError(f"Unsupported interpolation mode: {mode}")

        processed_tensor = processed_tensor.view(B, T, C, target_H, target_W)
        
        # 如果原始是布尔类型，转换回布尔类型
        if is_bool:
            processed_tensor = processed_tensor.bool()
        
        return processed_tensor
    
    def training_step(self, batch, batch_idx):
        _, x, y, target_mask, _ = batch
        target_mask = target_mask.bool()

        x = self._interpolate_batch_gpu(x, mode='max_pool')
        y = self._interpolate_batch_gpu(y, mode='max_pool')
        target_mask = self._interpolate_batch_gpu(target_mask, mode='nearest')

        # 🚨 [关键修正 1]: 模型输出 Logits Z
        logits_pred = self(x)
        
        # 损失函数现在传入 Logits Z
        # SparsePrecipitationLoss 内部会处理 Sigmoid、Clamp 和 BCEWithLogitsLoss
        loss = self.criterion(logits_pred, y, target_mask=target_mask)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        metadata, x, y, target_mask, input_mask = batch
        target_mask = target_mask.bool()

        x = self._interpolate_batch_gpu(x, mode='max_pool')
        y = self._interpolate_batch_gpu(y, mode='max_pool')
        target_mask = self._interpolate_batch_gpu(target_mask, mode='nearest')
        
        logits_pred = self(x)
        
        # 计算 Pred (用于 MAE/MSE 指标记录)
        y_pred = torch.sigmoid(logits_pred)
        y_pred_clamped = torch.clamp(y_pred, 0.0, 1.0)
        
        # 损失函数传入 Logits Z
        loss = self.criterion(logits_pred, y, target_mask=target_mask)
        
        # 指标计算使用 clamped Pred
        mae = F.l1_loss(y_pred_clamped, y)

        # === 新增：计算简化的加权 TS Score ===
        # 反归一化 (假设 max=30.0, 根据您的 test 代码)
        MM_MAX = 30.0
        pred_mm = y_pred_clamped * MM_MAX
        target_mm = y * MM_MAX
        
        # 选取关键阈值 (如竞赛规则)
        thresholds = [0.01, 0.1, 1.0, 2.0, 5.0, 8.0] 
        weights = [0.1, 0.1, 0.1, 0.2, 0.2, 0.3] # 给予强降水更高权重
        ts_sum = 0.0
        
        for t, w in zip(thresholds, weights):
            # 计算 TS
            hits = ((pred_mm >= t) & (target_mm >= t)).float().sum()
            misses = ((pred_mm < t) & (target_mm >= t)).float().sum()
            false_alarms = ((pred_mm >= t) & (target_mm < t)).float().sum()
            ts = hits / (hits + misses + false_alarms + 1e-6)
            ts_sum += ts * w
            
        # 记录加权 TS 作为验证指标 (越大越好)
        val_score = ts_sum / sum(weights)

        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_mae', mae, on_epoch=True, sync_dist=True)
        self.log('val_score', val_score, on_epoch=True, prog_bar=True, sync_dist=True)
    
    def on_test_epoch_start(self):
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        metadata, x, y, target_mask, input_mask = batch
        target_mask = target_mask.bool()

        x = self._interpolate_batch_gpu(x, mode='max_pool')
        y = self._interpolate_batch_gpu(y, mode='max_pool')
        target_mask = self._interpolate_batch_gpu(target_mask, mode='nearest')

        # 🚨 [关键修正 3]: 模型输出 Logits Z
        logits_pred = self(x)
        y_pred = torch.sigmoid(logits_pred)
        y_pred_clamped = torch.clamp(y_pred, 0.0, 1.0)
        
        with torch.no_grad():
            # 损失函数传入 Logits Z
            loss = self.criterion(logits_pred, y, target_mask=target_mask)
            
        try:
            self.log('test_loss', loss, on_epoch=True)
        except RuntimeError:
            pass
        
        result = {
            # 输出仍使用 [0, 1] 范围的预测值
            'inputs': x[0].cpu().float().numpy(),
            'preds': y_pred_clamped[0].cpu().float().numpy(),
            'trues': y[0].cpu().float().numpy()
        }
        
        self.test_outputs.append(result)
        return result
    
    def infer_step(self, batch, batch_idx):
        metadata, x, input_mask = batch 
        
        x = self._interpolate_batch_gpu(x, mode='max_pool')
        # 🚨 [关键修正 4]: 推理时输出 Pred
        logits_pred = self(x)
        y_pred = torch.sigmoid(logits_pred)
        return torch.clamp(y_pred, 0.0, 1.0)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.infer_step(batch, batch_idx)