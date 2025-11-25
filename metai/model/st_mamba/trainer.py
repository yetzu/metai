import lightning as l
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from metai.model.core import get_optim_scheduler, timm_schedulers
from .model import MeteoMamba
from .loss import HybridLoss

class MeteoMambaModule(l.LightningModule):
    def __init__(
        self, 
        in_shape: Tuple[int, int, int, int] = (10, 31, 256, 256),
        hid_S: int = 64,
        hid_T: int = 256,
        N_S: int = 4,
        N_T: int = 8,
        aft_seq_length: int = 20,
        # 优化器参数 (参数化，对应 config.py)
        lr: float = 5e-4,
        min_lr: float = 1e-5,
        warmup_lr: float = 1e-5,
        warmup_epoch: int = 10,
        weight_decay: float = 0.05,
        momentum: float = 0.9,
        opt: str = 'adamw',
        sched: str = 'cosine',
        max_epochs: int = 100,
        decay_epoch: int = 30,
        decay_rate: float = 0.1,
        # 策略参数
        use_curriculum_learning: bool = True,
        # 损失权重参数
        loss_weight_l1: float = 1.0,
        loss_weight_ssim: float = 0.5,
        loss_weight_csi: float = 1.0,
        loss_weight_spectral: float = 0.1,
        loss_weight_evo: float = 0.5,
        loss_weight_focal: float = 0.0,
        **kwargs 
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = MeteoMamba(
            in_shape=in_shape,
            hid_S=hid_S,
            hid_T=hid_T,
            N_S=N_S,
            N_T=N_T,
            aft_seq_length=aft_seq_length,
            out_channels=1
        )
        
        # [Fix] 使用传入的参数初始化 Loss，而非硬编码
        self.criterion = HybridLoss(
            l1_weight=loss_weight_l1,
            ssim_weight=loss_weight_ssim,
            csi_weight=loss_weight_csi,
            spectral_weight=loss_weight_spectral,
            evo_weight=loss_weight_evo,
            focal_weight=loss_weight_focal # [New]
        )
        
        self.resize_shape = (in_shape[2], in_shape[3])

    def configure_optimizers(self):
        # [Fix] 移除硬编码的 Args 类，直接使用 self.hparams (它包含了 __init__ 中的所有参数)
        # self.hparams 还会自动包含 config.py 中定义的额外参数
        
        # 确保 hparams 中包含 filter_bias_and_bn 标志
        if not hasattr(self.hparams, 'filter_bias_and_bn'):
             self.hparams.filter_bias_and_bn = True

        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.hparams, 
            self.hparams.max_epochs, 
            self.model
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch" if by_epoch else "step"
            }
        }

    def lr_scheduler_step(self, scheduler: Any, metric: Any):
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            scheduler.step(metric) if metric is not None else scheduler.step()

    def on_train_epoch_start(self):
        """
        Curriculum Learning 策略调整
        策略：
        - Phase 1 (Structure): L1 主导，Focal=0，快速收敛形状。
        - Phase 2 (Physics): 引入 Focal 和 Evo，开始关注强回波和物理演变。
        - Phase 3 (Metric): Focal + CSI 主导，攻坚强降水极值，并冻结 Encoder。
        """
        if not self.hparams.use_curriculum_learning:
            return

        epoch = self.current_epoch
        max_epochs = self.hparams.max_epochs
        
        phase_1_end = int(0.1 * max_epochs) # 0-10%
        phase_2_end = int(0.4 * max_epochs) # 10-40%
        
        weights = {}
        
        if epoch < phase_1_end:
            # Phase 1: Structure
            weights = {'l1': 10.0, 'ssim': 1.0, 'evo': 0.0, 'spec': 0.0, 'csi': 0.0, 'focal': 0.0}
            
        elif epoch < phase_2_end:
            # Phase 2: Physics Warmup
            p = (epoch - phase_1_end) / (phase_2_end - phase_1_end)
            weights = {
                'l1': 10.0 - p * 5.0,   # L1 降低
                'ssim': 1.0,
                'evo': p * 0.1,
                'spec': p * 0.05,
                'csi': p * 0.5,
                'focal': p * 5.0        # [New] Focal 线性预热到 5.0
            }
            
        else:
            # Phase 3: Metric Sprint (全速冲刺指标)
            p = (epoch - phase_2_end) / (max_epochs - phase_2_end)
            weights = {
                'l1': 5.0 - p * 4.0,    # L1 降至 1.0
                'ssim': 1.0 - p * 0.5,
                'evo': 0.1 + p * 0.4,   # 增加物理约束
                'spec': 0.05 + p * 0.15,
                'csi': 0.5 + (10.0 - 0.5) * (p**2), # [强化] CSI 权重极大增加，主攻评分
                'focal': 5.0 + p * 5.0 
            }

            # [修改] 仅仅降低 Encoder 学习率而非完全冻结，或者只冻结最底层的 Stem
            # 这里演示冻结 Stem (3D 卷积部分)，保留 2D ConvSC 微调
            if not getattr(self, 'stem_frozen', False):
                print(f"\n[Curriculum] Epoch {epoch}: Phase 3. Freezing 3D Stem only.")
                if hasattr(self.model, 'enc') and hasattr(self.model.enc, 'stem'):
                    self.model.enc.stem.eval()
                    for param in self.model.enc.stem.parameters():
                        param.requires_grad = False
                self.stem_frozen = True
                
        # 更新 Loss 权重
        if hasattr(self, 'criterion') and hasattr(self.criterion, 'weights'):
            self.criterion.weights.update(weights)
            
        # 记录日志
        for k, v in weights.items():
            self.log(f"train/weight_{k}", v, on_epoch=True, sync_dist=True)

    def _interpolate_batch(self, batch_tensor: torch.Tensor, mode: str = 'max_pool') -> torch.Tensor:
        if self.resize_shape is None: return batch_tensor
        T, C, H, W = batch_tensor.shape[1:]
        target_H, target_W = self.resize_shape
        if H == target_H and W == target_W: return batch_tensor
        
        is_bool = batch_tensor.dtype == torch.bool
        if is_bool: batch_tensor = batch_tensor.float()
        
        B = batch_tensor.shape[0]
        batch_tensor = batch_tensor.view(B * T, C, H, W)
        
        if mode == 'max_pool':
            if target_H < H or target_W < W:
                processed_tensor = F.adaptive_max_pool2d(batch_tensor, output_size=self.resize_shape)
            else:
                processed_tensor = F.interpolate(batch_tensor, size=self.resize_shape, mode='bilinear', align_corners=False)
        elif mode in ['nearest', 'bilinear']:
            align = False if mode == 'bilinear' else None
            processed_tensor = F.interpolate(batch_tensor, size=self.resize_shape, mode=mode, align_corners=align)
        else:
            raise ValueError(f"Unsupported interpolation mode: {mode}")

        processed_tensor = processed_tensor.view(B, T, C, target_H, target_W)
        if is_bool: processed_tensor = processed_tensor.bool()
        return processed_tensor

    def training_step(self, batch, batch_idx):
        _, x, y, mask, _ = batch
        x = self._interpolate_batch(x, mode='max_pool')
        y = self._interpolate_batch(y, mode='max_pool')
        mask = self._interpolate_batch(mask, mode='nearest')
        
        pred = self.model(x)
        loss, loss_dict = self.criterion(pred, y, mask)
        
        self.log("train_loss", loss, prog_bar=True)
        for k, v in loss_dict.items():
            if k != 'total': self.log(f"train_loss_{k}", v)
        return loss

    def validation_step(self, batch, batch_idx):
        _, x, y, mask, _ = batch
        x = self._interpolate_batch(x, mode='max_pool')
        y = self._interpolate_batch(y, mode='max_pool')
        mask = self._interpolate_batch(mask, mode='nearest')
        
        logits = self.model(x)
        loss, _ = self.criterion(logits, y, mask)
        
        pred = torch.sigmoid(logits)
        y_pred_clamped = torch.clamp(pred, 0.0, 1.0)
        
        if y_pred_clamped.dim() == 5: y_pred_clamped = y_pred_clamped.squeeze(2)
        if y.dim() == 5: y = y.squeeze(2)

        mae = F.l1_loss(y_pred_clamped, y)
        
        MM_MAX = 30.0
        pred_mm = y_pred_clamped * MM_MAX
        target_mm = y * MM_MAX
        thresholds = [0.01, 0.1, 1.0, 2.0, 5.0, 8.0] 
        weights =    [0.1,  0.1, 0.1, 0.2, 0.2, 0.3]
        ts_sum = 0.0
        
        for t, w in zip(thresholds, weights):
            hits = ((pred_mm >= t) & (target_mm >= t)).float().sum()
            misses = ((pred_mm < t) & (target_mm >= t)).float().sum()
            false_alarms = ((pred_mm >= t) & (target_mm < t)).float().sum()
            ts = hits / (hits + misses + false_alarms + 1e-6)
            ts_sum += ts * w
            
        val_score = ts_sum / sum(weights)
        
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        self.log("val_mae", mae, on_epoch=True, sync_dist=True)
        self.log("val_score", val_score, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        _, x, y, target_mask, input_mask = batch
        x = self._interpolate_batch(x, mode='max_pool')
        y = self._interpolate_batch(y, mode='max_pool')
        target_mask = self._interpolate_batch(target_mask, mode='nearest')
        
        logits = self.model(x)
        y_pred = torch.clamp(torch.sigmoid(logits), 0.0, 1.0)
        
        loss, _ = self.criterion(logits, y, mask=target_mask)
        self.log('test_loss', loss, on_epoch=True)
        
        return {
            'inputs': x[0, :, 0].cpu().float().numpy(), 
            'preds': y_pred[0].squeeze().cpu().float().numpy(), 
            'trues': y[0].squeeze().cpu().float().numpy() 
        }
    
    def infer_step(self, batch, batch_idx):
        metadata, x, input_mask = batch 
        x = self._interpolate_batch(x, mode='max_pool')
        logits_pred = self(x)
        y_pred = torch.sigmoid(logits_pred)
        return torch.clamp(y_pred, 0.0, 1.0)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.infer_step(batch, batch_idx)