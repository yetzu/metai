import lightning as l
import torch
import torch.nn.functional as F
from .model import MeteoMamba
from .loss import HybridLoss
from metai.model.core import get_optim_scheduler

class MeteoMambaModule(l.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = MeteoMamba(
            in_shape=self.hparams.in_shape,
            hid_S=self.hparams.hid_S,
            hid_T=self.hparams.hid_T,
            N_S=self.hparams.N_S,
            N_T=self.hparams.N_T,
            aft_seq_length=self.hparams.aft_seq_length,
            out_channels=1
        )
        
        self.criterion = HybridLoss(
            focal_weight=self.hparams.loss_weight_focal,
            evo_weight=self.hparams.loss_weight_evo,
            ssim_weight=self.hparams.loss_weight_ssim,
            csi_weight=self.hparams.loss_weight_csi
        )
        
        self.resize_shape = (self.hparams.in_shape[2], self.hparams.in_shape[3])
        
        # [验证配置] 定义竞赛标准的阈值和权重 (与 Loss 对齐)
        self.MM_MAX = 30.0
        # 阈值: 0.1(微量), 1.0(小), 2.0(中), 5.0(大), 8.0(暴)
        self.val_thresholds = [0.1, 1.0, 2.0, 5.0, 8.0] 
        # 权重: 强降水权重更高
        weights_raw = [0.1, 0.1, 0.2, 0.25, 0.35]
        total_w = sum(weights_raw)
        self.val_weights = [w / total_w for w in weights_raw]

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

    def on_train_epoch_start(self):
        """课程学习策略"""
        if not self.hparams.use_curriculum_learning:
            return

        epoch = self.current_epoch
        max_epochs = self.hparams.max_epochs
        
        phase_1_end = int(0.2 * max_epochs)
        phase_2_end = int(0.6 * max_epochs)
        
        weights = {}
        phase_name = ""

        if epoch < phase_1_end:
            # Phase 1: Distribution
            weights = {'focal': 10.0, 'evo': 0.0, 'ssim': 0.0, 'csi': 0.0}
            phase_name = "Phase 1: Distribution (Focal)"
        elif epoch < phase_2_end:
            # Phase 2: Physics
            p = (epoch - phase_1_end) / (phase_2_end - phase_1_end)
            weights = {
                'focal': 10.0 - p * 5.0, 
                'evo': p * 1.0,          
                'ssim': p * 0.5,         
                'csi': 0.0
            }
            phase_name = f"Phase 2: Physics [p={p:.2f}]"
        else:
            # Phase 3: Metric
            p = (epoch - phase_2_end) / (max_epochs - phase_2_end)
            weights = {
                'focal': 5.0 - p * 4.0,  
                'evo': 1.0 - p * 0.5,    
                'ssim': 0.5,             
                'csi': p * 5.0           
            }
            phase_name = f"Phase 3: Metric [p={p:.2f}]"

        self.criterion.weights.update(weights)
        
        if self.trainer.is_global_zero:
            print(f"\n[Curriculum] Epoch {epoch}/{max_epochs} | {phase_name}")
            print(f"             Weights: {weights}")
            
        for k, v in weights.items():
            self.log(f"train/w_{k}", v, on_epoch=True, sync_dist=True)

    def _interpolate_batch(self, x, mode='bilinear'):
        if self.resize_shape is None: return x
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        if mode == 'nearest':
            x = F.interpolate(x, size=self.resize_shape, mode='nearest')
        else:
            x = F.interpolate(x, size=self.resize_shape, mode='bilinear', align_corners=False)
        return x.view(B, T, C, *self.resize_shape)

    def training_step(self, batch, batch_idx):
        _, x, y, t_mask, _ = batch
        x = self._interpolate_batch(x, mode='bilinear')
        y = self._interpolate_batch(y, mode='bilinear')
        t_mask = self._interpolate_batch(t_mask.float(), mode='nearest')
        
        logits = self.model(x)
        loss, loss_dict = self.criterion(logits, y, mask=t_mask)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in loss_dict.items():
            if k != 'total':
                self.log(f'loss/{k}', v, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        🚀 [SOTA] 严谨的验证步骤
        计算全阈值加权 CSI，真实反映竞赛成绩。
        """
        _, x, y, t_mask, _ = batch
        
        # 1. 插值与预处理
        x = self._interpolate_batch(x, mode='bilinear')
        y = self._interpolate_batch(y, mode='bilinear')
        t_mask = self._interpolate_batch(t_mask.float(), mode='nearest')
        
        # 2. 计算 Loss (用于监控收敛)
        logits = self.model(x)
        loss, _ = self.criterion(logits, y, mask=t_mask)
        
        # 3. 预测值处理
        pred = torch.sigmoid(logits)
        pred = torch.clamp(pred, 0.0, 1.0)
        
        # 4. 计算 MAE (仅有效区域)
        if t_mask.sum() > 0:
            mae = (torch.abs(pred - y) * t_mask).sum() / (t_mask.sum() + 1e-6)
        else:
            mae = 0.0
            
        # 5. [关键] 计算全阈值加权 CSI (Rigorous Metric)
        # 还原到物理空间 (mm) 进行判定
        pred_mm = pred * self.MM_MAX
        target_mm = y * self.MM_MAX
        
        weighted_csi_sum = 0.0
        
        # 确保 Mask 为布尔值
        valid_mask = t_mask > 0.5
        
        for i, threshold in enumerate(self.val_thresholds):
            # 二值化 (Hits, Misses, False Alarms)
            # 必须在 Mask 有效区域内计算
            hits = ((pred_mm >= threshold) & (target_mm >= threshold) & valid_mask).float().sum()
            # Union = Hits + Misses + FalseAlarms
            #       = (Pred >= T) | (Target >= T)
            union = (((pred_mm >= threshold) | (target_mm >= threshold)) & valid_mask).float().sum()
            
            csi = hits / (union + 1e-6)
            
            # 加权累加
            weighted_csi_sum += csi * self.val_weights[i]
            
            # 记录分项指标 (可选，用于详细分析)
            # self.log(f'val_csi_t{threshold}', csi, on_epoch=True, sync_dist=True)

        # 6. 记录日志
        self.log('val_loss', loss, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('val_mae', mae, on_epoch=True, sync_dist=True)
        self.log('val_score', weighted_csi_sum, on_epoch=True, sync_dist=True, prog_bar=True)