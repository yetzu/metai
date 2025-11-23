# metai/model/simvp_trainer.py

import subprocess
import os
import sys
from typing import Any, cast, Dict, Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as l
from lightning.pytorch.utilities.types import OptimizerLRScheduler

# 导入实际依赖 (假设这些类和函数都存在于项目中)
from metai.model.core import get_optim_scheduler, timm_schedulers
from .simvp_model import SimVP_Model
from .simvp_loss import HybridLoss


class SimVP(l.LightningModule):
    def __init__(self, **args):
        super(SimVP, self).__init__()
        
        self.save_hyperparameters()
        config: Dict[str, Any] = dict(args)
        
        # 1. 模型初始化 (SimVP_Model)
        self.model = self._build_model(config)
        
        # 2. Loss Configuration Setup (HybridLoss 参数，统一使用 loss_weight_ 前缀)
        loss_weight_l1 = config.get('loss_weight_l1', 1.0)
        loss_weight_ssim = config.get('loss_weight_ssim', 0.5)
        loss_weight_csi = config.get('loss_weight_csi', 1.0)
        loss_weight_spectral = config.get('loss_weight_spectral', 0.1)
        loss_weight_evo = config.get('loss_weight_evo', 0.5)

        # 3. 初始化 Loss 函数
        self.criterion = HybridLoss(
            l1_weight=loss_weight_l1,
            ssim_weight=loss_weight_ssim,
            csi_weight=loss_weight_csi,
            spectral_weight=loss_weight_spectral,
            evo_weight=loss_weight_evo
        )
        
        rs = config.get('resize_shape', None)
        self.resize_shape = tuple(rs) if rs is not None else None

        # 测试相关配置
        self.auto_test_after_epoch = config.get('auto_test_after_epoch', True)
        self.test_script_path = config.get('test_script_path', None)
        # 如果没有指定脚本路径，尝试自动查找
        if self.test_script_path is None:
            # 尝试从项目根目录查找脚本
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            script_path = os.path.join(current_dir, 'run.scwds.simvp.sh')
            if os.path.exists(script_path):
                self.test_script_path = script_path
            else:
                # 如果找不到，使用相对路径
                self.test_script_path = 'run.scwds.simvp.sh'
    
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
        """训练 epoch 开始时的回调（HybridLoss 不支持课程学习，此方法保留为空）"""
        pass

    def on_train_epoch_end(self):
        """在每个训练epoch结束后执行测试脚本（后台执行，不阻塞训练）"""
        # 只在主进程执行（避免多GPU时重复执行）
        if self.trainer.is_global_zero and self.auto_test_after_epoch:
            try:
                # 检查脚本路径是否有效
                if not self.test_script_path:
                    print("[WARNING] Test script path not configured, skipping auto test")
                    return
                
                # 获取脚本的绝对路径
                if os.path.isabs(self.test_script_path):
                    script_path = str(self.test_script_path)
                else:
                    # 尝试从项目根目录查找
                    current_file = os.path.abspath(__file__)
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
                    script_path = os.path.join(project_root, str(self.test_script_path))
                
                if not os.path.exists(script_path):
                    print(f"[WARNING] Test script not found: {script_path}, skipping auto test")
                    return
                
                # 创建日志文件路径（保存测试输出）
                script_dir = os.path.dirname(script_path) or os.getcwd()
                log_dir = os.path.join(script_dir, 'test_logs')
                os.makedirs(log_dir, exist_ok=True)
                
                epoch = self.current_epoch
                log_file = os.path.join(log_dir, f'test_epoch_{epoch:03d}.log')
                
                print(f"\n[INFO] Epoch {epoch} completed. Running test script in background: {script_path}")
                print(f"[INFO] Test output will be saved to: {log_file}")
                
                # 后台执行测试脚本，输出重定向到日志文件
                # 打开文件用于写入（覆盖模式，每个epoch只调用一次）
                # 子进程会继承文件描述符，即使父进程关闭文件，子进程仍可继续写入
                log_fd = open(log_file, 'w')
                try:
                    process = subprocess.Popen(
                        ['bash', script_path, 'test'],
                        stdout=log_fd,
                        stderr=subprocess.STDOUT,  # 将stderr也合并到stdout
                        cwd=script_dir,
                        start_new_session=True  # 创建新的进程组，确保完全独立（Unix系统会自动调用setsid）
                    )
                    # 子进程已继承文件描述符，可以安全关闭父进程的文件句柄
                    # 在Linux/Unix系统中，子进程可以继续写入，直到子进程结束
                    log_fd.close()
                    log_fd = None  # 标记已关闭，避免在except中重复关闭
                    
                    # 不等待进程完成，立即返回（后台执行）
                    print(f"[INFO] Test process started (PID: {process.pid})")
                except Exception as proc_error:
                    # 如果启动失败，确保关闭文件
                    if log_fd:
                        log_fd.close()
                    raise proc_error
                
            except Exception as e:
                print(f"[ERROR] Failed to execute test script: {e}")
                import traceback
                traceback.print_exc()
    
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
        # HybridLoss 内部会处理 Sigmoid 和各项损失计算
        loss = self.criterion(logits_pred, y, mask=target_mask)
        
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
        loss = self.criterion(logits_pred, y, mask=target_mask)
        
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