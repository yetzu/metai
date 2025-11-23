# metai/model/simvp_trainer.py

import subprocess
import os
import sys
import time
import glob
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

        # 课程学习配置
        self.use_curriculum_learning = config.get('use_curriculum_learning', True)
        
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
             model_type=config.get('model_type', 'mamba'), out_channels=config.get('out_channels', 1),
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
        """课程学习：动态调整 HybridLoss 的权重（如果启用）"""
        # 如果禁用课程学习，使用命令行传入的固定权重，不进行动态调整
        if not self.use_curriculum_learning:
            return
        
        epoch = self.current_epoch
        
        # 获取 Loss 模块 (假设 self.criterion 是 HybridLoss)
        if not hasattr(self, 'criterion') or not isinstance(self.criterion, HybridLoss):
            return

        # === 阶段定义 ===
        if epoch < 10: 
            # Phase 1: 定性 (Warmup)
            weights = {'l1': 5.0, 'ssim': 1.0, 'evo': 0.1, 'spec': 0.0, 'csi': 0.0}
            phase_name = "Phase 1: Qualitative (Structure)"
        
        elif epoch < 30:
            # Phase 2: 定量 (Physics & Sharpness)
            # 线性过渡示例: evo 从 0.1 -> 2.0
            progress = (epoch - 10) / 20.0
            evo_w = 0.1 + progress * (2.0 - 0.1)
            spec_w = 0.0 + progress * (0.5 - 0.0)
            csi_w = 0.0 + progress * (1.0 - 0.0)
            
            weights = {'l1': 1.0, 'ssim': 0.5, 'evo': evo_w, 'spec': spec_w, 'csi': csi_w}
            phase_name = f"Phase 2: Quantitative (Physics & Sharpness) [p={progress:.2f}]"
            
        else:
            # Phase 3: 冲刺 (Score Maximization)
            weights = {'l1': 0.1, 'ssim': 0.2, 'evo': 1.0, 'spec': 1.0, 'csi': 5.0}
            phase_name = "Phase 3: Sprint (Score Maximization)"

        # 更新权重
        self.criterion.weights.update(weights)
        
        # 记录日志
        if self.trainer.is_global_zero:
            print(f"\n[Curriculum] Epoch {epoch} | Phase: {phase_name}")
            print(f"             Weights: {weights}")
        
        # TensorBoard 记录
        for k, v in weights.items():
            self.log(f"train/weight_{k}", v, on_epoch=True)

    def on_train_epoch_end(self):
        """在每个训练epoch结束后执行测试脚本（后台执行，不阻塞训练）"""
        if self.trainer.is_global_zero and self.auto_test_after_epoch:
            try:
                if not self.test_script_path:
                    return
                
                script_path = str(self.test_script_path)
                if not os.path.isabs(script_path):
                    # 尝试定位到项目根目录
                    current_file = os.path.abspath(__file__)
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
                    script_path = os.path.join(project_root, script_path)
                
                if not os.path.exists(script_path):
                    print(f"[WARNING] Test script not found: {script_path}")
                    return
                
                epoch = self.current_epoch
                
                # 获取 checkpoint 保存目录
                save_dir = None
                if hasattr(self, 'hparams'):
                    # hparams 可能是 dict 或 Namespace
                    if isinstance(self.hparams, dict):
                        save_dir = self.hparams.get('save_dir', None)
                    else:
                        save_dir = getattr(self.hparams, 'save_dir', None)
                
                if save_dir is None:
                    save_dir = getattr(self.trainer, 'default_root_dir', None)
                
                # 等待 checkpoint 文件出现
                if save_dir:
                    max_wait_time = 300  # 最多等待 5 分钟
                    check_interval = 2  # 每 2 秒检查一次
                    waited_time = 0
                    ckpt_pattern = os.path.join(save_dir, '*.ckpt')
                    
                    print(f"\n[INFO] Epoch {epoch} done. Waiting for checkpoint in {save_dir}...")
                    
                    while waited_time < max_wait_time:
                        ckpt_files = glob.glob(ckpt_pattern)
                        if len(ckpt_files) > 0:
                            # 找到最新的 checkpoint
                            latest_ckpt = max(ckpt_files, key=os.path.getmtime)
                            print(f"[INFO] Checkpoint found: {latest_ckpt}")
                            break
                        time.sleep(check_interval)
                        waited_time += check_interval
                        if waited_time % 10 == 0:  # 每 10 秒打印一次等待信息
                            print(f"[INFO] Still waiting for checkpoint... ({waited_time}s/{max_wait_time}s)")
                    else:
                        print(f"[WARNING] Timeout waiting for checkpoint after {max_wait_time}s. Proceeding anyway...")
                
                # 日志目录
                script_dir = os.path.dirname(script_path) or os.getcwd()
                log_dir = os.path.join(script_dir, 'test_logs')
                os.makedirs(log_dir, exist_ok=True)
                
                log_file = os.path.join(log_dir, f'test_epoch_{epoch:03d}.log')
                
                print(f"[INFO] Launching background test: {script_path}")
                
                # 后台执行
                log_fd = open(log_file, 'w')
                try:
                    subprocess.Popen(
                        ['bash', script_path, 'test'], # 传递 'test' 参数给脚本
                        stdout=log_fd,
                        stderr=subprocess.STDOUT,
                        cwd=script_dir,
                        start_new_session=True 
                    )
                    log_fd.close() 
                except Exception as proc_error:
                    if log_fd: log_fd.close()
                    raise proc_error
                
            except Exception as e:
                print(f"[ERROR] Failed to launch test script: {e}")
    
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
        loss, loss_dict = self.criterion(logits_pred, y, mask=target_mask)
        
        # 记录总损失
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # 记录各个损失组件（原始值和加权值）
        loss_components = ['l1', 'ssim', 'csi', 'spec', 'evo']
        for comp in loss_components:
            if comp in loss_dict:
                self.log(f'train_loss_{comp}', loss_dict[comp], on_step=True, on_epoch=True, prog_bar=False)
            if f'{comp}_weighted' in loss_dict:
                self.log(f'train_loss_{comp}_weighted', loss_dict[f'{comp}_weighted'], on_step=True, on_epoch=True, prog_bar=False)
        
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
        loss, loss_dict = self.criterion(logits_pred, y, mask=target_mask)
        
        # 记录验证阶段的各个损失组件
        loss_components = ['l1', 'ssim', 'csi', 'spec', 'evo']
        for comp in loss_components:
            if comp in loss_dict:
                self.log(f'val_loss_{comp}', loss_dict[comp], on_epoch=True, sync_dist=True)
            if f'{comp}_weighted' in loss_dict:
                self.log(f'val_loss_{comp}_weighted', loss_dict[f'{comp}_weighted'], on_epoch=True, sync_dist=True)
        
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
            loss, loss_dict = self.criterion(logits_pred, y, mask=target_mask)
            
            # 记录测试阶段的各个损失组件
            loss_components = ['l1', 'ssim', 'csi', 'spec', 'evo']
            for comp in loss_components:
                if comp in loss_dict:
                    self.log(f'test_loss_{comp}', loss_dict[comp], on_epoch=True)
                if f'{comp}_weighted' in loss_dict:
                    self.log(f'test_loss_{comp}_weighted', loss_dict[f'{comp}_weighted'], on_epoch=True)
            
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