# run/train_scwds_simvp.py
import sys
import os
from datetime import datetime
import argparse
import ast
from pydantic import ValidationError

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import lightning as l
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from metai.dataset.met_dataloader_scwds import ScwdsDataModule
from metai.model.simvp import SimVPConfig, SimVP

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train SCWDS SimVP Model')
    
    # 基础参数
    parser.add_argument('--data_path', type=str, default='data/samples.jsonl', help='Path to training data')
    parser.add_argument('--save_dir', type=str, default=None, help='Output directory for saving checkpoints and logs')
    parser.add_argument('--in_shape', type=int, nargs=4, default=[20, 29, 128, 128], help='Input shape: T C H W')
    parser.add_argument('--resize_shape', type=int, nargs=2, default=None, help='Resize shape: H W (Ignored as Config derives it)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=None, help='Maximum number of epochs')
    parser.add_argument('--task_mode', type=str, default=None, help='Task mode: precipitation or radar')
    
    # 数据加载器参数
    parser.add_argument('--num_workers', type=int, default=None, help='Number of data loader workers')
    
    # 模型结构参数
    parser.add_argument('--model_type', type=str, default=None, help='Model type')
    parser.add_argument('--hid_S', type=int, default=None, help='Hidden dim S')
    parser.add_argument('--hid_T', type=int, default=None, help='Hidden dim T')
    parser.add_argument('--N_S', type=int, default=None, help='Num layers S')
    parser.add_argument('--N_T', type=int, default=None, help='Num layers T')
    parser.add_argument('--mlp_ratio', type=float, default=None, help='MLP ratio')
    parser.add_argument('--drop', type=float, default=None, help='Dropout rate')
    parser.add_argument('--drop_path', type=float, default=None, help='Drop path rate')
    parser.add_argument('--spatio_kernel_enc', type=int, default=None, help='Spatial kernel enc')
    parser.add_argument('--spatio_kernel_dec', type=int, default=None, help='Spatial kernel dec')
    
    # 优化器/调度器参数
    parser.add_argument('--opt', type=str, default=None, help='Optimizer type')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--sched', type=str, default=None, help='Scheduler type')
    parser.add_argument('--min_lr', type=float, default=None, help='Min LR')
    parser.add_argument('--warmup_epoch', type=int, default=None, help='Warmup epochs')
    
    # 高级训练参数
    parser.add_argument('--accumulate_grad_batches', type=int, default=None, help='Grad accumulation')
    
    # GPU/设备参数
    parser.add_argument('--accelerator', type=str, default=None, choices=['auto', 'cpu', 'cuda'], help='Accelerator')
    parser.add_argument('--devices', type=str, default=None, help='Devices')
    parser.add_argument('--precision', type=str, default=None, help='Precision')
    
    # 损失函数参数 (HybridLoss，统一使用 loss_weight_ 前缀)
    parser.add_argument('--loss_weight_l1', type=float, default=None, help='L1 Loss weight (仅在禁用课程学习时生效)')
    parser.add_argument('--loss_weight_ssim', type=float, default=None, help='MS-SSIM Loss weight (仅在禁用课程学习时生效)')
    parser.add_argument('--loss_weight_csi', type=float, default=None, help='Soft-CSI Loss weight (仅在禁用课程学习时生效)')
    parser.add_argument('--loss_weight_spectral', type=float, default=None, help='Spectral Loss weight (仅在禁用课程学习时生效)')
    parser.add_argument('--loss_weight_evo', type=float, default=None, help='Evolution Loss weight (仅在禁用课程学习时生效)')
    
    # 课程学习参数
    parser.add_argument('--use_curriculum_learning', type=lambda x: x.lower() in ['true', '1', 'yes'], default=None, help='是否启用课程学习（默认: True）。如果禁用，将使用固定的 loss_weight_* 参数')
    
    return parser.parse_args()

def detect_precision():
    """检测并返回推荐的混合精度训练模式"""
    if not torch.cuda.is_available():
        return '16-mixed'
    
    try:
        device = torch.cuda.current_device()
        cap = torch.cuda.get_device_capability(device)
        major = cap[0]
        
        if major >= 8: # Ampere+
            print(f"[INFO] GPU Ampere+ ({major}.{cap[1]}) detected. Using bf16-mixed.")
            return 'bf16-mixed'
        else:
            print(f"[INFO] GPU ({major}.{cap[1]}) detected. Using 16-mixed.")
            return '16-mixed'
    except Exception:
        return '16-mixed'

def main():
    # 优化 Tensor Cores 性能
    torch.set_float32_matmul_precision('high')

    args = parse_args()
    
    # 1. 构建配置字典（仅包含非 None 的参数）
    config_kwargs = {k: v for k, v in vars(args).items() if v is not None}
    
    # 2. 特殊参数处理
    if 'in_shape' in config_kwargs:
        config_kwargs['in_shape'] = tuple(config_kwargs['in_shape'])
    
    # 关键：从 config_kwargs 中移除 resize_shape，由 Config 自动推导
    if 'resize_shape' in config_kwargs:
        del config_kwargs['resize_shape']
    
    # 3. 强制设置关键参数
    config_kwargs['out_channels'] = 1 
    
    # 处理 devices 参数 (支持 list/int/str)
    if 'devices' in config_kwargs and isinstance(config_kwargs['devices'], str):
        val = config_kwargs['devices'].strip()
        if val.lower() == 'auto':
            config_kwargs['devices'] = 'auto'
        elif val.startswith('[') or ',' in val:
            try:
                if val.startswith('['):
                    config_kwargs['devices'] = ast.literal_eval(val)
                else:
                    config_kwargs['devices'] = [int(x.strip()) for x in val.split(',')]
            except:
                config_kwargs['devices'] = val
        else:
            try:
                config_kwargs['devices'] = int(val)
            except:
                config_kwargs['devices'] = val
    
    # 默认值填充
    if 'accelerator' not in config_kwargs:
        config_kwargs['accelerator'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if 'precision' not in config_kwargs:
        config_kwargs['precision'] = detect_precision()
    
    # 布尔值处理 (String -> Bool) - 已移除，HybridLoss 不需要这些参数
    
    # 4. 实例化 Config 对象
    try:
        config = SimVPConfig(**config_kwargs)
    except ValidationError as e:
        print(f"[ERROR] Configuration Validation Failed:\n{e}")
        return
    except Exception as e:
        print(f"[ERROR] Unexpected error during config initialization: {e}")
        import traceback
        traceback.print_exc()
        return

    l.seed_everything(config.seed)

    # 5. 数据模块
    data_module = ScwdsDataModule(
        data_path=config.data_path,
        resize_shape=config.resize_shape, # 使用 Config 的计算属性
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_split=config.train_split,
        val_split=config.val_split,
        test_split=config.test_split,
        seed=config.seed # 传入 seed 保证数据划分可复现
    )
    
    # 6. 模型模块
    model_args = config.to_dict()
    model = SimVP(**model_args)

    # 7. Callbacks
    callbacks = [
        EarlyStopping(
            monitor=config.early_stop_monitor, 
            min_delta=config.early_stop_min_delta, 
            patience=config.early_stop_patience, 
            mode=config.early_stop_mode, 
            verbose=True
        ),
        # 最优模型检查点（保存最佳模型，文件名：best.ckpt）
        ModelCheckpoint(
            dirpath=config.save_dir, 
            filename="best",
            monitor=config.early_stop_monitor,
            save_top_k=1,  # 只保存最优的1个模型
            mode=config.early_stop_mode,
            save_last=False,
            auto_insert_metric_name=False  # 文件名不包含指标名
        ),
        # Top-3 模型检查点（用于对比分析，文件名包含epoch和score）
        ModelCheckpoint(
            dirpath=config.save_dir, 
            filename="{epoch:02d}-{val_score:.4f}",
            monitor=config.early_stop_monitor,
            save_top_k=3,  # 保存最好的3个模型
            mode=config.early_stop_mode,
            save_last=False 
        ),
        # 最后一个检查点（用于恢复训练）
        ModelCheckpoint(
            dirpath=config.save_dir, 
            filename="last",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False
        ),
        # 定期保存检查点（每5个epoch）
        ModelCheckpoint(
            dirpath=config.save_dir, 
            filename="periodic-{epoch:02d}",
            every_n_epochs=5, 
            save_top_k=-1  # 保存所有定期检查点
        ), 
        LearningRateMonitor(logging_interval="step")
    ]

    # 8. Logger
    logger = TensorBoardLogger(save_dir=config.save_dir, name=config.model_name, version=datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 9. Trainer 配置
    use_ddp = False
    if config.accelerator == 'cuda':
        devices = config.devices
        if devices == 'auto': pass 
        elif isinstance(devices, list) and len(devices) > 1: use_ddp = True
        elif isinstance(devices, int) and devices > 1: use_ddp = True
    
    strategy = 'ddp_find_unused_parameters_true' if use_ddp else 'auto'

    trainer = l.Trainer(
        max_epochs=config.max_epochs,
        default_root_dir=config.save_dir,
        precision=config.precision,
        accelerator=config.accelerator,
        devices=config.devices,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config.log_every_n_steps,
        val_check_interval=config.val_check_interval,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulate_grad_batches,
        strategy=strategy,
        sync_batchnorm=False, 
        enable_progress_bar=config.enable_progress_bar,
        enable_model_summary=config.enable_model_summary,
        num_sanity_val_steps=config.num_sanity_val_steps,
    )

    # 10. 训练或恢复
    print(f"Starting Training: Model={config.model_type}, Shape={config.in_shape} -> {config.resize_shape}")
    
    resume_ckpt = os.path.join(config.save_dir, "last.ckpt")
    
    if os.path.exists(resume_ckpt):
        print(f"[INFO] Resuming from checkpoint: {resume_ckpt}")
        trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt)
    else:
        print("[INFO] Starting fresh training")
        trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()