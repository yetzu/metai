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
    parser.add_argument('--save_dir', type=str, default=None, help='Output directory')
    # [注意] 这里默认值设为 None，以优先使用 Config 中的 SOTA 默认配置 (10, 54, 256, 256)
    parser.add_argument('--in_shape', type=int, nargs=4, default=None) 
    parser.add_argument('--resize_shape', type=int, nargs=2, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--max_epochs', type=int, default=None)
    parser.add_argument('--task_mode', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    
    # [关键新增] 预测序列长度 (10帧输入 -> 20帧输出)
    parser.add_argument('--aft_seq_length', type=int, default=None, help='Output sequence length (prediction frames)')

    # 模型结构参数
    parser.add_argument('--model_type', type=str, default=None)
    parser.add_argument('--hid_S', type=int, default=None)
    parser.add_argument('--hid_T', type=int, default=None)
    parser.add_argument('--N_S', type=int, default=None)
    parser.add_argument('--N_T', type=int, default=None)
    parser.add_argument('--mlp_ratio', type=float, default=None)
    parser.add_argument('--drop', type=float, default=None)
    parser.add_argument('--drop_path', type=float, default=None)
    parser.add_argument('--spatio_kernel_enc', type=int, default=None)
    parser.add_argument('--spatio_kernel_dec', type=int, default=None)
    
    # 优化器
    parser.add_argument('--opt', type=str, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--sched', type=str, default=None)
    parser.add_argument('--min_lr', type=float, default=None)
    parser.add_argument('--warmup_epoch', type=int, default=None)
    parser.add_argument('--accumulate_grad_batches', type=int, default=None)
    parser.add_argument('--gradient_clip_val', type=float, default=None)
    parser.add_argument('--gradient_clip_algorithm', type=str, default=None)
    
    # 设备
    parser.add_argument('--accelerator', type=str, default=None)
    parser.add_argument('--devices', type=str, default=None)
    parser.add_argument('--precision', type=str, default=None)
    
    # 损失函数参数 (HybridLoss)
    parser.add_argument('--loss_weight_l1', type=float, default=None)
    parser.add_argument('--loss_weight_ssim', type=float, default=None)
    parser.add_argument('--loss_weight_csi', type=float, default=None)
    parser.add_argument('--loss_weight_spectral', type=float, default=None)
    parser.add_argument('--loss_weight_evo', type=float, default=None)
    
    # 课程学习
    parser.add_argument('--use_curriculum_learning', type=str, default=None) # 兼容 'true'/'false' 字符串
    
    # [关键新增] 早停参数
    parser.add_argument('--early_stop_patience', type=int, default=None, help='Patience for early stopping')
    parser.add_argument('--early_stop_monitor', type=str, default=None)
    parser.add_argument('--early_stop_mode', type=str, default=None)

    # Resume
    parser.add_argument('--ckpt_path', type=str, default=None)

    return parser.parse_args()

def detect_precision():
    if not torch.cuda.is_available(): return '16-mixed'
    try:
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8: return 'bf16-mixed'
    except: pass
    return '16-mixed'

def main():
    torch.set_float32_matmul_precision('high')
    args = parse_args()
    
    # 过滤掉 None 的参数，优先使用 Config 类的默认值
    config_kwargs = {k: v for k, v in vars(args).items() if v is not None}
    
    if 'in_shape' in config_kwargs: config_kwargs['in_shape'] = tuple(config_kwargs['in_shape'])
    if 'resize_shape' in config_kwargs: del config_kwargs['resize_shape']
    config_kwargs['out_channels'] = 1 
    
    # Device Parsing
    if 'devices' in config_kwargs and isinstance(config_kwargs['devices'], str):
        val = config_kwargs['devices'].strip()
        if val.lower() == 'auto': config_kwargs['devices'] = 'auto'
        elif val.startswith('[') or ',' in val:
            try: config_kwargs['devices'] = ast.literal_eval(val)
            except: config_kwargs['devices'] = [int(x) for x in val.split(',')]
        else:
            try: config_kwargs['devices'] = int(val)
            except: config_kwargs['devices'] = val
    
    if 'accelerator' not in config_kwargs:
        config_kwargs['accelerator'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    if 'precision' not in config_kwargs:
        config_kwargs['precision'] = detect_precision()
        
    # Boolean Parsing
    if 'use_curriculum_learning' in config_kwargs and isinstance(config_kwargs['use_curriculum_learning'], str):
        config_kwargs['use_curriculum_learning'] = config_kwargs['use_curriculum_learning'].lower() == 'true'
    
    try:
        # 使用过滤后的参数初始化 Config
        config = SimVPConfig(**config_kwargs)
    except ValidationError as e:
        print(f"[ERROR] Config Validation: {e}")
        return

    l.seed_everything(config.seed)

    data_module = ScwdsDataModule(
        data_path=config.data_path,
        resize_shape=config.resize_shape,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_split=config.train_split,
        val_split=config.val_split,
        test_split=config.test_split,
        seed=config.seed
    )
    
    model_args = config.to_dict()
    model = SimVP(**model_args)

    # ==========================================
    # 🚀 优化后的 Callbacks 配置
    # ==========================================
    monitor_metric = config.early_stop_monitor # 'val_score'
    monitor_mode = config.early_stop_mode      # 'max'

    callbacks = [
        # 1. 早停
        EarlyStopping(
            monitor=monitor_metric, 
            min_delta=config.early_stop_min_delta, 
            patience=config.early_stop_patience, 
            mode=monitor_mode, 
            verbose=True
        ),
        
        # 2. 保存 Top-3 最优模型
        ModelCheckpoint(
            dirpath=config.save_dir, 
            filename="{epoch:02d}-{val_score:.4f}",
            monitor=monitor_metric,
            save_top_k=3, 
            mode=monitor_mode,
            save_last=False 
        ),
        
        # 3. 总是保存 last.ckpt (用于 Resume)
        ModelCheckpoint(
            dirpath=config.save_dir, 
            filename="last",
            save_top_k=0, 
            save_last=True, 
            every_n_epochs=1
        ),
        
        # 4. 定期保存 (每5轮)
        ModelCheckpoint(
            dirpath=config.save_dir, 
            filename="periodic-{epoch:02d}",
            every_n_epochs=5, 
            save_top_k=-1 
        ), 
        
        LearningRateMonitor(logging_interval="step")
    ]

    logger = TensorBoardLogger(save_dir=config.save_dir, name=config.model_name, version=datetime.now().strftime("%Y%m%d-%H%M%S"))

    # DDP Strategy
    use_ddp = False
    if config.accelerator == 'cuda':
        devices = config.devices
        if devices == 'auto': pass 
        elif isinstance(devices, list) and len(devices) > 1: use_ddp = True
        elif isinstance(devices, int) and devices > 1: use_ddp = True
    
    strategy = 'ddp_find_unused_parameters_false' if use_ddp else 'auto'

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
        gradient_clip_algorithm=config.gradient_clip_algorithm,
        accumulate_grad_batches=config.accumulate_grad_batches,
        strategy=strategy,
        sync_batchnorm=False, 
        enable_progress_bar=config.enable_progress_bar,
        enable_model_summary=config.enable_model_summary,
        num_sanity_val_steps=config.num_sanity_val_steps,
    )

    print(f"Starting Training: Model={config.model_type}, Shape={config.in_shape} -> {config.resize_shape}")
    print(f"  Patience: {config.early_stop_patience}")
    print(f"  Curriculum: {config.use_curriculum_learning}")
    print(f"  Input Channels: {config.in_shape[1]}, Output Frames: {config.aft_seq_length}")
    
    resume_ckpt = args.ckpt_path
    if resume_ckpt is None:
        possible_last = os.path.join(config.save_dir, "last.ckpt")
        if os.path.exists(possible_last): resume_ckpt = possible_last

    if resume_ckpt and os.path.exists(resume_ckpt):
        print(f"[INFO] Resuming from checkpoint: {resume_ckpt}")
        trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt)
    else:
        print("[INFO] Starting fresh training")
        trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()