# run/gan_train_scwds_simvp.py
import sys
import os
import argparse

import lightning as l
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# 路径修正
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.dataset.met_dataloader_scwds import ScwdsDataModule
from metai.model.simvp import SimVP_GAN

def main():
    parser = argparse.ArgumentParser()
    # 数据参数
    parser.add_argument('--data_path', type=str, default='data/samples.jsonl')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to pre-trained SimVP checkpoint')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    
    # GAN 参数
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lambda_adv', type=float, default=0.01, help='Weight for adversarial loss')
    parser.add_argument('--max_epochs', type=int, default=50)
    
    # 硬件参数
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--accelerator', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # 1. 数据模块 (沿用之前的逻辑)
    # 注意：这里 resize_shape 硬编码为 (256, 256)，需与 Backbone 一致
    data_module = ScwdsDataModule(
        data_path=args.data_path,
        resize_shape=(256, 256), 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=0.8, val_split=0.1, test_split=0.1
    )
    
    # 2. 模型模块 (GAN)
    model = SimVP_GAN(
        backbone_ckpt_path=args.ckpt_path,
        lr=args.lr,
        lambda_adv=args.lambda_adv
    )
    
    # 3. Logger & Callbacks
    logger = TensorBoardLogger("output", name="simvp_gan")
    checkpoint_callback = ModelCheckpoint(
        dirpath="output/simvp_gan/checkpoints",
        filename="{epoch:02d}-{val_score:.4f}",
        monitor="val_score",
        mode="max",
        save_top_k=5,
        save_last=True
    )
    
    # 4. Trainer
    # GAN 训练通常不需要 accumulate_grad_batches，因为只训练小网络，显存够用
    trainer = l.Trainer(
        accelerator=args.accelerator,
        devices=[int(x) for x in args.devices.split(',')],
        strategy='ddp_find_unused_parameters_true', # GAN 必需
        max_epochs=args.max_epochs,
        precision='bf16-mixed', # 保持 BF16 加速
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )
    
    print(f"🚀 Starting GAN Fine-tuning with Backbone: {args.ckpt_path}")
    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()