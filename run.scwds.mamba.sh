#!/bin/bash

# MeteoMamba 全流程脚本 (Optimized for 4x A800 80GB)
# 包含: Train (MeteoMamba) -> Test (MeteoMamba)
# Usage: bash run.scwds.mamba.sh [MODE]

# ================= 环境变量优化 =================
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

# ================= 参数检查 =================
if [ $# -eq 0 ]; then
    echo "错误: 请指定操作模式"
    echo "用法: bash run.scwds.mamba.sh [MODE]"
    echo "支持的模式:"
    echo " train      - 训练 MeteoMamba 模型"
    echo " test       - 测试 MeteoMamba 模型"
    exit 1
fi

MODE=$1

# ================= 配置参数 =================
DEVICES="[0,1,2,3]"
DATA_PATH="data/samples.jsonl"
SAVE_DIR="./output/meteo_mamba"

case $MODE in
    # ============================================================
    # 1. 训练 MeteoMamba (Stage 1)
    # ============================================================
    "train")
        echo "--------------------------------------------------------"
        echo "🚀 [4x A800] 开始训练 MeteoMamba (Monitor: val_score)..."
        echo "--------------------------------------------------------"
        
        # [核心修改]
        # 1. 添加 ModelCheckpoint: 监控 val_score (max), 保存 Top-3
        # 2. 添加 EarlyStopping: 监控 val_score (max), Patience=30
        # 3. 显式指定完整类名以通过 CLI 加载
        
        python run/train_scwds_mamba.py fit \
            --seed_everything 42 \
            --trainer.default_root_dir $SAVE_DIR \
            --trainer.accelerator gpu \
            --trainer.devices $DEVICES \
            --trainer.strategy ddp \
            --trainer.precision bf16-mixed \
            --trainer.max_epochs 50 \
            --trainer.gradient_clip_val 0.5 \
            --trainer.accumulate_grad_batches 8 \
            --trainer.log_every_n_steps 10 \
            \
            --trainer.callbacks+=lightning.pytorch.callbacks.ModelCheckpoint \
            --trainer.callbacks.monitor "val_score" \
            --trainer.callbacks.mode "max" \
            --trainer.callbacks.save_top_k 3 \
            --trainer.callbacks.save_last true \
            --trainer.callbacks.filename "{epoch:02d}-{val_score:.4f}" \
            \
            --trainer.callbacks+=lightning.pytorch.callbacks.EarlyStopping \
            --trainer.callbacks.monitor "val_score" \
            --trainer.callbacks.mode "max" \
            --trainer.callbacks.patience 30 \
            --trainer.callbacks.min_delta 0.0 \
            \
            --model.in_shape "[10, 31, 256, 256]" \
            --model.aft_seq_length 20 \
            --model.hid_S 64 \
            --model.hid_T 256 \
            --model.N_S 4 \
            --model.N_T 8 \
            --model.lr 1e-3 \
            --model.min_lr 1e-5 \
            --model.warmup_epoch 10 \
            --model.use_curriculum_learning true \
            \
            --data.data_path $DATA_PATH \
            --data.batch_size 2 \
            --data.num_workers 16
        ;;
        
    # ============================================================
    # 2. 测试 MeteoMamba
    # ============================================================
    "test")
        echo "----------------------------------------"
        echo "🧪 开始测试 MeteoMamba 模型..."
        echo "----------------------------------------"
        
        # 这里优先使用 'last.ckpt' 继续训练或测试，或者手动指定最佳权重
        CKPT_PATH=$(find $SAVE_DIR -name "*val_score*.ckpt" | sort -V | tail -n 1)
        
        if [ -z "$CKPT_PATH" ]; then
             CKPT_PATH=$(find $SAVE_DIR -name "last.ckpt" | head -n 1)
        fi
        
        if [ -z "$CKPT_PATH" ]; then
            echo "❌ 错误: 未找到 Checkpoint 文件 ($SAVE_DIR)"
            exit 1
        fi
        
        echo "Using Checkpoint: $CKPT_PATH"
        
        python run/train_scwds_mamba.py test \
            --trainer.accelerator gpu \
            --trainer.devices 1 \
            --trainer.precision bf16-mixed \
            --ckpt_path "$CKPT_PATH" \
            --data.data_path $DATA_PATH \
            --data.batch_size 1 \
            --data.num_workers 4
        ;;
        
    *)
        echo "错误: 不支持的操作模式 '$MODE'"
        echo "支持的模式: train, test"
        exit 1
        ;;
esac

echo "✅ 操作完成！"