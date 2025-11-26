#!/bin/bash

# MeteoMamba 全流程脚本 (Optimized for 4x A800 80GB)
# 包含: Train (MeteoMamba) -> Test (MeteoMamba Visualization)

export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

if [ $# -eq 0 ]; then
    echo "用法: bash run.scwds.mamba.sh [MODE]"
    exit 1
fi

MODE=$1
DEVICES="[0,1,2,3]"
DATA_PATH="data/samples.jsonl"
SAVE_DIR="./output/meteo_mamba"

case $MODE in
    "train")
        echo "🚀 [4x A800] 开始训练 MeteoMamba..."
        python run/train_scwds_mamba.py fit \
            --seed_everything 42 \
            --trainer.default_root_dir $SAVE_DIR \
            --trainer.accelerator gpu \
            --trainer.devices $DEVICES \
            --trainer.strategy ddp \
            --trainer.precision bf16-mixed \
            --trainer.max_epochs 50 \
            --trainer.accumulate_grad_batches 8 \
            --trainer.log_every_n_steps 10 \
            --trainer.callbacks+=lightning.pytorch.callbacks.ModelCheckpoint \
            --trainer.callbacks.monitor "val_score" \
            --trainer.callbacks.mode "max" \
            --trainer.callbacks.save_top_k 3 \
            --trainer.callbacks.save_last true \
            --trainer.callbacks.filename "{epoch:02d}-{val_score:.4f}" \
            --trainer.callbacks+=lightning.pytorch.callbacks.EarlyStopping \
            --trainer.callbacks.monitor "val_score" \
            --trainer.callbacks.mode "max" \
            --trainer.callbacks.patience 30 \
            --model.in_shape "[10, 31, 256, 256]" \
            --model.aft_seq_length 20 \
            --model.hid_S 64 \
            --model.hid_T 256 \
            --model.N_S 4 \
            --model.N_T 8 \
            --model.use_curriculum_learning true \
            --data.data_path $DATA_PATH \
            --data.batch_size 2 \
            --data.num_workers 16
        ;;
        
    "test")
        echo "🧪 开始测试 MeteoMamba (Visualization)..."
        
        # 自动寻找 Checkpoint
        CKPT_PATH=$(find $SAVE_DIR -name "*val_score*.ckpt" | sort -V | tail -n 1)
        if [ -z "$CKPT_PATH" ]; then CKPT_PATH=$(find $SAVE_DIR -name "last.ckpt" | head -n 1); fi
        
        if [ -z "$CKPT_PATH" ]; then
            echo "❌ 错误: 未找到 Checkpoint"
            exit 1
        fi
        
        echo "Using Checkpoint: $CKPT_PATH"
        
        # [修正] 使用纯 argparse 参数
        python run/test_scwds_mamba.py \
            --ckpt_path "$CKPT_PATH" \
            --save_dir "$SAVE_DIR/vis_check" \
            --num_samples 10 \
            --data_path "$DATA_PATH"
        ;;
        
    *)
        echo "错误: 不支持的操作模式 '$MODE'"
        exit 1
        ;;
esac

echo "✅ 操作完成！"