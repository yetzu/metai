#!/bin/bash

# MeteoMamba 全流程脚本 (Optimized for 4x A800 80GB)
# 包含: Train (MeteoMamba) -> Test (MeteoMamba)
# Usage: bash run.scwds.mamba.sh [MODE]

# ================= 环境变量优化 =================
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTORCH_ALLOC_CONF=expandable_segments:True
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
# 显卡设置
DEVICES="[0,1,2,3]"
# 基础路径
DATA_PATH="data/samples.jsonl"
SAVE_DIR="./output/meteo_mamba"

case $MODE in
    # ============================================================
    # 1. 训练 MeteoMamba (Stage 1)
    # ============================================================
    "train")
        echo "--------------------------------------------------------"
        echo "🚀 [4x A800] 开始训练 MeteoMamba 模型 (BF16 Mixed)..."
        echo "--------------------------------------------------------"
        
        # 注意：新的 train_scwds_mamba.py 使用 LightningCLI
        # 参数格式为 --section.arg value
        
        python run/train_scwds_mamba.py fit \
            --seed_everything 42 \
            \
            --trainer.default_root_dir $SAVE_DIR \
            --trainer.accelerator gpu \
            --trainer.devices $DEVICES \
            --trainer.strategy ddp \
            --trainer.precision bf16-mixed \
            --trainer.max_epochs 50 \
            --trainer.gradient_clip_val 0.5 \
            --trainer.accumulate_grad_batches 4 \
            --trainer.log_every_n_steps 10 \
            
            \
            --model.in_shape "[10, 31, 256, 256]" \
            --model.aft_seq_length 20 \
            --model.hid_S 64 \
            --model.hid_T 256 \
            --model.N_S 4 \
            --model.N_T 8 \
            --model.lr 5e-4 \
            --model.min_lr 1e-5 \
            --model.warmup_epoch 10 \
            --model.use_curriculum_learning true \
            \
            --data.data_path $DATA_PATH \
            --data.batch_size 4 \
            --data.num_workers 8 \
            --data.resize_shape "[256, 256]"
        ;;
        
    # ============================================================
    # 2. 测试 MeteoMamba
    # ============================================================
    "test")
        echo "----------------------------------------"
        echo "🧪 开始测试 MeteoMamba 模型..."
        echo "----------------------------------------"
        
        # 自动查找最佳 Checkpoint (如果 best 不存在则找 last)
        CKPT_PATH=$(find $SAVE_DIR -name "*.ckpt" | grep "last.ckpt" | head -n 1)
        if [ -z "$CKPT_PATH" ]; then
             # 尝试找 best
             CKPT_PATH=$(find $SAVE_DIR -name "*.ckpt" | grep "epoch=" | sort -V | tail -n 1)
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
            \
            --ckpt_path "$CKPT_PATH" \
            \
            --model.in_shape "[10, 31, 256, 256]" \
            --model.aft_seq_length 20 \
            --model.hid_S 64 \
            --model.hid_T 256 \
            --model.N_S 4 \
            --model.N_T 8 \
            \
            --data.data_path $DATA_PATH \
            --data.batch_size 1 \
            --data.num_workers 4 \
            --data.resize_shape "[256, 256]"
        ;;
        
    *)
        echo "错误: 不支持的操作模式 '$MODE'"
        echo "支持的模式: train, test"
        exit 1
        ;;
esac

echo "✅ 操作完成！"