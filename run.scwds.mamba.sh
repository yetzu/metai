#!/bin/bash

# MeteoMamba 全流程脚本 (Optimized for A800 80GB)
# 包含: Train (MeteoMamba) -> Test (MeteoMamba Visualization)

export PYTHONPATH=$PYTHONPATH:$(pwd)
# A800 显存足够，通常不需要过于激进的碎片整理
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

if [ $# -eq 0 ]; then
    echo "用法: bash run.scwds.mamba.sh [MODE]"
    exit 1
fi

MODE=$1

# [注意] 如果是单卡 A800，请改为 DEVICES="[0]"
# 如果是多卡，保持 "[0,1,2,3]"，Batch Size 会自动乘以卡数 (Global Batch Size)
DEVICES="[1,2,3]" 
DATA_PATH="data/samples.jsonl"
SAVE_DIR="./output" # 修改输出目录以免覆盖旧实验

case $MODE in
    "train")
        echo "--------------------------------------------------------"
        echo "🚀 [A800] 开始训练 Met Mamba 基座模型 (BF16 Mixed)..."
        echo "--------------------------------------------------------"
        python run/train_scwds_mamba.py fit \
            --seed_everything 42 \
            --trainer.default_root_dir $SAVE_DIR \
            --trainer.accelerator cuda \
            --trainer.devices $DEVICES \
            --trainer.strategy ddp \
            --trainer.precision bf16-mixed \
            --trainer.max_epochs 50 \
            --trainer.log_every_n_steps 50 \
            --trainer.accumulate_grad_batches 4 \
            --trainer.gradient_clip_val 1.0 \
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
            --model.in_shape "[31, 256, 256]" \
            --model.obs_seq_len 10 \
            --model.pred_seq_len 20 \
            --model.hid_S 64 \
            --model.hid_T 256 \
            --model.N_S 4 \
            --model.N_T 6 \
            --model.mamba_d_state 16 \
            --model.mamba_d_conv 4 \
            --model.mamba_expand 2 \
            --model.use_curriculum_learning false \
            --model.loss_weight_l1 1.0 \
            --model.loss_weight_gdl 5.0 \
            --data.data_path $DATA_PATH \
            --data.batch_size 4 \
            --data.num_workers 16 \
            --data.aft_seq_length 20 
        ;;
        
    "test")
        echo "----------------------------------------"
        echo "🧪 开始测试 Met Mamba 基座模型..."
        echo "----------------------------------------"
        
        # 自动寻找 Checkpoint
        CKPT_PATH=$(find $SAVE_DIR -name "*val_score*.ckpt" | sort -V | tail -n 1)
        if [ -z "$CKPT_PATH" ]; then CKPT_PATH=$(find $SAVE_DIR -name "last.ckpt" | head -n 1); fi
        
        if [ -z "$CKPT_PATH" ]; then
            echo "❌ 错误: 未找到 Checkpoint"
            exit 1
        fi
        
        echo "Using Checkpoint: $CKPT_PATH"
        
        # [Update] 这里的参数已更新为匹配 test_scwds_mamba.py 的新接口
        python run/test_scwds_mamba.py \
            --ckpt_path "$CKPT_PATH" \
            --save_dir "$SAVE_DIR/vis_check" \
            --num_samples 10 \
            --data_path "$DATA_PATH" \
            --accelerator cuda:0 \
            --in_shape 31 256 256 \
            --obs_seq_len 10 \
            --pred_seq_len 20
        ;;
        
    *)
        echo "错误: 不支持的操作模式 '$MODE'"
        exit 1
        ;;
esac

echo "✅ 操作完成！"