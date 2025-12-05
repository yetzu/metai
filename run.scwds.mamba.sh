#!/bin/bash

# =========================================================
# MeteoMamba Workflow Script (Fixed: Removed conflicting clip args)
# =========================================================

export PYTHONPATH=$PYTHONPATH:$(pwd)

# --- 显存与通信优化 ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

if [ $# -eq 0 ]; then
    echo "Usage: bash run.scwds.mamba.sh [MODE]"
    echo "  MODE: train | test | infer"
    exit 1
fi

MODE=$1
# 设置显卡设备 [1,2,3]
DEVICES="[1,2,3]" 
DATA_PATH="data/samples.jsonl"
SAVE_DIR="./output" 
BATCH_SIZE=4

case $MODE in
    "train")
        echo "--------------------------------------------------------"
        echo " [MetAI] Starting Training ..."
        echo "--------------------------------------------------------"
        
        python run/train_scwds_mamba.py fit \
            --seed_everything 42 \
            --trainer.default_root_dir $SAVE_DIR \
            --trainer.accelerator cuda \
            --trainer.devices $DEVICES \
            --trainer.strategy ddp_find_unused_parameters_true \
            --trainer.precision bf16-mixed \
            --trainer.max_epochs 100 \
            --trainer.log_every_n_steps 100 \
            --trainer.accumulate_grad_batches 4 \
            --trainer.callbacks+=lightning.pytorch.callbacks.ModelCheckpoint \
            --trainer.callbacks.monitor "val_score" \
            --trainer.callbacks.mode "max" \
            --trainer.callbacks.save_top_k -1 \
            --trainer.callbacks.save_last true \
            --trainer.callbacks.filename "epoch={epoch:02d}-score={val_score:.4f}" \
            --trainer.callbacks+=lightning.pytorch.callbacks.EarlyStopping \
            --trainer.callbacks.monitor "val_score" \
            --trainer.callbacks.mode "max" \
            --trainer.callbacks.patience 20 \
            --trainer.gradient_clip_val 0.5 \
            --trainer.gradient_clip_algorithm "norm" \
            --data.data_path $DATA_PATH \
            --data.batch_size $BATCH_SIZE \
            --data.num_workers 8
        ;;
        
    "test")
        echo "----------------------------------------"
        echo " [MetAI] Starting Test (Metrics & Evaluation)..."
        echo "----------------------------------------"
        
        CKPT_PATH=$(find $SAVE_DIR -name "*val_score*.ckpt" | sort -V | tail -n 1)
        if [ -z "$CKPT_PATH" ]; then CKPT_PATH=$(find $SAVE_DIR -name "last.ckpt" | head -n 1); fi
        
        if [ -z "$CKPT_PATH" ]; then
            echo " Error: Checkpoint not found in $SAVE_DIR"
            exit 1
        fi
        
        echo "Using Checkpoint: $CKPT_PATH"
        
        python run/test_scwds_mamba.py \
            --ckpt_path "$CKPT_PATH" \
            --save_dir "$SAVE_DIR" \
            --data_path "$DATA_PATH" \
            --in_shape 31 256 256 \
            --in_seq_len 10 \
            --out_seq_len 20 \
            --num_samples 10 \
            --accelerator cuda:0 
        ;;

    "infer")
        echo "----------------------------------------"
        echo " [MetAI] Starting Inference (Submission Generation)..."
        echo "----------------------------------------"
        
        CKPT_PATH=$(find $SAVE_DIR -name "*val_score*.ckpt" | sort -V | tail -n 1)
        if [ -z "$CKPT_PATH" ]; then CKPT_PATH=$(find $SAVE_DIR -name "last.ckpt" | head -n 1); fi

        python run/infer_scwds_mamba.py \
            --ckpt_dir "$SAVE_DIR" \
            --data_path "data/samples.testset.jsonl" \
            --save_dir "./submit/output" \
            --vis_output "./submit/vis_infer" \
            --resize_shape 256 256 \
            --accelerator cuda:0 \
            --vis
        ;;
        
    *)
        echo "Error: Unsupported mode '$MODE'"
        echo "Supported modes: train, test, infer"
        exit 1
        ;;
esac

echo " Operation Completed."