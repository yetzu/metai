#!/bin/bash

# MeteoMamba Workflow Script
# Modes: 
#   1. Train (MeteoMamba Training)
#   2. Test (Metrics & Evaluation)
#   3. Infer (Generate Submission Files 301x301)

export PYTHONPATH=$PYTHONPATH:$(pwd)
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

# Device Configuration
# Adjust DEVICES based on your available GPUs (e.g., "[0]" for single GPU)
DEVICES="[1,2,3]" 
DATA_PATH="data/samples.jsonl"
SAVE_DIR="./output" 
BATCH_SIZE=4

case $MODE in
    "train")
        echo "--------------------------------------------------------"
        echo " [MetAI] Starting Training (MeteoMamba) ..."
        echo "--------------------------------------------------------"

        python run/train_scwds_mamba.py fit \
            --seed_everything 42 \
            --trainer.default_root_dir $SAVE_DIR \
            --trainer.accelerator cuda \
            --trainer.devices $DEVICES \
            --trainer.strategy ddp \
            --trainer.precision bf16-mixed \
            --trainer.max_epochs 100 \
            --trainer.log_every_n_steps 100 \
            --trainer.accumulate_grad_batches 1 \
            --trainer.gradient_clip_val 1.0 \
            --trainer.callbacks+=lightning.pytorch.callbacks.ModelCheckpoint \
            --trainer.callbacks.monitor "val_score" \
            --trainer.callbacks.mode "max" \
            --trainer.callbacks.save_top_k -1 \
            --trainer.callbacks.save_last true \
            --trainer.callbacks.filename "{epoch:02d}-{val_score:.4f}" \
            --trainer.callbacks+=lightning.pytorch.callbacks.EarlyStopping \
            --trainer.callbacks.monitor "val_score" \
            --trainer.callbacks.mode "max" \
            --trainer.callbacks.patience 20 \
            --model.batch_size $BATCH_SIZE \
            --model.in_shape "[31, 256, 256]" \
            --model.obs_seq_len 10 \
            --model.pred_seq_len 20 \
            --model.hid_S 128 \
            --model.hid_T 512 \
            --model.N_S 4 \
            --model.N_T 8 \
            --model.mamba_d_state 32 \
            --model.mamba_d_conv 4 \
            --model.mamba_expand 2 \
            --model.use_checkpoint true \
            --model.warmup_epoch 20 \
            --model.lr 5e-4 \
            --model.min_lr 1e-6 \
            --model.weight_focal 1.0 \
            --model.weight_msssim 1.0 \
            --model.weight_corr 0.5 \
            --model.weight_csi 1.0 \
            --model.weight_evo 0.5 \
            --model.focal_alpha 2.0 \
            --model.focal_gamma 1.0 \
            --model.false_alarm_penalty 5.0 \
            --data.data_path $DATA_PATH \
            --data.batch_size $BATCH_SIZE \
            --data.num_workers 8
        ;;
        
    "test")
        echo "----------------------------------------"
        echo " [MetAI] Starting Test (Metrics & Evaluation)..."
        echo "----------------------------------------"
        
        # Automatically find best Checkpoint
        CKPT_PATH=$(find $SAVE_DIR -name "*val_score*.ckpt" | sort -V | tail -n 1)
        if [ -z "$CKPT_PATH" ]; then CKPT_PATH=$(find $SAVE_DIR -name "last.ckpt" | head -n 1); fi
        
        if [ -z "$CKPT_PATH" ]; then
            echo " Error: Checkpoint not found"
            exit 1
        fi
        
        echo "Using Checkpoint: $CKPT_PATH"
        
        python run/test_scwds_mamba.py \
            --ckpt_path "$CKPT_PATH" \
            --save_dir "$SAVE_DIR" \
            --data_path "$DATA_PATH" \
            --in_shape 31 256 256 \
            --obs_seq_len 10 \
            --pred_seq_len 20 \
            --num_samples 20 \
            --accelerator cuda:0 \
            
        ;;

    "infer")
        echo "----------------------------------------"
        echo " [MetAI] Starting Inference (Submission Generation)..."
        echo "----------------------------------------"
        
        # Automatically use best/latest model in SAVE_DIR
        # Output saved to ./submit/output
        
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