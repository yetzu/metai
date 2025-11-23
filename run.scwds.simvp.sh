#!/bin/bash

# SimVP SCWDS 全流程脚本 (Optimized for 4x A800 80GB)
# 包含: Train (SimVP) -> Test (SimVP) -> Train (GAN) -> Test (GAN) -> Infer
# Usage: bash run.scwds.simvp.sh [MODE]

# ================= 环境变量优化 =================
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

# ================= 参数检查 =================
if [ $# -eq 0 ]; then
    echo "错误: 请指定操作模式"
    echo "用法: bash run.scwds.simvp.sh [MODE]"
    echo "支持的模式:"
    echo " train      - 训练 SimVP 基座模型"
    echo " test       - 测试 SimVP 基座模型"
    echo " train_gan  - 基于 SimVP 训练 Refiner GAN (需先完成 train)"
    echo " test_gan   - 测试 GAN 模型"
    echo " infer      - 使用 SimVP 基座进行推理"
    echo " infer_gan  - 使用 GAN 模型进行推理"
    exit 1
fi

MODE=$1

case $MODE in
    # ============================================================
    # 1. 训练 SimVP 基座 (Stage 1)
    # ============================================================
    "train")
        echo "--------------------------------------------------------"
        echo "🚀 [4x A800] 开始训练 SimVP 基座模型 (BF16 Mixed)..."
        echo "--------------------------------------------------------"
        
        python run/train_scwds_simvp.py \
            --data_path data/samples.jsonl \
            --save_dir ./output/simvp \
            \
            --batch_size 4 \
            --accumulate_grad_batches 2 \
            --num_workers 8 \
            \
            --in_shape 20 28 256 256 \
            --max_epochs 100 \
            --opt adamw \
            --lr 1e-4 \
            --sched cosine \
            --min_lr 1e-6 \
            --warmup_epoch 5 \
            \
            --model_type mamba \
            --hid_S 128 \
            --hid_T 256 \
            --N_S 4 \
            --N_T 12 \
            --mlp_ratio 4.0 \
            --drop 0.1 \
            --drop_path 0.1 \
            --spatio_kernel_enc 5 \
            --spatio_kernel_dec 5 \
            \
            --positive_weight 30.0 \
            --sparsity_weight 2.0 \
            --l1_weight 5 \
            --bce_weight 8.0 \
            --ssim_weight 0.5 \
            --evolution_weight 2.0 \
            --bce_weight 15.0 \
            --temporal_weight_enabled true \
            --temporal_weight_max 5.0 \
            --temporal_consistency_weight 0.0 \
            --loss_threshold 0.01 \
            --use_threshold_weights \
            --use_composite_loss true \
            \
            --use_curriculum_learning true \
            --curriculum_warmup_epochs 5 \
            --curriculum_transition_epochs 20 \
            \
            --accelerator cuda \
            --devices 0,1,2,3 \
            --precision bf16-mixed
        ;;
        
    # ============================================================
    # 2. 测试 SimVP 基座 (其余模式保持不变)
    # ============================================================
    "test")
        echo "----------------------------------------"
        echo "🧪 开始测试 SimVP 基座模型..."
        echo "----------------------------------------"
        
        python run/test_scwds_simvp.py \
            --data_path data/samples.jsonl \
            --in_shape 20 28 256 256 \
            --save_dir ./output/simvp \
            --num_samples 10 \
            --accelerator cuda
        ;;
        
    # ... (train_gan, test_gan, infer 模式保持不变)
    "train_gan")
        echo "----------------------------------------"
        echo "🎨 开始 SimVP-GAN 二阶段微调..."
        echo "----------------------------------------"
        
        BACKBONE_CKPT="./output/simvp/last.ckpt"
        if [ ! -f "$BACKBONE_CKPT" ]; then
            BACKBONE_CKPT=$(find ./output/simvp -name "epoch=*.ckpt" | sort -V | tail -n 1)
        fi
        
        if [ ! -f "$BACKBONE_CKPT" ]; then
            echo "❌ 错误: 未找到基座模型权重 (./output/simvp/last.ckpt 或其他)"
            echo "请先运行 'bash run.scwds.simvp.sh train'"
            exit 1
        fi
        
        echo "Using Backbone: $BACKBONE_CKPT"

        mkdir -p ./output/simvp_gan

        python run/gan_train_scwds_simvp.py \
            --data_path data/samples.jsonl \
            --ckpt_path $BACKBONE_CKPT \
            --batch_size 8 \
            --num_workers 16 \
            --max_epochs 50 \
            --lr 1e-4 \
            --lambda_content 100.0 \
            --lambda_adv 0.01 \
            --lambda_fm 10.0 \
            --accelerator cuda \
            --devices 0,1,2,3 \
            --resume_ckpt ./output/simvp_gan/checkpoints/last.ckpt
        ;;

    "test_gan")
        echo "----------------------------------------"
        echo "🧪 开始测试 GAN 模型..."
        echo "----------------------------------------"
        
        python run/gan_test_scwds_simvp.py \
            --data_path data/samples.jsonl \
            --save_dir ./output/simvp_gan \
            --num_samples 10 \
            --accelerator cuda \
            --device cuda
        ;;

    "infer")
        echo "----------------------------------------"
        echo "🔮 开始推理 SimVP 模型..."
        echo "----------------------------------------"
        
        python run/infer_scwds_simvp.py \
            --data_path data/samples.testset.jsonl \
            --in_shape 20 28 256 256 \
            --save_dir ./output/simvp \
            --accelerator cuda \
            --vis \
        ;;
        
    "infer_gan")
        echo "----------------------------------------"
        echo "🎨 开始推理 GAN 模型..."
        echo "----------------------------------------"
        
        python run/gan_infer_scwds_simvp.py \
            --data_path data/samples.testset.jsonl \
            --in_shape 20 28 256 256 \
            --save_dir ./output/simvp \
            --accelerator cuda \
            --vis \
        ;;
    *)
        echo "错误: 不支持的操作模式 '$MODE'"
        echo "支持的模式: train, test, train_gan, test_gan, infer, infer_gan"
        exit 1
        ;;
esac

echo "✅ 操作完成！"