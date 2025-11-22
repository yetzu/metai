# 1. 运行通道重要性分析（基于数据相关性，不依赖模型）
# 使用流式处理模式支持大样本量（自动启用，因为num_samples > 500）
python prework/run.analyze.channel.py \
    --is_debug True \
    --data_path data/samples.jsonl \
    --in_shape 20 30 301 301 \
    --task_mode precipitation \
    --num_samples 200 \
    --method pearson \
    --device cuda \
    --output_dir output/channel_analysis
