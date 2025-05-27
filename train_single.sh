#!/bin/bash

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# Launch training without DDP
python src/train_infer/train_single.py \
    --dataset simple \
    --train_txt_path data/simple/simple.txt \
    --val_txt_path data/simple/simple.txt \
    --checkpoint_path checkpoints/simple_single \
    --batch_size 24 \
    --sequence_len 64 \
    --num_channels 128 \
    --num_heads 4 \
    --dropout 0.1 \
    --diffusion_steps 100 \
    --noise_schedule cosine \
    --lr 1e-4 \
    --weight_decay 0.0 \
    --lr_anneal_steps 50000 \
    --save_interval 10 \
    --log_interval 10 \
    --eval_interval 50 \
    --model_arch transformer \
    --training_mode e2e \
    --use_pretrained_embeddings False \
    --init_pretrained False \
    --freeze_embeddings False \
    --seed 42 \
    --use_wandb True 