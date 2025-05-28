#!/bin/bash

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# Launch training on single GPU
CUDA_VISIBLE_DEVICES=0 python src/train_infer/train_ddp.py \
    --dataset detox \
    --train_txt_path data/detox_train.jsonl \
    --val_txt_path data/detox_valid.jsonl \
    --checkpoint_path ckpts/detox \
    --batch_size 32 \
    --sequence_len 150 \
    --num_channels 512 \
    --num_heads 8 \
    --dropout 0.1 \
    --diffusion_steps 1000 \
    --noise_schedule cosine \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --lr_anneal_steps 3650 \
    --save_interval 1000 \
    --log_interval 100 \
    --eval_interval 1000 \
    --model_arch transformer \
    --training_mode detox \
    --init_pretrained true \
    --freeze_embeddings true \
    --seed 42 \
    --schedule_sampler uniform \
    --use_wandb true 