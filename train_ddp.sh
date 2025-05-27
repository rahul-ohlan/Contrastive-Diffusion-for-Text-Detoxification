#!/bin/bash

# Set the number of processes (GPUs) to use
export WORLD_SIZE=1
export MASTER_PORT=29501

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# Launch training with DDP
torchrun \
    --nnodes=1 \
    --nproc_per_node=$WORLD_SIZE \
    --master_port=$MASTER_PORT \
    src/train_infer/train.py \
    --dataset greetings \
    --train_txt_path data/greetings-train.txt \
    --val_txt_path data/greetings-test.txt \
    --checkpoint_path ckpts/greetings \
    --batch_size 64 \
    --sequence_len 50 \
    --num_channels 128 \
    --num_heads 4 \
    --dropout 0.1 \
    --diffusion_steps 2000 \
    --noise_schedule sqrt \
    --lr 1e-4 \
    --weight_decay 0.0 \
    --lr_anneal_steps 5000 \
    --save_interval 2000 \
    --log_interval 100 \
    --eval_interval 500 \
    --model_arch transformer \
    --training_mode e2e \
    --use_pretrained_embeddings false \
    --init_pretrained false \
    --freeze_embeddings false \
    --seed 10708 \
    --use_wandb true 