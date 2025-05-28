# Text Detoxification with Diffusion Models

This repository implements a text detoxification system using diffusion models with cross-attention for toxic text conditioning.

## Features

- BERT embeddings for text representation
- Cross-attention mechanism for toxic text conditioning
- Distributed training support (DDP) for multi-GPU setups
- Gaussian diffusion process with customizable noise schedules
- WandB integration for experiment tracking

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your data:
Place your paired toxic-clean text data in `data/detox_train.jsonl` with the format:
```json
{"toxic": "toxic text here", "clean": "clean version here"}
```

## Usage

### Testing Components

To verify all components are working:
```bash
python test_all.py
```

### Training

For single GPU:
```bash
python src/train_infer/train.py \
    --data_path data/detox_train.jsonl \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 1e-4
```

For multi-GPU training (DDP):
```bash
python src/train_infer/train_ddp.py \
    --data_path data/detox_train.jsonl \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 1e-4
```

### Inference

To generate detoxified text:
```bash
python src/train_infer/text_sample.py \
    --model_name_or_path checkpoints/checkpoint_99.pt \
    --num_samples 10
```

## Model Architecture

The system consists of the following key components:

1. **Dataset**: `DetoxificationDataset`
   - Handles paired toxic-clean text data
   - Uses BERT for text embeddings
   - Freezes BERT parameters

2. **Diffusion Model**: `DiffusionTransformer`
   - Cross-attention for toxic text conditioning
   - Transformer-based architecture
   - Predicts noise in the diffusion process

3. **Diffusion Process**: `GaussianDiffusion`
   - Implements forward and reverse diffusion
   - Supports linear and cosine noise schedules
   - Handles the denoising process

## Training Process

The training pipeline:
1. Embeds toxic and clean text using BERT
2. Adds noise to clean embeddings
3. Model predicts the noise
4. Uses MSE loss between predicted and actual noise
5. Updates model parameters through backpropagation

## Distributed Training

The system supports distributed training using DistributedDataParallel (DDP):
- Automatically utilizes all available GPUs
- Synchronizes gradients across devices
- Scales batch size per GPU
- Handles proper process initialization

## Monitoring

Training progress is tracked using Weights & Biases (wandb):
- Loss curves
- Sample generations
- Model parameters
- Training configuration

## License

MIT

