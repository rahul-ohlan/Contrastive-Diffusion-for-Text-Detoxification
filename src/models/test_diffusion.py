"""
Test script to verify diffusion model with cross-attention.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from diffusion_model import DiffusionTransformer
from src.data.detox_dataset import DetoxificationDataset

def test_diffusion_model():
    # Initialize dataset
    dataset = DetoxificationDataset(
        data_path="data/detox_train.jsonl",
        max_length=150
    )
    
    # Create model
    model = DiffusionTransformer(
        input_dim=768,  # BERT hidden size
        hidden_dim=512,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        max_seq_len=150
    )
    
    # Test forward pass
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=4)))
    
    # Add noise to clean embeddings
    noisy_embeddings = batch['clean_embeddings'] + torch.randn_like(batch['clean_embeddings']) * 0.1
    
    # Forward pass
    output = model(
        noisy_embeddings,
        timesteps=torch.zeros(4).long(),  # Batch size 4
        toxic_embeddings=batch['toxic_embeddings'],
        toxic_mask=batch['toxic_mask']
    )
    
    print("\nModel test:")
    print(f"Input shape: {noisy_embeddings.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.3f}")
    print(f"Output std: {output.std().item():.3f}")
    
    # Test cross attention
    print("\nCross attention test:")
    attn_weights = model.get_attention_weights(
        noisy_embeddings,
        toxic_embeddings=batch['toxic_embeddings'],
        toxic_mask=batch['toxic_mask']
    )
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Attention weights sum: {attn_weights.sum(-1).mean().item():.3f}")  # Should be close to 1
    
    # Test gradients
    loss = F.mse_loss(output, batch['clean_embeddings'])
    loss.backward()
    
    # Verify parameter gradients
    print("\nGradient test:")
    total_params = sum(p.numel() for p in model.parameters())
    params_with_grad = sum(p.grad is not None for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(f"Parameters with gradients: {params_with_grad}")

if __name__ == "__main__":
    test_diffusion_model() 