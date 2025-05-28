"""
Test script to verify all components of the text detoxification system.
"""
import torch
import os
import sys
from src.data.detox_dataset import DetoxificationDataset
from src.models.diffusion_model import DiffusionTransformer
from src.utils.diffusion import GaussianDiffusion

def test_components():
    # Set device at the start
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "data/detox_test.jsonl"  # Correct path to the test data file
    
    # Debug file existence and size
    print(f"\nChecking data file...")
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} does not exist!")
        sys.exit(1)
    
    print(f"File exists: {os.path.exists(data_path)}")
    print(f"File size: {os.path.getsize(data_path)} bytes")
    
    # Debug file content
    print("\nFirst few lines of data file:")
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 3 and line.strip():  # Print first 3 non-empty lines
                print(f"Line {i+1}: {line.strip()}")
    
    print("\nTesting dataset...")
    dataset = DetoxificationDataset(
        data_path=data_path,
        max_length=150
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Test single item
    item = dataset[0]
    print("\nSample item:")
    print(f"Toxic text: {item['toxic_text']}")
    print(f"Clean text: {item['clean_text']}")
    print(f"Embeddings shape: {item['clean_embeddings'].shape}")
    
    print("\nTesting model...")
    model = DiffusionTransformer(
        input_dim=768,  # BERT hidden size
        hidden_dim=512,
        num_heads=8,
        num_layers=6,
        dropout=0.1
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    print("\nTesting diffusion...")
    diffusion = GaussianDiffusion(timesteps=1000, device=device)  # Pass device to diffusion
    
    # Test batch processing
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))
    
    # Move to device
    clean_embeddings = batch['clean_embeddings'].to(device)
    toxic_embeddings = batch['toxic_embeddings'].to(device)
    toxic_mask = batch['toxic_mask'].to(device)
    
    print("\nTesting forward pass...")
    # Sample timestep and add noise - create t on CPU for diffusion indexing
    t = torch.randint(0, 1000, (clean_embeddings.shape[0],))
    noisy_embeddings, noise = diffusion.add_noise(clean_embeddings, t)
    
    # Move t to GPU for model forward pass
    t = t.to(device)
    
    # Model prediction
    pred_noise = model(noisy_embeddings, t, toxic_embeddings, toxic_mask)
    print(f"Prediction shape: {pred_noise.shape}")
    
    # Test loss
    loss = torch.nn.functional.mse_loss(pred_noise, noise)
    print(f"Loss: {loss.item():.4f}")
    
    print("\nTesting backward pass...")
    loss.backward()
    print("Backward pass successful")
    
    print("\nTesting sampling...")
    # Generate sample
    sample = diffusion.p_sample_loop(
        model,
        shape=clean_embeddings.shape,
        toxic_embeddings=toxic_embeddings,
        toxic_mask=toxic_mask
    )
    print(f"Generated sample shape: {sample.shape}")
    
    print("\nAll components verified successfully!")

if __name__ == "__main__":
    test_components() 