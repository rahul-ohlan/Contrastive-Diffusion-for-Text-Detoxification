"""
Single GPU training script for text detoxification.
"""
import os
import torch
from torch.utils.data import DataLoader
import wandb

from src.data.detox_dataset import DetoxificationDataset
from src.models.diffusion_model import DiffusionTransformer
from src.utils.diffusion import GaussianDiffusion

def main():
    import argparse
    parser = argparse.ArgumentParser()
    
    # Data args
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--train_txt_path', type=str, required=True)
    parser.add_argument('--val_txt_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--sequence_len', type=int, default=150)
    
    # Model args
    parser.add_argument('--num_channels', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--model_arch', type=str, default='transformer')
    parser.add_argument('--training_mode', type=str, default='detox')
    parser.add_argument('--init_pretrained', type=bool, default=True)
    parser.add_argument('--freeze_embeddings', type=bool, default=True)
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--noise_schedule', type=str, default='cosine')
    parser.add_argument('--lr_anneal_steps', type=int, default=3650)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--schedule_sampler', type=str, default='uniform')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_wandb', type=bool, default=True)
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_path, exist_ok=True)
    
    # Create dataset
    train_dataset = DetoxificationDataset(
        data_path=args.train_txt_path,
        max_length=args.sequence_len
    )
    val_dataset = DetoxificationDataset(
        data_path=args.val_txt_path,
        max_length=args.sequence_len
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionTransformer(
        input_dim=768,  # BERT embedding dimension
        hidden_dim=args.num_channels,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    # Create diffusion process
    diffusion = GaussianDiffusion(
        timesteps=args.diffusion_steps,
        beta_schedule=args.noise_schedule
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project="text-detoxification",
            config=args
        )
    
    # Training loop
    model.train()
    global_step = 0
    for epoch in range(args.num_epochs):
        total_loss = 0
        
        for batch in train_loader:
            # Move batch to device
            clean_embeddings = batch['clean_embeddings'].to(device)
            toxic_embeddings = batch['toxic_embeddings'].to(device)
            toxic_mask = batch['toxic_mask'].to(device)
            
            # Sample timestep and add noise
            t = torch.randint(0, args.diffusion_steps, (clean_embeddings.shape[0],), device=device)
            noisy_embeddings, noise = diffusion.add_noise(clean_embeddings, t)
            
            # Model prediction
            pred_noise = model(noisy_embeddings, t, toxic_embeddings, toxic_mask)
            
            # Loss
            loss = torch.nn.functional.mse_loss(pred_noise, noise)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            global_step += 1
            
            # Log metrics
            if global_step % args.log_interval == 0:
                avg_loss = total_loss / args.log_interval
                print(f"Step {global_step}, Loss: {avg_loss:.4f}")
                if args.use_wandb:
                    wandb.log({
                        "step": global_step,
                        "loss": avg_loss
                    })
                total_loss = 0
            
            # Save checkpoint
            if global_step % args.save_interval == 0:
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': global_step,
                    'epoch': epoch
                }
                torch.save(
                    state,
                    os.path.join(args.checkpoint_path, f"checkpoint_{global_step}.pt")
                )
        
        print(f"Epoch {epoch} completed")

if __name__ == "__main__":
    main() 