"""
Train a diffusion model for text detoxification.
"""
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

import json, os
import pathlib
import sys
import wandb
from transformers import set_seed
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from src.utils import dist_util, logger
from src.modeling.diffusion.resample import create_named_schedule_sampler
from src.train_infer.factory_methods import create_model_and_diffusion
from src.train_infer.train_loop_detox import TrainLoopDetox
from src.data.detox_dataset import DetoxificationDataset
from src.utils.args_utils import create_argparser, args_to_dict, model_and_diffusion_defaults
from src.modeling.losses.contrastive_loss import CombinedLoss

def setup_wandb(args, is_rank_0):
    """Setup wandb with DDP support"""
    if not is_rank_0:
        os.environ["WANDB_MODE"] = "disabled"
    
    if args.use_wandb:
        wandb_run = wandb.init(
            project="text-detoxification",
            name=f"detox_bs{args.batch_size}_lr{args.lr}",
            config=args.__dict__,
            resume="allow"
        )
        return wandb_run
    return None

def main():
    args = create_argparser().parse_args()
    
    # Add detoxification specific arguments
    args.triplet_weight = 0.1  # Weight for triplet loss
    args.margin = 1.0  # Margin for triplet loss
    
    set_seed(args.seed)
    local_rank = dist_util.setup_dist()
    
    is_rank_0 = dist_util.get_rank() == 0
    if is_rank_0:
        logger.configure()
        logger.log("creating data loader...")
        pathlib.Path(args.checkpoint_path).mkdir(parents=True, exist_ok=True)

    # Create datasets
    train_dataset = DetoxificationDataset(
        data_path="data/detox_train.jsonl",
        max_length=args.sequence_len
    )
    
    val_dataset = DetoxificationDataset(
        data_path="data/detox_valid.jsonl",
        max_length=args.sequence_len
    )

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    if is_rank_0:
        logger.log("creating model and diffusion...")
        wandb_run = setup_wandb(args, is_rank_0)

    try:
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        model.to(dist_util.dev())

        if dist_util.get_world_size() > 1:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
                broadcast_buffers=True
            )
            
        if is_rank_0 and args.use_wandb:
            pytorch_total_params = sum(p.numel() for p in model.parameters())
            logger.log(f"the parameter count is {pytorch_total_params}")
            wandb.log({"total_parameters": pytorch_total_params})
            
            with open(f"{args.checkpoint_path}/training_args.json", "w") as f:
                json.dump(args.__dict__, f, indent=2)

        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
        
        # Create combined loss
        combined_loss = CombinedLoss(
            triplet_weight=args.triplet_weight,
            margin=args.margin
        )

        if is_rank_0:
            logger.log("training...")

        TrainLoopDetox(
            model=model,
            diffusion=diffusion,
            data=train_dataloader,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            resume_checkpoint=args.resume_checkpoint,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
            checkpoint_path=args.checkpoint_path,
            gradient_clipping=args.gradient_clipping,
            eval_data=val_dataloader,
            eval_interval=args.eval_interval,
            use_wandb=args.use_wandb,
            combined_loss=combined_loss,
        ).run_loop()

    except Exception as e:
        if is_rank_0:
            logger.log(f"Error during training: {str(e)}")
        raise e
    finally:
        if is_rank_0 and args.use_wandb:
            wandb.finish()
        if dist_util.get_world_size() > 1:
            dist_util.cleanup_dist()

if __name__ == "__main__":
    main() 