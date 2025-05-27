"""
Train a diffusion model on images (Single GPU version).
"""

import json
import pathlib
import wandb
from transformers import set_seed
import torch

from src.utils import logger
from src.modeling.diffusion.resample import create_named_schedule_sampler
from src.train_infer.factory_methods import create_model_and_diffusion
from src.train_infer.train_loop_single import TrainLoop
from src.utils import data_utils_sentencepiece
from src.utils.args_utils import create_argparser, args_to_dict, model_and_diffusion_defaults
from src.utils.custom_tokenizer import create_tokenizer


def make_wandb_name_from_args(args):
    """Create a unique wandb run name from args"""
    keys_to_add = ["batch_size", "lr", "num_heads", "lr_anneal_steps", "model_arch", "seed"]
    name = f"{args.dataset}_"
    for key in keys_to_add:
        name += f"{key}={getattr(args, key)}_"
    return name.rstrip('_')

def setup_wandb(args):
    """Setup wandb"""
    if args.use_wandb:
        wandb_run = wandb.init(
            project="minimal-text-diffusion",
            entity="cb-solo",
            name=make_wandb_name_from_args(args),
            config=args.__dict__,
            resume="allow"
        )
        return wandb_run
    return None

def main():
    args = create_argparser().parse_args()
    set_seed(args.seed)
    
    # Configure logging
    logger.configure()
    logger.log("creating data loader...")
    pathlib.Path(args.checkpoint_path).mkdir(parents=True, exist_ok=True)

    # Create tokenizer
    tokenizer = create_tokenizer(
        return_pretokenized=args.use_pretrained_embeddings, 
        path=f"data/{args.dataset}/"
    )

    # Create dataloaders
    train_dataloader = data_utils_sentencepiece.get_dataloader(
        tokenizer=tokenizer,
        data_path=args.train_txt_path,
        batch_size=args.batch_size,
        max_seq_len=args.sequence_len
    )

    val_dataloader = data_utils_sentencepiece.get_dataloader(
        tokenizer=tokenizer,
        data_path=args.val_txt_path,
        batch_size=args.batch_size,
        max_seq_len=args.sequence_len
    )

    args.vocab_size = tokenizer.vocab_size
    logger.log("creating model and diffusion...")
    
    # Initialize wandb
    wandb_run = setup_wandb(args)

    try:
        # Create model and diffusion
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
            
        if args.use_wandb:
            print(model)
            pytorch_total_params = sum(p.numel() for p in model.parameters())
            logger.log(f"the parameter count is {pytorch_total_params}")
            wandb.log({"total_parameters": pytorch_total_params})
            
        # Save hyperparameters
        logger.log(f"saving the hyperparameters to {args.checkpoint_path}/training_args.json")
        with open(f"{args.checkpoint_path}/training_args.json", "w") as f:
            json.dump(args.__dict__, f, indent=2)

        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
        logger.log("training...")

        # Initialize training loop
        TrainLoop(
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
        ).run_loop()

    except Exception as e:
        logger.log(f"Error during training: {str(e)}")
        raise e
    finally:
        if args.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main() 