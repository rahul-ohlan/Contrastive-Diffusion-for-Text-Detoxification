import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from tqdm import tqdm
import wandb

from src.utils import dist_util, logger
from src.utils.fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from src.modeling.diffusion.nn import update_ema
from src.modeling.diffusion.resample import LossAwareSampler, UniformSampler

INITIAL_LOG_LOSS_SCALE = 20.0

class TrainLoopDetox:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        checkpoint_path='',
        gradient_clipping=-1.,
        eval_data=None,
        eval_interval=-1,
        use_wandb=False,
        combined_loss=None,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else [float(x) for x in ema_rate.split(",")]
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.gradient_clipping = gradient_clipping
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size
        self.use_wandb = use_wandb
        self.checkpoint_path = checkpoint_path
        self.eval_data = eval_data
        self.eval_interval = eval_interval
        self.combined_loss = combined_loss
        self.is_rank_0 = dist_util.get_rank() == 0

        self.sync_cuda = th.cuda.is_available()

        self._setup_model_and_optimizer()
        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = self.model if isinstance(self.model, DDP) else DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            self.use_ddp = False
            self.ddp_model = self.model

        self._load_ema_parameters()
        self._load_optimizer_state()

    def _setup_model_and_optimizer(self):
        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)

    def _setup_fp16(self):
        """Initialize FP16 training setup."""
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        if self.is_rank_0:
            pbar = tqdm(total=self.lr_anneal_steps, desc="Training")
            
        while (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
            batch = next(iter(self.data))
            self.run_step(batch)
            
            if self.is_rank_0:
                if self.step % self.log_interval == 0:
                    logger.dumpkvs()
                if self.step % self.save_interval == 0:
                    self.save()
                pbar.update(1)
                
            if self.eval_data is not None and self.step % self.eval_interval == 0:
                self.evaluate()
            
            self.step += 1

    def run_step(self, batch):
        # Get input tensors and prepare model kwargs
        model_kwargs = {
            "toxic_embeddings": batch["toxic_embeddings"].to(dist_util.dev()),
            "toxic_mask": batch["toxic_mask"].to(dist_util.dev())
        }
        clean_embeddings = batch["clean_embeddings"].to(dist_util.dev())
        
        # Sample timesteps
        t, weights = self.schedule_sampler.sample(clean_embeddings.shape[0], dist_util.dev())

        # Forward pass
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.ddp_model,
            clean_embeddings,
            t,
            model_kwargs=model_kwargs
        )

        if self.use_fp16:
            losses = self.forward_backward_fp16(batch, compute_losses)
        else:
            losses = self.forward_backward(batch, compute_losses)

        self.optimize()
        self._anneal_lr()
        self.log_step(losses)
        
        return losses

    def evaluate(self):
        """Run evaluation on validation data"""
        self.model.eval()
        eval_losses = []
        
        with th.no_grad():
            for batch in self.eval_data:
                # Get input tensors and prepare model kwargs
                model_kwargs = {
                    "toxic_embeddings": batch["toxic_embeddings"].to(dist_util.dev()),
                    "toxic_mask": batch["toxic_mask"].to(dist_util.dev())
                }
                clean_embeddings = batch["clean_embeddings"].to(dist_util.dev())
                
                t, weights = self.schedule_sampler.sample(clean_embeddings.shape[0], dist_util.dev())
                
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    clean_embeddings,
                    t,
                    model_kwargs=model_kwargs
                )
                
                losses = compute_losses()
                eval_losses.append((losses["loss"] * weights).mean().item())
        
        avg_loss = sum(eval_losses) / len(eval_losses)
        if self.is_rank_0 and self.use_wandb:
            wandb.log({"eval/loss": avg_loss}, step=self.step)
        
        self.model.train()
        return avg_loss

    def optimize(self):
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self._update_ema()

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(resume_checkpoint, map_location=dist_util.dev())
                )

    def _load_ema_parameters(self):
        self.ema_params = [
            copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
        ]
        self.ema_params = [
            self._load_ema_parameters_from_checkpoint(rate) for rate in self.ema_rate
        ]

    def _load_ema_parameters_from_checkpoint(self, rate):
        ema_params = copy.deepcopy(self.master_params)
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(ema_checkpoint, map_location=dist_util.dev())
            ema_params = self._state_dict_to_master_params(state_dict)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if main_checkpoint:
            opt_checkpoint = bf.join(bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt")
            if bf.exists(opt_checkpoint):
                logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
                state_dict = dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
                self.opt.load_state_dict(state_dict)

    def save(self):
        """Save model, EMA models, and optimizer state."""
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{self.step:06d}.pt"
                else:
                    filename = f"ema_{rate}_{self.step:06d}.pt"
                with bf.BlobFile(bf.join(self.checkpoint_path, filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(self.checkpoint_path, f"opt{self.step:06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                list(self.model.parameters()), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self, losses):
        if self.is_rank_0:
            logger.logkv("step", self.step + self.resume_step)
            logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
            
            if self.use_fp16:
                logger.logkv("lg_loss_scale", self.lg_loss_scale)

            for k, v in losses.items():
                logger.logkv_mean(f"train/{k}", v)
                if self.use_wandb:
                    wandb.log({f"train/{k}": v}, step=self.step)

    def forward_backward(self, batch, compute_losses):
        """Forward and backward pass for normal precision training."""
        zero_grad(self.model_params)
        for i in range(0, batch["clean_embeddings"].shape[0], self.microbatch):
            micro_batch = {k: v[i:i+self.microbatch] for k, v in batch.items()}
            last_batch = (i + self.microbatch) >= batch["clean_embeddings"].shape[0]
            
            # Get input tensors and prepare model kwargs
            model_kwargs = {
                "toxic_embeddings": micro_batch["toxic_embeddings"].to(dist_util.dev()),
                "toxic_mask": micro_batch["toxic_mask"].to(dist_util.dev())
            }
            clean_embeddings = micro_batch["clean_embeddings"].to(dist_util.dev())
            
            # Sample timesteps
            t, weights = self.schedule_sampler.sample(clean_embeddings.shape[0], dist_util.dev())
            
            # Compute diffusion losses
            losses = compute_losses()
            
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            # Compute the main diffusion loss
            diffusion_loss = (losses["loss"] * weights).mean()
            
            # Apply combined loss if available
            if self.combined_loss is not None:
                # Get model output for contrastive loss
                with th.no_grad():
                    model_output = self.ddp_model(clean_embeddings, t, **model_kwargs)
                
                # Compute combined loss
                total_loss, loss_dict = self.combined_loss(
                    diffusion_loss=diffusion_loss,
                    anchor=model_output,
                    positive=clean_embeddings,
                    negative=model_kwargs["toxic_embeddings"]
                )
                
                # Update losses dict for logging
                losses.update(loss_dict)
                losses["total_loss"] = total_loss
                
                # Use total loss for backward
                loss = total_loss
            else:
                loss = diffusion_loss
            
            log_loss_dict = {k: v.mean() if hasattr(v, 'mean') else v for k, v in losses.items()}
            
            if last_batch or not self.use_ddp:
                loss.backward()
            else:
                with self.ddp_model.no_sync():
                    loss.backward()
                    
        return log_loss_dict

    def forward_backward_fp16(self, batch, compute_losses):
        """Forward and backward pass for FP16 training."""
        zero_grad(self.model_params)
        for i in range(0, batch["clean_embeddings"].shape[0], self.microbatch):
            micro_batch = {k: v[i:i+self.microbatch] for k, v in batch.items()}
            last_batch = (i + self.microbatch) >= batch["clean_embeddings"].shape[0]
            
            # Get input tensors and prepare model kwargs
            model_kwargs = {
                "toxic_embeddings": micro_batch["toxic_embeddings"].to(dist_util.dev()),
                "toxic_mask": micro_batch["toxic_mask"].to(dist_util.dev())
            }
            clean_embeddings = micro_batch["clean_embeddings"].to(dist_util.dev())
            
            # Sample timesteps
            t, weights = self.schedule_sampler.sample(clean_embeddings.shape[0], dist_util.dev())
            
            # Compute losses with FP16
            with th.cuda.amp.autocast():
                losses = compute_losses()
                
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            # Compute the main diffusion loss
            diffusion_loss = (losses["loss"] * weights).mean()
            
            # Apply combined loss if available
            if self.combined_loss is not None:
                # Get model output for contrastive loss
                with th.no_grad():
                    model_output = self.ddp_model(clean_embeddings, t, **model_kwargs)
                
                # Compute combined loss
                total_loss, loss_dict = self.combined_loss(
                    diffusion_loss=diffusion_loss,
                    anchor=model_output,
                    positive=clean_embeddings,
                    negative=model_kwargs["toxic_embeddings"]
                )
                
                # Update losses dict for logging
                losses.update(loss_dict)
                losses["total_loss"] = total_loss
                
                # Use total loss for backward with FP16 scaling
                loss = total_loss * (2 ** self.lg_loss_scale)
            else:
                loss = diffusion_loss * (2 ** self.lg_loss_scale)
            
            log_loss_dict = {k: v.mean() if hasattr(v, 'mean') else v for k, v in losses.items()}
            
            if last_batch or not self.use_ddp:
                loss.backward()
            else:
                with self.ddp_model.no_sync():
                    loss.backward()
                    
        return log_loss_dict

def find_resume_checkpoint():
    return None

def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0 