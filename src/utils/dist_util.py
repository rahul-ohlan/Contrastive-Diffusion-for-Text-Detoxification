"""
Helpers for distributed training.
"""

import io
import os
import socket
import torch as th
import torch.distributed as dist

# Change this to reflect your device (CPU or CUDA)
CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES', '')

def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:{th.cuda.current_device()}")
    return th.device("cpu")

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    return th.load(path, **kwargs)

def sync_params(params):
    """
    Synchronize parameters across ranks.
    """
    if not dist.is_initialized():
        return
    for p in params:
        with th.no_grad():
            dist.broadcast(p.data, src=0)

def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if th.cuda.is_available():
        th.cuda.set_device(local_rank)  # Set device based on local_rank

    if world_size >= 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        synchronize()
    return

def cleanup_dist():
    """
    Cleanup the distributed environment.
    """
    if dist.is_initialized():
        dist.destroy_process_group()

def synchronize():
    """
    Helper function to synchronize between multiple processes when using distributed training.
    """
    if not dist.is_initialized():
        return
    dist.barrier()

def get_rank():
    """
    Get the rank of the current process.
    """
    if dist.is_initialized():
        return dist.get_rank()
    return 0

def get_world_size():
    """
    Get the world size (total number of processes).
    """
    if dist.is_initialized():
        return dist.get_world_size()
    return 1
