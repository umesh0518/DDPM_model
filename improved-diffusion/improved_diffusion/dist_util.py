"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
# from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 1  # Set to 1 for single-GPU setup

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    # Only initialize the process group if there are multiple GPUs
    if th.cuda.device_count() > 1:
        backend = "gloo" if not th.cuda.is_available() else "nccl"

        if backend == "gloo":
            hostname = "localhost"
        else:
            hostname = socket.gethostbyname(socket.getfqdn())
        os.environ["MASTER_ADDR"] = hostname
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

        port = _find_free_port()
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(backend=backend, init_method="env://")



def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device("cuda:0")  # Default to GPU 0 for single-GPU setup
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)



def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    if dist.is_available() and dist.is_initialized():
        for p in params:
            with torch.no_grad():
                dist.broadcast(p, 0)



def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()