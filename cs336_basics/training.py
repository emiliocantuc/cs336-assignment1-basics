import numpy as np
import torch
import os
from typing import BinaryIO, IO


def get_batch(x: np.array, batch_size: int, context_length: int, device: str, sample: bool = True):
    # clever: use broadcasting to create the input and target sequences
    # and avoid appending to lists and then stacking

    if sample:
        starts = np.random.choice(np.arange(len(x) - context_length), size=batch_size, replace=False)
    else:
        starts = np.arange(0, len(x) - context_length, context_length)[:batch_size]  # TODO test
    offsets = np.arange(context_length)

    ixs = starts[:, None] + offsets  # (batch_size, context_length)
    inputs = torch.as_tensor(x[ixs], dtype=torch.long, device=device)
    targets = torch.as_tensor(x[ixs + 1], dtype=torch.long, device=device)
    return inputs, targets


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
