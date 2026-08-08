import os
from typing import BinaryIO

import torch
import torch.nn as nn
import torch.optim as optim

from cs336_basics.transformer_lm import Model
from cs336_basics.adamw import AdamW

def save_checkpoint(
        model: nn.Module,
        optimizer: optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(
        src: str | os.PathLike | BinaryIO,
        model: nn.Module,
        optimizer: optim.Optimizer
):
    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["iteration"]

if __name__ == "__main__":
    x = torch.arange(0, 8)
    x = x.view(2, 4)
    print(x)
    model = Model(100, 10, 8, 3, 2, 64, 10000)

    optimizer = AdamW(model.parameters(),0.01, 1e-5, (0.9, 0.99), 1e-8)

    save_checkpoint(model, optimizer, 1, "checkpoint.txt")