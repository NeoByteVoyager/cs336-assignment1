from typing import Iterable
import torch
import torch.nn as nn


def gradientClipping(
        parameters: Iterable[nn.Parameter],
        max_l2_normal: float,
        eps: float = 1e-6
):
    parameters = list(parameters) # can not traverse twice

    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm += torch.sum(p.grad ** 2)

    total_norm = torch.sqrt(total_norm)

    if total_norm >= max_l2_normal:
        for p in parameters:
            if p.grad is not None:
                p.grad *= max_l2_normal / (total_norm + eps)


