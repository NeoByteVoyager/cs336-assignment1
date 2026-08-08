from math import sqrt
from typing import Any, Callable

import torch
from torch.optim.optimizer import ParamsT


class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params: ParamsT,
            lr: torch.float,
            weight_decay: torch.float,
            betas: tuple,
            eps: torch.float
    ):
        default = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps
        }
        super().__init__(params, default)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()

        # main algorithm
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                # get state
                state = self.state[p]
                m = state.get("m", torch.zeros(p.shape, device=p.device)) # tensor
                v = state.get("v", torch.zeros(p.shape, device=p.device))
                t = state.get("t", 1) # initial not 0

                grad = p.grad.data  # get gradient

                lr_t = lr * sqrt(1 - beta2 ** t) / (1 - beta1 ** t)   # compute new_lr
                p.data -= lr * weight_decay * p.data    # Apply weight decay

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)

                p.data -=  lr_t * m / (torch.sqrt(v) + eps) # update original data
                # update state
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss