import torch
import torch.nn as nn

from cs336_basics.linear import Linear

def siLU(x: torch.Tensor):
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.proj1 = Linear(d_model, d_ff)
        self.proj2 = Linear(d_ff, d_model)
        self.proj3 = Linear(d_model, d_ff)
    def forward(self, x):
        x1 = siLU(self.proj1(x))
        x2 = self.proj3(x)

        return self.proj2(x1 * x2)