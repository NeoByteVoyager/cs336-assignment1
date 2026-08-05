

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int,
                 eps: float= 1e-5,
                 device: torch.device| None =None,
                 dtype:torch.dtype | None = None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )
        self.eps = eps
    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x_normal = x * torch.rsqrt(torch.mean(x ** 2, -1, keepdim=True) + self.eps)
        res = x_normal * self.weight
        return res.to(in_dtype)
if __name__ == "__main__":
    rms = RMSNorm(8)
    x = torch.randn(2, 4, 8)
    print(rms(x))