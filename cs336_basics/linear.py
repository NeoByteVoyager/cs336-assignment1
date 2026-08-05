import torch
import torch.nn as nn
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features: int,
                 out_features: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None =None):
        super().__init__()
        std = (2 / (in_features + out_features)) ** 0.5
        self.weights = nn.Parameter(
            torch.nn.init.trunc_normal_(torch.randn(out_features, in_features, device=device, dtype=dtype),
                                        std=std,
                                        a=-3 * std,
                                        b=3 * std)
        )

    def forward(self, x):
        # -1 dimension linear transform
        return einsum(x, self.weights, "... in_dim, out_dim in_dim -> ... out_dim")

if __name__ == "__main__":
    model = Linear(2, 3)
    x = torch.ones(4, 2)
    print(model(x))
