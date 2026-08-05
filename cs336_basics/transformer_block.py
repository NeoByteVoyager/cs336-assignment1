import torch
import torch.nn as nn
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.attention import multihead_self_attention_with_rope
from cs336_basics.SwiGLU import SwiGLU

class Transformer(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff,
            max_seq_len,
            theta,
    ):
        super().__init__()
        self.RMSNorm1 = RMSNorm(d_model)
        self.RMSNorm2 = RMSNorm(d_model)

        self.attention = multihead_self_attention_with_rope(d_model, num_heads, max_seq_len, theta)
        self.SwiGLU = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor):
        attention_out = self.attention(self.RMSNorm1(x))
        x = x + attention_out
        return x + self.SwiGLU(self.RMSNorm2(x))

if __name__ == "__main__":
    x = torch.randn(2, 4, 8)
    model = Transformer(8, 2, 64, 100, 10000)
    print(model(x).shape)