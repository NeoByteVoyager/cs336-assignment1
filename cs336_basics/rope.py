import torch
import torch.nn as nn

class Rope(nn.Module):
    def __init__(self,
                 theta: float,
                 d_k: int,
                 seq_len: int,
                 device:torch.device | None = None):
        super().__init__()

        inv_freq = theta ** (- torch.arange(0, d_k, 2, device=device).float() / d_k) # bug
        pos = torch.arange(0, seq_len, 1, device=device).float() # float
        freqs = torch.outer(pos, inv_freq)

        self.register_buffer("cos", torch.cos(freqs))
        self.register_buffer("sin",  torch.sin(freqs))

    def forward(self, x, token_positions):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        cos = self.cos[token_positions] # index
        sin = self.sin[token_positions]

        x_rot = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

        return x_rot.flatten(-2) # -2 don not know the former dim

if __name__ == "__main__":
    rope = Rope(10000, 4, 4)
    print(rope.sin)
    print(rope.cos)
    x = torch.randn(2, 2, 4, 4)
    token_position = torch.arange(0, 4)
    print(rope(x, token_position).shape)