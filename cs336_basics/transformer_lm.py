import torch
import torch.nn as nn
from cs336_basics.transformer_block import Transformer
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.RMSNorm import RMSNorm

class Model(nn.Module):
    def __init__(
            self,
            vocab_size,
            context_length,
            d_model,
            num_layers,
            num_heads,
            d_ff,
            rope_theta
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            Transformer(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.layers:
            x = block(x)
        x = self.norm(x)
        x = self.linear(x)
        return x

if __name__ == "__main__":
    x = torch.arange(0, 8)
    x = x.view(2, 4)
    print(x)
    model = Model(100, 10, 8, 3, 2, 64, 10000)
    print(model(x).shape)

    for name, param in model.named_parameters():
        print(name, param.shape)
