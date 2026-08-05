import torch
from einops import einsum, rearrange
from cs336_basics.softmax import softmax
import torch.nn as nn
from cs336_basics.linear import Linear
from cs336_basics.rope import Rope

def scaled_dot_product_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None
):
    '''
    returns:
        v: [..., d_v]
    '''
    d_k = q.shape[-1]
    attention_map = einsum(q, k, "... seq1 d_k, ... seq2 d_k -> ... seq1 seq2") / (d_k ** 0.5)
    if mask is not None:
        attention_map = attention_map.masked_fill(~mask, float("-inf"))
    scaled_attention = softmax(attention_map, -1)

    return einsum(scaled_attention, v, "... seq1 seq2, ... seq2 d_v -> ... seq1 d_v")

class multihead_self_attention(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # Wq, Wk, Wv
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        # Wo
        self.proj = Linear(d_model, d_model)

    def forward(self, x):
        seq = x.shape[1]
        # transform
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        # batch num_head seq dim
        Q = rearrange(Q, "batch seq (num_heads dim) -> batch num_heads seq dim", num_heads=self.num_heads)
        K = rearrange(K, "batch seq (num_heads dim) -> batch num_heads seq dim", num_heads=self.num_heads)
        V = rearrange(V, "batch seq (num_heads dim) -> batch num_heads seq dim", num_heads=self.num_heads)

        # self_attention
        mask = torch.tril(torch.ones(seq, seq, device=x.device)).bool()
        attention_out = scaled_dot_product_attention(Q, K, V, mask)

        out = rearrange(attention_out, "batch num_heads seq dim -> batch seq (num_heads dim)")
        return self.proj(out)

class multihead_self_attention_with_rope(multihead_self_attention):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 max_seq_len: int,
                 theta: float,
                 ):
        super().__init__(d_model, num_heads)
        # rope
        self.rope = Rope(theta, d_model // num_heads, max_seq_len)
        # mask
        self.mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()

    def forward(self, x, token_positions: torch.Tensor = None):
        seq = x.shape[1]
        # transform
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        # batch num_head seq dim
        Q = rearrange(Q, "batch seq (num_heads dim) -> batch num_heads seq dim", num_heads=self.num_heads)
        K = rearrange(K, "batch seq (num_heads dim) -> batch num_heads seq dim", num_heads=self.num_heads)
        V = rearrange(V, "batch seq (num_heads dim) -> batch num_heads seq dim", num_heads=self.num_heads)
        if token_positions is None:
            token_positions = torch.arange(0, seq, device=x.device)

        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        # self_attention
        mask = self.mask[:seq, :seq]
        attention_out = scaled_dot_product_attention(Q, K, V, mask)

        out = rearrange(attention_out, "batch num_heads seq dim -> batch seq (num_heads dim)")
        return self.proj(out)

if __name__ == "__main__":
    q = torch.randn(2, 3, 4)
    k = torch.randn(2, 3, 4)
    v = torch.randn(2, 3, 4)
    print(scaled_dot_product_attention(q, k, v))
    model = multihead_self_attention(8, 2,)
    x = torch.randn(2, 4, 8)
    print(model(x).shape)
    print(model)
    model = multihead_self_attention_with_rope(8, 2, 100, 1000)
    print(model)
