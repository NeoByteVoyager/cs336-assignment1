import torch
import torch.nn as nn
from einops import einsum, rearrange

def softmax(x: torch.Tensor, dim:int):
    max_val, _ = torch.max(x, dim=dim, keepdim=True)
    x = x - max_val

    exp_x = torch.exp(x)

    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

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

class Rope(nn.Module):
    def __init__(self,
                 theta: float,
                 d_k: int,
                 seq_len: int):
        super().__init__()

        inv_freq = theta ** (- torch.arange(0, d_k, 2).float() / d_k) # bug
        pos = torch.arange(0, seq_len, 1).float() # float
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
    x = torch.randn(2, 4, 8)
    model = Transformer(8, 2, 64, 100, 10000)
    print(model(x).shape)