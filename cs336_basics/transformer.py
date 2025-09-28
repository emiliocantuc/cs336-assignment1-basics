import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange, reduce, einsum
from jaxtyping import Float, Int, Bool


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))

        sigma = 2 / (in_features + out_features)
        nn.init.trunc_normal_(self.weight.data, std=sigma, a=sigma * -3, b=sigma * 3)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einsum(x, self.weight, "... in, out in -> ... out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))

        nn.init.trunc_normal_(self.weight.data, a=-3, b=3)

    def forward(self, token_ids: Int[Tensor, "... 1"]) -> Int[Tensor, "... d_model"]:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.gain = nn.Parameter(torch.ones((d_model,), dtype=dtype, device=device))
        self.eps = eps

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rrms = torch.rsqrt(torch.pow(x, 2).mean(-1, keepdim=True) + self.eps)
        x = x * self.gain * rrms
        return x.to(in_dtype)


def SiLU(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """Computes SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) * W3x),"""

    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.W1 = Linear(d_model, d_ff, device, dtype)
        self.W3 = Linear(d_model, d_ff, device, dtype)
        self.W2 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.W2(SiLU(self.W1(x)) * self.W3(x))


# TODO
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()

    def forward(
        self, x: Float[Tensor, "... seq_len d_k"], token_positions: Float[Float, "... seq_len"]
    ) -> Float[Tensor, "... seq_len d_k"]:
        pass


def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    max_in_dim = x.amax(dim, keepdim=True)
    x = x - max_in_dim
    exps = torch.exp(x)
    return exps / exps.sum(dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, "... q d_k"],
    K: Float[Tensor, "... k d_k"],
    V: Float[Tensor, "... k d_v"],
    mask: Bool[Tensor, "... q k"] | None = None,
) -> Float[Tensor, "... q d_v"]:
    d_k = K.shape[-1]
    x = einsum(K, Q, "... k d_k, ... q d_k -> ... q k") / (d_k**0.5)

    if mask is not None:
        x = x.masked_fill(~mask, -torch.inf)

    x = softmax(x, dim=-1)
    return einsum(x, V, "... q k, ... k d_v -> ... q d_v")


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = None,
        rope: bool = False,  # TODO
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads

        self.W_q = nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        self.W_k = nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        self.W_v = nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        self.W_o = nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))

        sigma = 2 / (d_model + d_model)
        for w in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.trunc_normal_(w.data, std=sigma, a=sigma * -3, b=sigma * 3)

        if max_seq_len is not None:
            self.register_buffer(
                "mask",
                torch.tril(torch.ones((max_seq_len, max_seq_len), dtype=torch.bool, device=device)),
                persistent=False,
            )

    def forward(self, x: Float[Tensor, "... seq_len d_model"]) -> Float[Tensor, "... seq_len d_model"]:
        seq_len = x.shape[-2]
        Q = einsum(x, self.W_q, "... d_in, d_out d_in -> ... d_out")
        K = einsum(x, self.W_k, "... d_in, d_out d_in -> ... d_out")
        V = einsum(x, self.W_v, "... d_in, d_out d_in -> ... d_out")

        Q = rearrange(Q, "... s (h d) -> ... h s d", h=self.num_heads)
        K = rearrange(K, "... s (h d) -> ... h s d", h=self.num_heads)
        V = rearrange(V, "... s (h d) -> ... h s d", h=self.num_heads)

        if hasattr(self, "mask"):
            mask = self.mask[:seq_len, :seq_len]
        else:
            mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))

        out = scaled_dot_product_attention(Q, K, V, mask)
        out = rearrange(out, "... h s d -> ... s (h d)")
        out = einsum(out, self.W_o, "... d_in, d_out d_in -> ... d_out")
        return out
