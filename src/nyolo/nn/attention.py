"""Attention implementations selected explicitly before model execution."""

from collections.abc import Callable
from typing import Literal

from torch import Tensor
from torch.nn import functional as F

AttentionBackend = Literal["naive", "sdpa"]
AttentionKernel = Callable[[Tensor, Tensor, Tensor, float], Tensor]


def naive_attention(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
    """Original attention with B,H,D,N inputs and a B,H,Dv,N output."""
    weights = (q.transpose(-1, -2) @ k * scale).softmax(-1)
    return v @ weights.transpose(-1, -2)


def sdpa_attention(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
    """Use PyTorch's native dispatch, preserving unequal Q/K and V dimensions."""
    return F.scaled_dot_product_attention(
        q.transpose(-1, -2).contiguous(),
        k.transpose(-1, -2).contiguous(),
        v.transpose(-1, -2).contiguous(),
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    ).transpose(-1, -2)


def resolve_attention_kernel(backend: AttentionBackend) -> AttentionKernel:
    if backend == "naive":
        return naive_attention
    if backend == "sdpa":
        return sdpa_attention
    raise ValueError(
        f"Unsupported attention backend {backend!r}; choose 'naive' or 'sdpa'"
    )
