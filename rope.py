# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple

import torch


def build_rope_cache_from_tau(
    tau: torch.Tensor,
    dim: int,
    base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct the RoPE cache based on the continuous time τ
    """
    if dim % 2 == 1:
        dim = dim - 1
    half = dim // 2
    device = tau.device
    dtype = tau.dtype

    idx = torch.arange(half, device=device, dtype=dtype)
    inv = 1.0 / (base ** (idx / half))
    freqs = tau.unsqueeze(-1) * inv.unsqueeze(0)  # (L, half)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, _, _, dh = q.shape
    dh2 = (dh // 2) * 2

    q1, q2 = q[..., :dh2:2], q[..., 1:dh2:2]
    k1, k2 = k[..., :dh2:2], k[..., 1:dh2:2]

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    rq1 = q1 * cos - q2 * sin
    rq2 = q1 * sin + q2 * cos
    rk1 = k1 * cos - k2 * sin
    rk2 = k1 * sin + k2 * cos

    q_rot = torch.zeros_like(q)
    k_rot = torch.zeros_like(k)
    q_rot[..., :dh2:2] = rq1
    q_rot[..., 1:dh2:2] = rq2
    k_rot[..., :dh2:2] = rk1
    k_rot[..., 1:dh2:2] = rk2

    if dh2 < dh:
        q_rot[..., dh2:] = q[..., dh2:]
        k_rot[..., dh2:] = k[..., dh2:]

    return q_rot, k_rot