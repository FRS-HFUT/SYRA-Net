# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DropPath, LayerScale, RMSNorm
from .rope import apply_rope


class MaskedGatedAttnPool(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, temp: float = 1.5):
        super().__init__()
        self.score = nn.Linear(d_model, 1, bias=False)
        self.gate = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.temp = temp

    def forward(self, h_seq: torch.Tensor, mask_bool: torch.Tensor) -> torch.Tensor:
        # h_seq:(B,L,D), mask_bool:(B,L)
        scores = self.score(h_seq).squeeze(-1)                 # (B,L)
        scores = scores.masked_fill(~mask_bool, float("-inf"))
        alpha = F.softmax(scores / self.temp, dim=-1).unsqueeze(-1)  # (B,L,1)
        gated = torch.sigmoid(self.gate(h_seq)) * h_seq              # (B,L,D)
        return self.drop((alpha * gated).sum(dim=1))                 # (B,D)


class BiGLSTMCell(nn.Module):
    """可感知 mask 的双向 LSTM 记忆分支"""
    def __init__(self, d_model: int):
        super().__init__()
        self.in_proj = nn.Linear(d_model, 4 * d_model, bias=True)
        self.out_proj = nn.Linear(2 * d_model, d_model, bias=False)

    def scan(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor], direction: str = "fwd") -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        h_list = []
        c = x.new_zeros(bsz, dim)
        h_prev = x.new_zeros(bsz, dim)
        rng = range(seq_len) if direction == "fwd" else range(seq_len - 1, -1, -1)

        for t in rng:
            xt = x[:, t, :]
            i, f, o, g = self.in_proj(xt).chunk(4, dim=-1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            o = torch.sigmoid(o)
            g = torch.tanh(g)
            c_new = f * c + i * g
            h_new = o * torch.tanh(c_new)

            if valid_mask is not None:
                m = valid_mask[:, t].float().unsqueeze(-1)
                c = m * c_new + (1.0 - m) * c
                h_prev = m * h_new + (1.0 - m) * h_prev
                h_list.append(h_prev)
            else:
                c = c_new
                h_prev = h_new
                h_list.append(h_prev)

        if direction != "fwd":
            h_list = h_list[::-1]
        return torch.stack(h_list, dim=1)

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        h_f = self.scan(x, valid_mask, "fwd")
        h_b = self.scan(x, valid_mask, "bwd")
        h = torch.cat([h_f, h_b], dim=-1)
        return self.out_proj(h)


class TransMemBlock(nn.Module):
    """RoPE-MHSA + BiGLSTM Memory + SwiGLU-FFN"""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_mult: float = 4.0,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        ls_init: float = 1e-4,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MHSA_RoPE(d_model, n_heads, attn_dropout, proj_dropout=ffn_dropout)
        self.ls1 = LayerScale(d_model, init_value=ls_init)
        self.dp1 = DropPath(drop_path)

        self.norm_mem = RMSNorm(d_model)
        self.memory = BiGLSTMCell(d_model)
        self.ls_mem = LayerScale(d_model, init_value=ls_init)
        self.dp_mem = DropPath(drop_path)

        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU_FFN(d_model, hidden_mult=ffn_mult, dropout=ffn_dropout)
        self.ls2 = LayerScale(d_model, init_value=ls_init)
        self.dp2 = DropPath(drop_path)

    def forward(
        self,
        x: torch.Tensor,
        rope_cache=None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        xa = self.attn(self.norm1(x), rope_cache=rope_cache, key_padding_mask=key_padding_mask)
        x = x + self.dp1(self.ls1(xa))

        xm = self.memory(self.norm_mem(x), key_padding_mask)
        x = x + self.dp_mem(self.ls_mem(xm))

        xf = self.ffn(self.norm2(x))
        x = x + self.dp2(self.ls2(xf))
        return x