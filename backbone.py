# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import LayerScale
from .time_utils import build_time_metadata, load_date_strings


class ShapeTimeEncoder(nn.Module):
    """
    输入:
        x: (B,L) 或 (B,1,L)
    """
    def __init__(
        self,
        in_tokens: int,
        d_model: int,
        ndvi_valid_min: float,
        ndvi_valid_max: float,
        token_drop_prob: float,
        dropout: float = 0.1,
        date_txt_path: Optional[str] = None,
    ):
        super().__init__()
        self.in_tokens = in_tokens
        self.d_model = d_model
        self.ndvi_min = ndvi_valid_min
        self.ndvi_max = ndvi_valid_max
        self.token_drop_prob = token_drop_prob
        self.dropout = dropout

        date_strs = load_date_strings(date_txt_path)
        if in_tokens != len(date_strs):
            raise ValueError(f"[ShapeTimeEncoder] 期望 L={len(date_strs)}，实际 L={in_tokens}。")

        meta = build_time_metadata(date_strs)
        self.num_years = int(meta["num_years"].item())

        self.register_buffer("doy", meta["doy"], persistent=False)
        self.register_buffer("delta_days", meta["delta_days"], persistent=False)
        self.register_buffer("tau_days", meta["tau_days"], persistent=False)
        self.register_buffer("year_ids", meta["year_ids"], persistent=False)

        self.token_embed_raw = nn.Conv1d(1, d_model, kernel_size=1, bias=False)
        self.token_embed_rob = nn.Conv1d(1, d_model, kernel_size=1, bias=False)
        self.token_gate = nn.Linear(2 * d_model, d_model, bias=True)

        self.year_embed = nn.Embedding(num_embeddings=self.num_years, embedding_dim=d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(4, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=False),
        )

        self.token_embed_dx = nn.Conv1d(2, d_model, kernel_size=1, bias=False)
        self.der_fuse = nn.Conv1d(4 * d_model, d_model, kernel_size=1, bias=False)
        self.der_gain = LayerScale(d_model, init_value=1e-4)

        self.temporal_dw = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=5,
            padding=2,
            groups=d_model,
            bias=False,
        )
        self.temporal_pw = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.temporal_gain = LayerScale(d_model, init_value=34)

        nn.init.xavier_uniform_(self.token_embed_raw.weight)
        nn.init.xavier_uniform_(self.token_embed_rob.weight)
        nn.init.xavier_uniform_(self.der_fuse.weight)
        nn.init.kaiming_uniform_(self.temporal_dw.weight, nonlinearity="linear")
        nn.init.xavier_uniform_(self.temporal_pw.weight)

    def _build_time_features(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        doy = self.doy.to(device=device)
        doy_frac = doy / 365.0
        sin_doy = torch.sin(2 * math.pi * doy_frac)
        cos_doy = torch.cos(2 * math.pi * doy_frac)

        delta = self.delta_days.to(device=device)
        tau = self.tau_days.to(device=device)
        delta_norm = delta / (delta.median() + 1e-6)
        tau_norm = tau / (tau.max() + 1e-6)

        base_feat = torch.stack([sin_doy, cos_doy, delta_norm, tau_norm], dim=-1)  # (L,4)
        time_proj = self.time_proj(base_feat)  # (L,D)
        return time_proj, tau, delta


    def forward(self, x: torch.Tensor):
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B,1,L)

        _, _, L = x.shape
        if L != self.in_tokens:
            raise ValueError(f"Input length mismatch: expected {self.in_tokens}，actual {L}")

        device = x.device

        xs = x.squeeze(1)
        M = self._season_token_drop(M)

        h_raw = self.token_embed_raw(x)  # (B,D,L)
        time_proj, tau_seq, _ = self._build_time_features(device)

        h_rob = self.token_embed_rob(z)
        h_cat = torch.cat([h_raw, h_rob], dim=1).transpose(1, 2)  # (B,L,2D)
        gate = torch.sigmoid(self.token_gate(h_cat))
        h_seq = gate * h_raw.transpose(1, 2) + (1.0 - gate) * h_rob.transpose(1, 2)  # (B,L,D)

        dx = F.pad(xs[:, 1:] - xs[:, :-1], (1, 0)).unsqueeze(1)    # (B,1,L)
        d2x = F.pad(dx.squeeze(1)[:, 1:] - dx.squeeze(1)[:, :-1], (1, 0)).unsqueeze(1)
        h_all = torch.cat([h_raw, h_rob, h_dx, h_d2], dim=1)
        h_der = self.der_fuse(h_all).transpose(1, 2)
        h_seq = h_seq + self.der_gain(h_der) * M.unsqueeze(-1).float()

        h_seq = (
            h_seq
            + time_proj.unsqueeze(0)
            + self.year_embed(self.year_ids.to(device)).unsqueeze(0)
        )

        hs = h_seq.transpose(1, 2)  # (B,D,L)
        den = self.temporal_pw(self.temporal_dw(hs)).transpose(1, 2)
        h_seq = h_seq + self.temporal_gain(den * M.unsqueeze(-1).float())

        return x, h_seq, M, tau_seq