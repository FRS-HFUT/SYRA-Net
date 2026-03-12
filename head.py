# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import MaskedGatedAttnPool
from .common import NormedLinear, TinyExpert


class PhaseMoEHead(nn.Module):
    """
    """
    def __init__(
        self,
        d_model: int,
        num_classes: int,
        dropout: float = 0.1,
        enable_moe: bool = True,
        num_experts: int = 3,
        router_temp: float = 1.0,
        moe_aux_coef: float = 0.0,
        tv_coef: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        self.enable_moe = enable_moe
        self.num_experts = num_experts
        self.router_temp = router_temp
        self.moe_aux_coef = moe_aux_coef
        self.tv_coef = tv_coef

        self.seq_readout = MaskedGatedAttnPool(d_model, dropout=dropout, temp=1.5)

        self.rep_fuse = nn.Sequential(
            nn.Linear(3 * d_model, d_model, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model, bias=False),
        )

        if self.enable_moe:
            self.router = nn.Linear(d_model * 3, num_experts, bias=False)
            self.experts = nn.ModuleList([
                TinyExpert(d_model, hidden_mult=0.5, dropout=dropout)
                for _ in range(num_experts)
            ])
            self.register_buffer("router_usage", torch.full((num_experts,), 1.0 / num_experts))
        else:
            self.router = None
            self.experts = None
            self.register_buffer("router_usage", torch.ones(1), persistent=False)

        self.head = NormedLinear(d_model, num_classes, scale_init=16.0)
        self.last_moe_aux: Optional[torch.Tensor] = None
        self.last_tv: Optional[torch.Tensor] = None

    def _forward_simple(self, seq_part: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        self.last_moe_aux = None
        self.last_tv = None
        M_f = M.to(dtype=seq_part.dtype).unsqueeze(-1)
        sum_w = M_f.sum(dim=1).clamp_min(1e-6)
        rep_vec = (seq_part * M_f).sum(dim=1) / sum_w
        return self.head(rep_vec)

    def forward(
        self,
        cls_vec: torch.Tensor,
        phase_vec: torch.Tensor,
        seq_part: torch.Tensor,
        M: torch.Tensor,
        enable_advanced: bool = True,
    ) -> torch.Tensor:
        if not enable_advanced:
            return self._forward_simple(seq_part, M)

        attn_vec = self.seq_readout(seq_part, M)
        rep_vec = self.rep_fuse(torch.cat([cls_vec, attn_vec, phase_vec], dim=-1))

        if self.enable_moe and self.router is not None and self.experts is not None:
            router_in = torch.cat([phase_vec, attn_vec, cls_vec], dim=-1)
            router_logits = self.router(router_in) / max(1e-6, self.router_temp)
            usage = self.router_usage.clamp_min(1e-6)
            router_logits = router_logits - torch.log(usage.unsqueeze(0))
            gate = F.softmax(router_logits, dim=-1)

            if self.training:
                with torch.no_grad():
                    self.router_usage.mul_(0.99).add_((1.0 - 0.99) * gate.mean(dim=0))
                if self.moe_aux_coef > 0:
                    E = gate.size(-1)
                    uni = torch.full_like(gate, 1.0 / E)
                    kl = (gate * (gate.clamp_min(1e-8).log() - uni.log())).sum(dim=-1).mean()
                    self.last_moe_aux = self.moe_aux_coef * kl
                else:
                    self.last_moe_aux = None
            else:
                self.last_moe_aux = None

            expert_res = 0.0
            for e, expert in enumerate(self.experts):
                expert_res = expert_res + gate[:, e:e + 1] * expert(rep_vec)
            rep_vec = rep_vec + expert_res
        else:
            self.last_moe_aux = None

        logits = self.head(rep_vec)

        if self.training and self.tv_coef > 0:
            diff = seq_part[:, 1:, :] - seq_part[:, :-1, :]
            m2 = (M[:, 1:] & M[:, :-1]).float()
            tv = (diff.pow(2).mean(dim=-1) * m2).sum() / m2.sum().clamp_min(1.0)
            self.last_tv = self.tv_coef * tv
        else:
            self.last_tv = None

        return logits