# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

class TransNDVIClassifier(nn.Module):

    def __init__(
        self,
        in_tokens: int,
        num_classes: int,
        d_model: int = 128,
        depth: int = 6,
        n_heads: int = 4,
        ffn_mult: float = 4.0,
        dropout: float = 0.1,
        cls_token: bool = True,
        rope_base: float = 10000.0,
        ndvi_valid_min: float = -1.2,
        ndvi_valid_max: float = 1.2,
        enable_moe: bool = True,
        num_experts: int = 3,
        router_temp: float = 1.0,
        moe_aux_coef: float = 0.0,
        tv_coef: float = 0.0,
        drop_path_rate: float = 0.10,
        token_drop_prob: float = 0.10,
        date_txt_path: Optional[str] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_tokens = in_tokens
        self.d_model = d_model
        self.date_txt_path = date_txt_path

        self.ShapeTimeEncoder = ShapeTimeEncoder(
            in_tokens=in_tokens,
            d_model=d_model,
            ndvi_valid_min=ndvi_valid_min,
            ndvi_valid_max=ndvi_valid_max,
            token_drop_prob=token_drop_prob,
            dropout=dropout,
            date_txt_path=date_txt_path,
        )

        self.TimePhaseBackbone = TimePhaseBackbone(
            in_tokens=in_tokens,
            d_model=d_model,
            depth=depth,
            n_heads=n_heads,
            ffn_mult=ffn_mult,
            dropout=dropout,
            cls_token=cls_token,
            rope_base=rope_base,
            drop_path_rate=drop_path_rate,
            date_txt_path=date_txt_path,
        )

        self.PhaseMoEHead = PhaseMoEHead(
            d_model=d_model,
            num_classes=num_classes,
            dropout=dropout,
            enable_moe=enable_moe,
            num_experts=num_experts,
            router_temp=router_temp,
            moe_aux_coef=moe_aux_coef,
            tv_coef=tv_coef,
        )

        self.last_moe_aux: Optional[torch.Tensor] = None
        self.last_tv: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, h_seq, M, tau_seq = self.ShapeTimeEncoder(x)
        cls_vec, phase_vec, seq_part, M = self.TimePhaseBackbone(h_seq, M, tau_seq)
        logits = self.PhaseMoEHead(cls_vec, phase_vec, seq_part, M)

        self.last_moe_aux = self.PhaseMoEHead.last_moe_aux
        self.last_tv = self.PhaseMoEHead.last_tv
        return logits

