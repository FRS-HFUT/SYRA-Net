# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * (x * (x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()))


class LayerScale(nn.Module):
    def __init__(self, d: int, init_value: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = x.new_empty(shape).uniform_(0, 1)
        mask = (rand < keep).float()
        return x * mask / keep




class NormedLinear(nn.Module):
    def __init__(self, d_model: int, num_classes: int, scale_init: float = 16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, d_model))
        nn.init.xavier_uniform_(self.weight)
        self.log_scale = nn.Parameter(torch.log(torch.tensor(scale_init, dtype=torch.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.normalize(self.weight, dim=1)  # (K,D)
        x = F.normalize(x, dim=1)            # (B,D)
        s = torch.exp(self.log_scale)
        return s * (x @ w.t())