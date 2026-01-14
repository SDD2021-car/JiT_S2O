import math
from typing import Iterable

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=0, lora_alpha=1.0, lora_dropout=0.0, bias=True):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.reset_lora_parameters()
        else:
            self.lora_A = None
            self.lora_B = None

    def reset_lora_parameters(self):
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x):
        out = self.base(x)
        if self.r > 0:
            lora_out = self.lora_dropout(x) @ self.lora_A.t()
            lora_out = lora_out @ self.lora_B.t()
            out = out + lora_out * self.scaling
        return out

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        if self.r > 0:
            return [self.lora_A, self.lora_B]
        return []
