import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveRouter(nn.Module):
    # сложность токена ≈ энтропия гейта: уверенные токены берут k_min экспертов, мутные — до k_max
    def __init__(self, dim, n_expert, k_min=1, k_max=4, noise=0.02):
        super().__init__()
        self.n_expert = n_expert
        self.k_min = k_min
        self.k_max = k_max
        self.noise = noise
        self.gate = nn.Linear(dim, n_expert, bias=False)
        self.diff = nn.Sequential(nn.Linear(dim, dim // 4), nn.GELU(), nn.Linear(dim // 4, 1))

    def custom_init(self):
        nn.init.normal_(self.diff[-1].weight, std=0.01)
        nn.init.constant_(self.diff[-1].bias, -2.0)

    def forward(self, x):
        g = self.gate(x)
        if self.training and self.noise > 0:
            g = g + torch.randn_like(g) * self.noise
        p = torch.softmax(g.float(), dim=-1)
        ent = -(p * (p + 1e-9).log()).sum(-1) / math.log(self.n_expert)
        d = torch.sigmoid(self.diff(x).squeeze(-1))
        aux_d = F.mse_loss(d, ent.detach())
        kf = self.k_min + d * (self.k_max - self.k_min)
        k = torch.round(kf).long()
        order = g.argsort(dim=-1, descending=True)
        pos = torch.arange(self.n_expert, device=x.device).view(1, 1, -1)
        sel = torch.zeros_like(g, dtype=torch.bool).scatter(-1, order, pos < k.unsqueeze(-1))
        w = g.float().masked_fill(~sel, float("-inf")).softmax(dim=-1)
        f = sel.float().mean(dim=(0, 1))
        aux_lb = self.n_expert * (f * p.mean(dim=(0, 1))).sum()
        return w, aux_lb + aux_d, k.float().mean()
