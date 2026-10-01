import torch
import torch.nn as nn
import torch.nn.functional as F

from .router import AdaptiveRouter


class Expert(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class MicroMoE(nn.Module):
    # сотни микро-экспертов вместо пары больших; у heavy скрытый слой шире
    def __init__(self, dim, n_expert, expert_hidden, n_heavy, heavy_mult, k_min, k_max, aux_coef):
        super().__init__()
        self.n_expert = n_expert
        self.aux_coef = aux_coef
        heavy_hidden = expert_hidden * heavy_mult
        self.experts = nn.ModuleList(
            [Expert(dim, expert_hidden) for _ in range(n_expert - n_heavy)]
            + [Expert(dim, heavy_hidden) for _ in range(n_heavy)])
        self.router = AdaptiveRouter(dim, n_expert, k_min, k_max)

    def custom_init(self):
        self.router.custom_init()

    def forward(self, x):
        B, T, D = x.shape
        w, aux, k = self.router(x)
        xf = x.reshape(-1, D)
        wf = w.reshape(-1, self.n_expert)
        y = torch.zeros_like(xf)
        for e, expert in enumerate(self.experts):
            m = wf[:, e] > 0
            if m.any():
                y[m] += expert(xf[m]) * wf[m, e].unsqueeze(-1)
        return y.view(B, T, D), self.aux_coef * aux, k
