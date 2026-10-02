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
        self.n_heavy = n_heavy
        self.aux_coef = aux_coef
        heavy_hidden = expert_hidden * heavy_mult
        self.experts = nn.ModuleList(
            [Expert(dim, expert_hidden) for _ in range(n_expert - n_heavy)]
            + [Expert(dim, heavy_hidden) for _ in range(n_heavy)])
        self.router = AdaptiveRouter(dim, n_expert, k_min, k_max)

    def custom_init(self):
        self.router.custom_init()

    def _grouped(self, xf, tok, exp, lo, hi):
        # gather-маппинг: пары (токен, эксперт) -> сортировка -> паддинг -> bmm вместо цикла
        # пары приходят отсортированными по эксперту из forward
        E, M, D = hi - lo, tok.numel(), xf.shape[1]
        cnt = torch.bincount(exp, minlength=E)
        mp = int(cnt.max())
        if E * mp * D > (64 << 20) // 4:  # экстремальный перекос роутера — дорогой паддинг
            return self._loop(xf, tok, exp, lo, hi)
        ptr = torch.cat([cnt.new_zeros(1), cnt.cumsum(0)])
        slot = exp * mp + (torch.arange(M) - ptr[exp])
        X = xf.new_zeros(E * mp, D)
        X[slot] = xf[tok]
        es = self.experts[lo:hi]
        w1 = torch.stack([e.fc1.weight for e in es])
        b1 = torch.stack([e.fc1.bias for e in es])
        w2 = torch.stack([e.fc2.weight for e in es])
        b2 = torch.stack([e.fc2.bias for e in es])
        H = F.gelu(torch.baddbmm(b1.unsqueeze(1), X.view(E, mp, D), w1.transpose(1, 2)))
        Y = torch.baddbmm(b2.unsqueeze(1), H, w2.transpose(1, 2))
        return Y.view(E * mp, -1)[slot]

    def _loop(self, xf, tok, exp, lo, hi):
        Y = xf.new_zeros(tok.numel(), xf.shape[1])
        for e in range(lo, hi):
            m = exp == e - lo
            if m.any():
                Y[m] = self.experts[e](xf[tok[m]])
        return Y

    def forward(self, x):
        B, T, D = x.shape
        w, aux, k = self.router(x)
        xf = x.reshape(-1, D)
        wf = w.reshape(-1, self.n_expert)
        tok, exp = (wf > 0).nonzero(as_tuple=True)
        so = exp.argsort(stable=True)
        tok, exp = tok[so], exp[so]
        wts = wf[tok, exp].to(xf.dtype)
        y = torch.zeros_like(xf)
        nl = self.n_expert - self.n_heavy
        for lo, hi in ((0, nl), (nl, self.n_expert)):
            m = (exp >= lo) & (exp < hi)
            if m.any():
                g = self._grouped(xf, tok[m], exp[m] - lo, lo, hi)
                y.index_add_(0, tok[m], g * wts[m].unsqueeze(-1))
        return y.view(B, T, D), self.aux_coef * aux, k
