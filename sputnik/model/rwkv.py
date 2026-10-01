import torch
import torch.nn as nn


def wkv_parallel(w, u, k, v, chunk=32):
    # чанковый wkv: внутри чанка матрица затуханий e^{w(i-j)}, состояние (a, b) переносится дальше
    B, T, D = k.shape
    k = k.float().clamp(-50, 50)
    v = v.float()
    ek = torch.exp(k)
    kv = ek * v
    bonus = torch.exp(u.view(1, 1, D) + k)
    i = torch.arange(chunk, device=k.device, dtype=torch.float32)
    diff = (i.view(-1, 1) - i.view(1, -1)).masked_fill(i.view(-1, 1) < i.view(1, -1), 0.0)
    tril = (i.view(-1, 1) >= i.view(1, -1)).float()
    W = torch.exp(w.view(D, 1, 1) * diff.view(1, chunk, chunk)) * tril.view(1, chunk, chunk)
    dec = torch.exp(w.view(1, D) * (i + 1).view(chunk, 1))
    a = k.new_zeros(B, D)
    b = k.new_zeros(B, D)
    outs = []
    for s in range(0, T, chunk):
        Cc = min(chunk, T - s)
        kc, vc = k[:, s:s + Cc], v[:, s:s + Cc]
        num = torch.einsum('dij,bdj->bid', W[:, :Cc, :Cc], kv[:, s:s + Cc].transpose(1, 2))
        den = torch.einsum('dij,bdj->bid', W[:, :Cc, :Cc], ek[:, s:s + Cc].transpose(1, 2))
        num = num + dec[:Cc].unsqueeze(0) * a.unsqueeze(1)
        den = den + dec[:Cc].unsqueeze(0) * b.unsqueeze(1)
        bc = bonus[:, s:s + Cc]
        outs.append((num + bc * vc) / (den + bc))
        a, b = num[:, -1], den[:, -1]
    return torch.cat(outs, dim=1)


class RWKVAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mix_r = nn.Parameter(torch.zeros(dim))
        self.mix_k = nn.Parameter(torch.zeros(dim))
        self.mix_v = nn.Parameter(torch.zeros(dim))
        self.w_r = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        self.w_o = nn.Linear(dim, dim, bias=False)
        self.time_decay = nn.Parameter(torch.zeros(dim))
        self.time_first = nn.Parameter(torch.zeros(dim))

    def custom_init(self):
        with torch.no_grad():
            for m in (self.mix_r, self.mix_k, self.mix_v):
                m.fill_(0.5)
            self.time_decay.copy_(-5.0 * torch.arange(1, self.dim + 1) / self.dim)
            self.time_first.fill_(3.0)

    def forward(self, x):
        xx = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        r = torch.sigmoid(self.w_r(x * self.mix_r + xx * (1 - self.mix_r)))
        k = self.w_k(x * self.mix_k + xx * (1 - self.mix_k))
        v = self.w_v(x * self.mix_v + xx * (1 - self.mix_v))
        out = wkv_parallel(self.time_decay.float(), self.time_first.float(), k, v)
        return self.w_o(out.to(v.dtype) * r)

    def init_state(self, B, device):
        z = lambda *s: torch.zeros(B, *s, device=device)
        return (None, z(self.dim), z(self.dim))

    def step(self, x, state):
        prev, a, b = state
        # в параллельном проходе первый токен смешивается сам с собой
        xx = x if prev is None else prev.unsqueeze(1)
        r = torch.sigmoid(self.w_r(x * self.mix_r + xx * (1 - self.mix_r)))
        k = self.w_k(x * self.mix_k + xx * (1 - self.mix_k)).float().clamp(-50, 50).squeeze(1)
        v = self.w_v(x * self.mix_v + xx * (1 - self.mix_v)).float().squeeze(1)
        w, u = self.time_decay.float(), self.time_first.float()
        ew = torch.exp(w)
        ek = torch.exp(k)
        euk = torch.exp(u + k)
        out = (ew * a + ek * v + euk * v) / (ew * b + ek + euk)
        a = ew * a + ek * v
        b = ew * b + ek
        return self.w_o(out.unsqueeze(1).to(x.dtype) * r), (x.squeeze(1), a, b)
