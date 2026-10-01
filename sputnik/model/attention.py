import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttention(nn.Module):
    CHUNK = 64

    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.h = heads
        self.dh = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2) * self.dh ** -0.5
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        q = F.elu(q.float()) + 1
        k = F.elu(k.float()) + 1
        v = v.float()
        out = self._attend(q, k, v)
        return self.out(out.transpose(1, 2).reshape(B, T, D).to(x.dtype))

    def _attend(self, q, k, v):
        # состояние S и нормировщик z тянутся через чанки, внутри чанка — маска
        B, H, T, dh = q.shape
        S = q.new_zeros(B, H, dh, dh)
        z = q.new_zeros(B, H, dh)
        outs = []
        C = self.CHUNK
        for i in range(0, T, C):
            qc, kc, vc = q[:, :, i:i + C], k[:, :, i:i + C], v[:, :, i:i + C]
            sc = qc @ kc.transpose(-1, -2)
            mask = torch.tril(torch.ones(sc.shape[-2:], dtype=torch.bool, device=q.device))
            sc = sc.masked_fill(~mask, 0)
            num = sc @ vc + qc @ S
            den = sc.sum(-1, keepdim=True) + qc @ z.unsqueeze(-1)
            outs.append(num / (den + 1e-6))
            S = S + kc.transpose(-1, -2) @ vc
            z = z + kc.sum(-2)
        return torch.cat(outs, dim=2)

    def init_state(self, B, device):
        S = torch.zeros(B, self.h, self.dh, self.dh, device=device)
        z = torch.zeros(B, self.h, self.dh, device=device)
        return (S, z)

    def step(self, x, state):
        S, z = state
        B = x.shape[0]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, self.h, 1, self.dh) * self.dh ** -0.5
        k = k.view(B, self.h, 1, self.dh)
        v = v.view(B, self.h, 1, self.dh)
        q = F.elu(q.float()) + 1
        k = F.elu(k.float()) + 1
        v = v.float()
        sc = (q * k).sum(-1, keepdim=True)
        num = sc * v + q @ S
        den = sc + q @ z.unsqueeze(-1)
        out = num / (den + 1e-6)
        S = S + k.transpose(-1, -2) @ v
        z = z + k.sum(-2)
        return self.out(out.reshape(B, self.h * self.dh)).unsqueeze(1), (S, z)
