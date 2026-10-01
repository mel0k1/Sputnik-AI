import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from sputnik.model.attention import LinearAttention
from sputnik.model.mamba import selective_scan
from sputnik.model.rwkv import wkv_parallel


def scan_ref(x, delta, A, Bp, Cp):
    B, T, E = x.shape
    N = A.shape[-1]
    h = torch.zeros(B, E, N)
    ys = []
    for t in range(T):
        h = torch.exp(delta[:, t].unsqueeze(-1) * A.view(1, E, N)) * h \
            + (delta[:, t] * x[:, t]).unsqueeze(-1) * Bp[:, t].unsqueeze(1)
        ys.append((Cp[:, t].unsqueeze(1) * h).sum(-1))
    return torch.stack(ys, dim=1)


def wkv_ref(w, u, k, v):
    B, T, D = k.shape
    ew = torch.exp(w).view(1, D)
    a = torch.zeros(B, D)
    b = torch.zeros(B, D)
    out = []
    for t in range(T):
        kt, vt = k[:, t], v[:, t]
        a = a * ew + torch.exp(kt) * vt
        b = b * ew + torch.exp(kt)
        bonus = torch.exp(u.view(1, D) + kt)
        out.append((a + bonus * vt) / (b + bonus))
    return torch.stack(out, dim=1)


def lin_ref(q, k, v):
    B, H, T, dh = q.shape
    S = torch.zeros(B, H, dh, dh)
    z = torch.zeros(B, H, dh)
    out = []
    for t in range(T):
        S = S + k[:, :, t].unsqueeze(-1) * v[:, :, t].unsqueeze(-2)
        z = z + k[:, :, t]
        num = torch.einsum("bhd,bhde->bhe", q[:, :, t], S)
        den = torch.einsum("bhd,bhd->bh", q[:, :, t], z)
        out.append(num / (den.unsqueeze(-1) + 1e-6))
    return torch.stack(out, dim=2)


def main():
    torch.manual_seed(0)

    B, T, E, N = 2, 96, 6, 4
    x = torch.randn(B, T, E)
    delta = torch.rand(B, T, E) * 0.1 + 1e-3
    A = -(torch.rand(E, N) * 2 + 0.5)
    Bp, Cp = torch.randn(B, T, N), torch.randn(B, T, N)
    d = (selective_scan(x, delta, A, Bp, Cp) - scan_ref(x, delta, A, Bp, Cp)).abs().max()
    print(f"mamba scan: max diff {d:.2e}")
    assert d < 1e-3

    D = 8
    w = -(torch.rand(D) * 4 + 0.1)
    u = torch.rand(D) * 3
    k, v = torch.randn(2, 50, D) * 2, torch.randn(2, 50, D)
    d = (wkv_parallel(w, u, k, v) - wkv_ref(w, u, k, v)).abs().max()
    print(f"rwkv wkv: max diff {d:.2e}")
    assert d < 1e-2

    attn = LinearAttention(16, 2)
    x = torch.randn(2, 32, 16)
    q, k, v = attn.qkv(x).chunk(3, dim=-1)
    q = q.view(2, 32, 2, 8).transpose(1, 2) * (8 ** -0.5)
    k = k.view(2, 32, 2, 8).transpose(1, 2)
    v = v.view(2, 32, 2, 8).transpose(1, 2)
    q, k, v = F.elu(q) + 1, F.elu(k) + 1, F.elu(v) + 1
    d = (attn._attend(q, k, v) - lin_ref(q, k, v)).abs().max()
    print(f"linear attention: max diff {d:.2e}")
    assert d < 1e-3

    print("все сканы сходятся с эталоном")


if __name__ == "__main__":
    main()
