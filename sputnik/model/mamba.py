import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def selective_scan(x, delta, A, Bp, Cp, chunk=16):
    # чанковый скан: внутри чанка всё через cumsum, состояние h переносится дальше
    B, T, E = x.shape
    N = A.shape[-1]
    dtype = x.dtype
    x, delta, Bp, Cp = x.float(), delta.float(), Bp.float(), Cp.float()
    l = delta.unsqueeze(-1) * A.view(1, 1, E, N)
    G = (x * delta).unsqueeze(-1) * Bp.unsqueeze(2)
    h = torch.zeros(B, E, N, device=x.device, dtype=torch.float32)
    ys = []
    for i in range(0, T, chunk):
        L = l[:, i:i + chunk].cumsum(1)
        P = torch.exp(L)
        H = (torch.exp(-L) * G[:, i:i + chunk]).cumsum(1)
        y = (torch.einsum("bten,btn->bte", P * H, Cp[:, i:i + chunk])
             + torch.einsum("bten,ben,btn->bte", P, h, Cp[:, i:i + chunk]))
        ys.append(y)
        h = P[:, -1] * (h + H[:, -1])
    return torch.cat(ys, dim=1).to(dtype)


class MambaBlock(nn.Module):
    def __init__(self, dim, state_dim=16, conv_width=4, dt_min=1e-3, dt_max=0.1, expand=2):
        super().__init__()
        self.d_inner = dim * expand
        self.state_dim = state_dim
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_rank = max(1, dim // 16)
        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, conv_width,
                              groups=self.d_inner, padding=conv_width - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * state_dim, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        self.A_log = nn.Parameter(torch.zeros(self.d_inner, state_dim))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def custom_init(self):
        with torch.no_grad():
            self.A_log.copy_(torch.log(torch.arange(
                1, self.state_dim + 1, dtype=torch.float32)).repeat(self.d_inner, 1))
            self.D.fill_(1.0)
            dt = self.dt_min + (self.dt_max - self.dt_min) * torch.rand(self.d_inner)
            self.dt_proj.bias.copy_(torch.log(torch.expm1(dt)))
        nn.init.normal_(self.dt_proj.weight, std=0.01)

    def forward(self, x):
        B, T, _ = x.shape
        xb, z = self.in_proj(x).chunk(2, dim=-1)
        xb = F.silu(self.conv(xb.transpose(1, 2))[..., :T].transpose(1, 2))
        dt, Bp, Cp = torch.split(self.x_proj(xb),
                                 [self.dt_rank, self.state_dim, self.state_dim], dim=-1)
        delta = F.softplus(self.dt_proj(dt)).clamp(max=self.dt_max)  # иначе exp переполняется
        A = -torch.exp(self.A_log.clamp(max=math.log(20.0)))
        y = selective_scan(xb, delta, A, Bp, Cp)
        y = y + xb * self.D
        return self.out_proj(y * F.silu(z))

    def init_state(self, B, device):
        c = torch.zeros(B, self.d_inner, self.conv.kernel_size[0] - 1, device=device)
        h = torch.zeros(B, self.d_inner, self.state_dim, device=device)
        return (c, h)

    def step(self, x, state):
        cbuf, h = state
        B = x.shape[0]
        xb_raw, z = self.in_proj(x).squeeze(1).chunk(2, dim=-1)
        w = self.conv.weight.squeeze(1)
        conv_out = (cbuf * w[..., :-1].unsqueeze(0)).sum(-1) + w[..., -1] * xb_raw + self.conv.bias
        cbuf = torch.cat([cbuf[:, :, 1:], xb_raw.unsqueeze(-1)], dim=-1)
        xb = F.silu(conv_out)
        dt, Bp, Cp = torch.split(self.x_proj(xb),
                                 [self.dt_rank, self.state_dim, self.state_dim], dim=-1)
        delta = F.softplus(self.dt_proj(dt)).clamp(max=self.dt_max)
        A = -torch.exp(self.A_log.clamp(max=math.log(20.0)))
        exl = torch.exp(delta.unsqueeze(-1) * A)
        h = exl * h + (delta * xb).unsqueeze(-1) * Bp.unsqueeze(1)
        y = (h * Cp.unsqueeze(1)).sum(-1) + xb * self.D
        return self.out_proj((y * F.silu(z)).unsqueeze(1)), (cbuf, h)
