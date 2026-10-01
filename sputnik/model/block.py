import torch.nn as nn

from .attention import LinearAttention
from .mamba import MambaBlock
from .moe import MicroMoE
from .rwkv import RWKVAttention


class SputnikBlock(nn.Module):
    def __init__(self, cfg, i):
        super().__init__()
        kind = cfg.attn_kind(i)
        if kind == "lin":
            self.attn = LinearAttention(cfg.dim, cfg.heads)
        elif kind == "rwkv":
            self.attn = RWKVAttention(cfg.dim)
        elif kind == "mamba":
            self.attn = MambaBlock(cfg.dim, cfg.state_dim, cfg.conv_width, cfg.dt_min, cfg.dt_max)
        else:
            raise ValueError(kind)
        self.n1 = nn.LayerNorm(cfg.dim)
        self.n2 = nn.LayerNorm(cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.moe = cfg.is_moe(i)
        if self.moe:
            self.ffn = MicroMoE(cfg.dim, cfg.n_expert, cfg.expert_hidden,
                                cfg.n_heavy, cfg.heavy_mult, cfg.k_min, cfg.k_max, cfg.aux_coef)
        else:
            hidden = int(cfg.dim * cfg.ffn_mult)
            self.mlp = nn.Sequential(nn.Linear(cfg.dim, hidden), nn.GELU(),
                                     nn.Linear(hidden, cfg.dim))

    def forward(self, x):
        x = x + self.drop(self.attn(self.n1(x)))
        h = self.n2(x)
        if self.moe:
            h, aux, k = self.ffn(h)
        else:
            h, aux, k = self.mlp(h), None, None
        return x + self.drop(h), aux, k

    def init_state(self, B, device):
        return self.attn.init_state(B, device)

    def step(self, x, state):
        a, state = self.attn.step(self.n1(x), state)
        x = x + self.drop(a)
        h = self.n2(x)
        h = self.ffn(h)[0] if self.moe else self.mlp(h)
        return x + self.drop(h), state
