import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import SputnikBlock
from .mamba import MambaBlock
from .moe import MicroMoE
from .rwkv import RWKVAttention


class SputnikModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos = nn.Embedding(cfg.seq_len, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([SputnikBlock(cfg, i) for i in range(cfg.n_layer)])
        self.norm = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, RWKVAttention):
            m.custom_init()
        elif isinstance(m, MambaBlock):
            m.custom_init()
        elif isinstance(m, MicroMoE):
            m.custom_init()

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        x = self.drop(x)
        aux, ks = None, []
        for blk in self.blocks:
            x, a, k = blk(x)
            if a is not None:
                aux = a if aux is None else aux + a
                ks.append(k)
        x = self.norm(x)
        logits = self.head(x)
        loss, info = None, {"aux": 0.0, "k": 0.0}
        if aux is not None:
            info["aux"] = float(aux.detach())
            info["k"] = float(torch.stack(ks).mean())
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
            if aux is not None:
                loss = loss + aux
        return logits, loss, info
