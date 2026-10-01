import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sputnik.config import SputnikConfig
from sputnik.model.model import SputnikModel


def main():
    torch.manual_seed(0)
    cfg = SputnikConfig(vocab_size=256, dim=64, n_layer=5, heads=4, seq_len=64,
                        n_expert=8, n_heavy=2, expert_hidden=16, moe_every=2)
    m = SputnikModel(cfg)
    m.eval()
    idx = torch.randint(3, 256, (2, 48))

    with torch.no_grad():
        logits_full = m(idx)[0]
        states = m.init_state(2)
        outs = []
        for t in range(idx.shape[1]):
            lo, states = m.step(idx[:, t], torch.tensor([t, t]), states)
            outs.append(lo)
        logits_step = torch.stack(outs, dim=1)

    err = (logits_full - logits_step).abs().max().item()
    print(f"max err: {err:.2e}")
    assert err < 2e-3, "step-режим разошёлся с полным форвардом"
    print("ok: кэш состояний совпадает с полным проходом")


if __name__ == "__main__":
    main()
