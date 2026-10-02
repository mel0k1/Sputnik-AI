import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sputnik.config import SputnikConfig
from sputnik.model.model import SputnikModel
from sputnik.model.moe import MicroMoE


def old_forward(moe, x):
    # эталон: цикл по экспертам, как было до gather-маппинга
    B, T, D = x.shape
    w, aux, k = moe.router(x)
    xf = x.reshape(-1, D)
    wf = w.reshape(-1, moe.n_expert)
    y = torch.zeros_like(xf)
    for e, expert in enumerate(moe.experts):
        m = wf[:, e] > 0
        if m.any():
            y[m] += expert(xf[m]) * wf[m, e].unsqueeze(-1)
    return y.view(B, T, D), moe.aux_coef * aux, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=int, default=0, help="0 = эквивалентность, N = замер fwd+bwd на N итераций")
    args = ap.parse_args()

    torch.manual_seed(0)
    cfg = SputnikConfig(dim=160, n_layer=4, seq_len=256, n_expert=32,
                        n_heavy=8, expert_hidden=64, moe_every=2)
    moe = MicroMoE(cfg.dim, cfg.n_expert, cfg.expert_hidden,
                   cfg.n_heavy, cfg.heavy_mult, cfg.k_min, cfg.k_max, cfg.aux_coef)
    moe.eval()

    x = torch.randn(4, 64, cfg.dim)
    with torch.no_grad():
        y_new, aux_new, k_new = moe(x)
        y_old, aux_old, k_old = old_forward(moe, x)
    print(f"эквивалентность: max err {(y_new - y_old).abs().max().item():.2e} | "
          f"aux {(aux_new - aux_old).abs().item():.2e}")

    # градиенты
    moe.train()
    x1 = torch.randn(2, 32, cfg.dim, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    moe(x1)[0].pow(2).mean().backward()
    old_forward(moe, x2)[0].pow(2).mean().backward()
    g1 = torch.cat([p.grad.flatten() for p in moe.parameters() if p.grad is not None])
    g2 = torch.cat([p.grad.flatten() for p in moe.parameters() if p.grad is not None])
    print(f"градиенты: max err {(g1 - g2).abs().max().item():.2e}")

    if args.bench:
        moe.eval()
        x = torch.randn(16, cfg.seq_len, cfg.dim)
        for name, fn in (("gather", lambda: moe(x)), ("loop  ", lambda: old_forward(moe, x))):
            y = fn()
            loss = y[0].pow(2).mean()
            t0 = time.time()
            for _ in range(args.bench):
                moe.zero_grad(set_to_none=True)
                loss = fn()[0].pow(2).mean()
                loss.backward()
            dt = (time.time() - t0) / args.bench
            print(f"{name}: {dt * 1000:.1f} мс/итер (fwd+bwd, {cfg.n_expert} экспертов)")

        # bf16 на целом трансформере
        m = SputnikModel(cfg)
        m.eval()
        idx = torch.randint(3, 16000, (16, cfg.seq_len))
        tgt = torch.randint(3, 16000, (16, cfg.seq_len))
        for bf16 in (False, True):
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
                logits, loss, _ = m(idx, tgt)
                t0 = time.time()
                for _ in range(args.bench):
                    m.zero_grad(set_to_none=True)
                    logits, loss, _ = m(idx, tgt)
                    loss.backward()
                dt = (time.time() - t0) / args.bench
            print(f"модель {'bf16' if bf16 else 'fp32 '}: {dt * 1000:.0f} мс/итер "
                  f"({16 * cfg.seq_len / dt:.0f} tok/s), loss {loss.item():.3f}")


if __name__ == "__main__":
    main()
