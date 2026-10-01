import argparse
import json
import math
import os
import random
import time

import torch

from sputnik.config import SputnikConfig
from sputnik.model.model import SputnikModel


def make_batch(examples, bs, device):
    exs = random.sample(examples, bs)
    seqs = [e["p"] + e["r"] for e in exs]
    L = max(len(s) for s in seqs)
    x = torch.zeros(bs, L - 1, dtype=torch.long)
    y = torch.full((bs, L - 1), -100, dtype=torch.long)
    for i, (e, s) in enumerate(zip(exs, seqs)):
        lp = len(e["p"])
        x[i, :len(s) - 1] = torch.tensor(s[:-1])
        y[i, :len(s) - 1] = torch.tensor(s[1:])
        y[i, :lp - 1] = -100
    return x.to(device), y.to(device)


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    p = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft.jsonl")
    ap.add_argument("--base", default="out/sputnik.pt")
    ap.add_argument("--out", default="out/sft.pt")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--log", type=int, default=50)
    ap.add_argument("--save", type=int, default=150)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    examples = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    print(f"примеров: {len(examples)}")

    ck = torch.load(args.base, map_location="cpu")
    model = SputnikModel(SputnikConfig(**ck["cfg"]))
    model.load_state_dict(ck["model"])
    model.train()

    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))

    step0 = 0
    if args.resume and os.path.exists(args.out):
        ck2 = torch.load(args.out, map_location="cpu")
        model.load_state_dict(ck2["model"])
        opt.load_state_dict(ck2["optim"])
        step0 = ck2["step"] + 1
        print(f"продолжаем с шага {step0}")

    t0 = time.time()
    for step in range(step0, args.steps):
        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = make_batch(examples, args.batch, "cpu")
        logits, loss, info = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log == 0:
            rate = args.batch * (step - step0 + 1) / max(1e-6, time.time() - t0)
            print(f"step {step} | sft loss {loss.item():.3f} | k {info['k']:.2f} | {rate:.0f} прим/с")
        if (args.save and step and step % args.save == 0) or step == args.steps - 1:
            torch.save({"model": model.state_dict(), "optim": opt.state_dict(),
                        "cfg": ck["cfg"], "step": step}, args.out)
            print(f"step {step} | сохранено: {args.out}")


if __name__ == "__main__":
    main()
