import argparse
import json
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from sputnik.config import SputnikConfig
from sputnik.model.model import SputnikModel


def get_batch(data, bs, seq, device):
    ix = torch.randint(len(data) - seq - 1, (bs,))
    x = torch.stack([torch.from_numpy(data[i:i + seq].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + seq].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_loss(model, data, bs, seq, device, iters=20):
    model.eval()
    losses = [model(*get_batch(data, bs, seq, device))[1].item() for _ in range(iters)]
    model.train()
    return sum(losses) / len(losses)


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    p = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log", type=int, default=10)
    ap.add_argument("--eval", type=int, default=500)
    ap.add_argument("--save", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--dim", type=int, default=0, help="переопределения размера для мелких прогонов")
    ap.add_argument("--n-layer", type=int, default=0)
    ap.add_argument("--n-expert", type=int, default=0)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.data, "meta.json")))
    cfg = SputnikConfig(vocab_size=meta["vocab_size"], seq_len=args.seq)
    if args.dim:
        cfg.dim = args.dim
    if args.n_layer:
        cfg.n_layer = args.n_layer
    if args.n_expert:
        cfg.n_expert = args.n_expert
    model = SputnikModel(cfg).to(args.device)
    if args.compile:
        model = torch.compile(model)
    print(f"device: {args.device} | параметров: {model.num_params() / 1e6:.1f}M")

    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))

    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "sputnik.pt")
    step0 = 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        step0 = ck["step"] + 1
        print(f"продолжаем с шага {step0}")

    train_data = np.memmap(os.path.join(args.data, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(args.data, "val.bin"), dtype=np.uint16, mode="r")

    model.train()
    t0, toks = time.time(), 0
    for step in range(step0, args.steps):
        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(train_data, args.batch, args.seq, args.device)
        logits, loss, info = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        toks += x.numel()
        if step % args.log == 0:
            rate = toks / max(1e-6, time.time() - t0)
            print(f"step {step} | loss {loss.item():.3f} | k {info['k']:.2f} | "
                  f"lr {lr:.2e} | {rate:.0f} tok/s")
        if args.eval and step % args.eval == 0:
            vl = eval_loss(model, val_data, args.batch, args.seq, args.device)
            print(f"step {step} | val loss {vl:.3f}")
        if (args.save and step and step % args.save == 0) or step == args.steps - 1:
            torch.save({"model": model.state_dict(), "optim": opt.state_dict(),
                        "cfg": asdict(cfg), "step": step}, ckpt_path)
            print(f"step {step} | чекпоинт сохранён: {ckpt_path}")


if __name__ == "__main__":
    main()
