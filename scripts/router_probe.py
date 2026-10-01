import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sputnik.config import SputnikConfig
from sputnik.model.model import SputnikModel
from sputnik.tokenizer.bpe import BPETokenizer


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/sputnik.pt")
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--chars", type=int, default=20000)
    ap.add_argument("--windows", type=int, default=24)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    model = SputnikModel(SputnikConfig(**ck["cfg"]))
    model.load_state_dict(ck["model"])
    model.eval()
    tok = BPETokenizer.load(args.tokenizer)
    seq = ck["cfg"]["seq_len"]

    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            ids = np.asarray(tok.encode(f.read(args.chars)), dtype=np.int64)
        ks, losses = [], []
        for _ in range(args.windows):
            i = np.random.randint(0, len(ids) - seq - 1)
            x = torch.tensor(ids[i:i + seq])[None]
            y = torch.tensor(ids[i + 1:i + 1 + seq])[None]
            logits, loss, info = model(x, y)
            ks.append(info["k"])
            losses.append(loss.item())
        print(f"{os.path.basename(path):24s} k={np.mean(ks):.2f} loss={np.mean(losses):.2f}")


if __name__ == "__main__":
    main()
