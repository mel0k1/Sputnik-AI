import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sputnik.config import SputnikConfig
from sputnik.model.model import SputnikModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import eval_loss, get_batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/sputnik.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--iters", type=int, default=40)
    args = ap.parse_args()

    import numpy as np
    ck = torch.load(args.ckpt, map_location="cpu")
    model = SputnikModel(SputnikConfig(**ck["cfg"]))
    model.load_state_dict(ck["model"])
    val = np.memmap(os.path.join(args.data, "val.bin"), dtype=np.uint16, mode="r")
    vl = eval_loss(model, val, 16, ck["cfg"]["seq_len"], "cpu", iters=args.iters)
    print(f"ckpt step {ck['step']} | val loss {vl:.3f}")


if __name__ == "__main__":
    main()
