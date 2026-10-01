import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sputnik.tokenizer.bpe import BPETokenizer


def encode_to(tok, text, path):
    with open(path, "wb") as f:
        step = 1_000_000
        for i in range(0, len(text), step):
            np.asarray(tok.encode(text[i:i + step]), dtype=np.uint16).tofile(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--val", type=float, default=0.005)
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    parts = []
    for path in args.input:
        with open(path, encoding="utf-8") as f:
            parts.append(f.read())
    text = "\n".join(parts)
    cut = int(len(text) * (1 - args.val))

    os.makedirs(args.out, exist_ok=True)
    encode_to(tok, text[:cut], os.path.join(args.out, "train.bin"))
    encode_to(tok, text[cut:], os.path.join(args.out, "val.bin"))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"vocab_size": tok.vocab_size}, f)
    print(f"train: {cut} символов | val: {len(text) - cut} | vocab: {tok.vocab_size}")


if __name__ == "__main__":
    main()
