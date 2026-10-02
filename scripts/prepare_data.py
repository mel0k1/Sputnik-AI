import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sputnik.tokenizer.bpe import BPETokenizer

BLOCK = 1_000_000
VAL_CHARS = 2_000_000


def encode_stream(tok, text_iter, path):
    buf, n = "", 0
    with open(path, "wb") as f:
        for piece in text_iter:
            buf += piece
            n += len(piece)
            if n >= BLOCK:
                np.asarray(tok.encode(buf), dtype=np.uint16).tofile(f)
                buf, n = "", 0
        if buf:
            np.asarray(tok.encode(buf), dtype=np.uint16).tofile(f)


def read_lines(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        while True:
            line = f.readline()
            if not line:
                break
            yield line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--val", type=float, default=0.005)
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    os.makedirs(args.out, exist_ok=True)

    last = args.input[-1]
    tail, tail_len = [], 0
    limit = int(os.path.getsize(last) * args.val) + VAL_CHARS  # хвост последнего файла -> val
    with open(last, encoding="utf-8", errors="ignore") as f:
        f.seek(max(0, os.path.getsize(last) - limit * 2))
        f.readline()  # добиваем до конца строки
        tail = f.readlines()

    def train_iter():
        for path in args.input[:-1]:
            yield from read_lines(path)
        limit_chars = os.path.getsize(last) - sum(len(t) for t in tail)
        done = 0
        for line in read_lines(last):
            done += len(line)
            if done >= limit_chars:
                break
            yield line

    def val_iter():
        yield from tail

    encode_stream(tok, train_iter(), os.path.join(args.out, "train.bin"))
    encode_stream(tok, val_iter(), os.path.join(args.out, "val.bin"))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"vocab_size": tok.vocab_size}, f)
    tr = os.path.getsize(os.path.join(args.out, "train.bin")) // 2
    vl = os.path.getsize(os.path.join(args.out, "val.bin")) // 2
    print(f"train: {tr} токенов | val: {vl} | vocab: {tok.vocab_size}")


if __name__ == "__main__":
    main()
