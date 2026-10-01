import argparse
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sputnik.tokenizer.bpe import BPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--chars", type=int, default=400000)
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    for path in args.input:
        with open(path, "rb") as f:
            mid = f.seek(0, 2) // 2
            f.seek(max(0, mid - args.chars // 2))
            f.readline()
            sample = f.read(args.chars).decode("utf-8", errors="ignore")
        t0 = time.time()
        ids = tok.encode(sample)
        dt = time.time() - t0
        print(f"{os.path.basename(path):24s} {len(sample) / len(ids):5.2f} симв/токен "
              f"({len(ids) / dt / 1000:.0f}k ток/с)")


if __name__ == "__main__":
    main()
