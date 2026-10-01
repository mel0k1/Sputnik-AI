import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow.parquet as pq

from sputnik.tokenizer.bpe import BPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/home/z/my-project/download/datasets/russian-code/russian-code.parquet")
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--out", default="data/sft.jsonl")
    ap.add_argument("--max-tokens", type=int, default=255)
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    t = pq.read_table(args.parquet, columns=["problem", "solution"])
    problems = t.column("problem").to_pylist()
    solutions = t.column("solution").to_pylist()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    seen, kept, drop_long = set(), 0, 0
    with open(args.out, "w", encoding="utf-8") as f:
        for p, s in zip(problems, solutions):
            p, s = (p or "").strip(), (s or "").strip()
            if not p or not s:
                continue
            if p in seen:
                continue
            seen.add(p)
            pi = tok.encode(f"\nвопрос: {p}\nответ:", bos=True)
            ri = tok.encode(" " + s + "\n", eos=True)
            if len(pi) + len(ri) > args.max_tokens:
                drop_long += 1
                continue
            f.write(json.dumps({"p": pi, "r": ri}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"сохранено: {kept} | длинных срезано: {drop_long} -> {args.out}")


if __name__ == "__main__":
    main()
