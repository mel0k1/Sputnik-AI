import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow.parquet as pq

from sputnik.tokenizer.bpe import BPETokenizer


def pack(lines, budget, tok):
    # решение режем по строкам: строки не рвём, каждая пара влезает в бюджет
    chunks, cur, cur_len = [], [], 0
    for ln in lines:
        t = tok.encode((" " if not cur else "") + ln + "\n")
        if cur and cur_len + len(t) > budget:
            chunks.append(cur)
            cur, cur_len = [], 0
            t = tok.encode(" " + ln + "\n")
        if len(t) > budget:
            continue  # строка сама длиннее бюджета (минифицированный код)
        cur.append(t)
        cur_len += len(t)
    if cur:
        chunks.append(cur)
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/home/z/my-project/download/datasets/russian-code/russian-code.parquet")
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--out", default="data/sft.jsonl")
    ap.add_argument("--max-tokens", type=int, default=255)
    ap.add_argument("--split", action="store_true",
                    help="длинное решение -> несколько пар по строкам, а не только первая")
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    t = pq.read_table(args.parquet, columns=["problem", "solution"])
    problems = t.column("problem").to_pylist()
    solutions = t.column("solution").to_pylist()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    seen, pairs, extra, dropped = set(), 0, 0, 0
    with open(args.out, "w", encoding="utf-8") as f:
        for p, s in zip(problems, solutions):
            p, s = (p or "").strip(), (s or "").strip()
            if not p or not s or p in seen:
                continue
            seen.add(p)
            pi = tok.encode(f"\nвопрос: {p}\nответ:", bos=True)
            budget = args.max_tokens - len(pi) - 1  # под eos
            if budget < 8:
                dropped += 1
                continue
            chunks = pack(s.split("\n"), budget, tok)
            if not chunks:
                dropped += 1
                continue
            if not args.split:
                chunks = chunks[:1]
            extra += max(0, len(chunks) - 1)
            for ch in chunks:
                ri = [i for tl in ch for i in tl] + [tok.eos_id]
                f.write(json.dumps({"p": pi, "r": ri}, ensure_ascii=False) + "\n")
                pairs += 1
    print(f"пар: {pairs} (добор от нарезки длинных: {extra}) | скипнули: {dropped} -> {args.out}")


if __name__ == "__main__":
    main()
