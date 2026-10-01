import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sputnik.tokenizer.bpe import BPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=0, help="символов на файл, 0 = всё")
    ap.add_argument("--out", default="tokenizer.json")
    args = ap.parse_args()

    parts = []
    for path in args.input:
        with open(path, encoding="utf-8") as f:
            parts.append(f.read(args.limit) if args.limit else f.read())
    text = "\n".join(parts)

    tok = BPETokenizer()
    tok.train(text, args.vocab)
    tok.save(args.out)

    sample = text[len(text) // 2: len(text) // 2 + 100000]
    ids = tok.encode(sample)
    print(f"vocab: {tok.vocab_size}")
    print(f"sample: {len(sample)} символов -> {len(ids)} токенов "
          f"({len(sample) / max(1, len(ids)):.1f} симв/токен)")


if __name__ == "__main__":
    main()
