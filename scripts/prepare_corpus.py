import argparse
import glob
import hashlib
import os

import pyarrow.parquet as pq

MINC, MAXC = 200, 100_000


def rows(path):
    t = pq.read_table(path)
    cols = set(t.column_names)
    if "problem" in cols:
        for p, s in zip(t.column("problem").to_pylist(), t.column("solution").to_pylist()):
            yield f"{p}\n{s}"
    elif "content" in cols:
        for path_, c in zip(t.column("path").to_pylist(), t.column("content").to_pylist()):
            ext = os.path.splitext(path_ or "")[1].lstrip(".") or "txt"
            yield f"# {path_}\n{c}\n"
    elif "text" in cols:
        for x in t.column("text").to_pylist():
            yield x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="datasets")
    ap.add_argument("--out", default="corpus")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for parquet in sorted(glob.glob(os.path.join(args.in_dir, "*", "*.parquet"))):
        name = os.path.basename(os.path.dirname(parquet))
        seen, buf, chars = set(), [], 0
        for doc in rows(parquet):
            if not doc:
                continue
            n = len(doc)
            if n < MINC or n > MAXC:
                continue
            h = hashlib.sha1(doc.encode("utf-8", "ignore")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            buf.append(doc)
            chars += n
        out = os.path.join(args.out, f"{name}.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n\n".join(buf))
        print(f"{name}: {len(buf)} доков, {chars} символов -> {out}")


if __name__ == "__main__":
    main()
