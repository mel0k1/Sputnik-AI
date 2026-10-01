import argparse
import time

import torch
from dataclasses import asdict

from sputnik.config import SputnikConfig
from sputnik.model.model import SputnikModel
from sputnik.tokenizer.bpe import BPETokenizer


@torch.no_grad()
def sample(model, tok, ids, n, temp, top_k, top_p, device, seq_len, verbose=True):
    model.eval()
    ids = list(ids)
    states = model.init_state(1, device)
    t0, prefill = time.time(), len(ids)
    for t, i in enumerate(ids):
        pos = torch.tensor([min(t, seq_len - 1)], device=device)
        logits, states = model.step(torch.tensor([i], device=device), pos, states)
    logits = logits[0] / max(temp, 1e-2)
    if verbose:
        print(f"\r[{prefill} токенов префилла за {time.time() - t0:.1f}s]", end="", flush=True)
    for _ in range(n):
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = -float("inf")
        if top_p < 1.0:
            sp, si = torch.softmax(logits, -1).sort(descending=True)
            mask = (sp.cumsum(-1) - sp) >= top_p
            logits[si[mask]] = -float("inf")
        nxt = torch.multinomial(torch.softmax(logits, -1), 1).item()
        ids.append(nxt)
        t = len(ids) - 1
        pos = torch.tensor([min(t, seq_len - 1)], device=device)
        logits, states = model.step(torch.tensor([nxt], device=device), pos, states)
        logits = logits[0] / max(temp, 1e-2)
        if verbose:
            print("\r" + tok.decode(ids), end="", flush=True)
    if verbose:
        print()
    return tok.decode(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chat", action="store_true",
                    help="болталка по-простому: модель базовая, без SFT")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    ck = torch.load(args.ckpt, map_location=device)
    model = SputnikModel(SputnikConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    tok = BPETokenizer.load(args.tokenizer)
    seq_len = ck["cfg"]["seq_len"]

    if args.chat:
        hist = ""
        while True:
            q = input("ты: ")
            if not q:
                break
            hist += "\nвопрос: " + q + "\nответ:"
            ids = tok.encode(hist, bos=True)
            print("спутник: ", end="")
            hist += sample(model, tok, ids, args.tokens, args.temp,
                           args.top_k, args.top_p, device, seq_len)
    else:
        ids = tok.encode(args.prompt, bos=True)
        sample(model, tok, ids, args.tokens, args.temp,
               args.top_k, args.top_p, device, seq_len)


if __name__ == "__main__":
    main()
