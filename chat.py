import argparse
import sys

import torch

from sputnik.config import SputnikConfig
from sputnik.model.model import SputnikModel
from sputnik.tokenizer.bpe import BPETokenizer


@torch.no_grad()
def prefill(model, ids, states, pos0, seq_len):
    # реплика прогоняется по одному токену через кэш состояний, без перекодировки истории
    logits = None
    for j, i in enumerate(ids):
        p = min(pos0 + j, seq_len - 1)
        logits, states = model.step(torch.tensor([i]), torch.tensor([p]), states)
    return logits, states


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/sft.pt")
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ck = torch.load(args.ckpt, map_location="cpu")
    model = SputnikModel(SputnikConfig(**ck["cfg"]))
    model.load_state_dict(ck["model"])
    model.eval()
    tok = BPETokenizer.load(args.tokenizer)
    seq_len = ck["cfg"]["seq_len"]

    buf, states, first = [], model.init_state(1), True
    for line in sys.stdin:
        q = line.strip()
        if not q:
            continue
        seg = tok.encode(("\n" if buf else "") + "вопрос: " + q + "\nответ:", bos=first)
        first = False
        if len(buf) + len(seg) > seq_len - 8:
            # окно кончилось: кэш сбрасываем и перезаполняем с хвоста диалога
            buf = buf[-(seq_len // 2):]
            states = model.init_state(1)
            logits, states = prefill(model, buf + seg, states, 0, seq_len)
        else:
            logits, states = prefill(model, seg, states, len(buf), seq_len)
        buf.extend(seg)

        out = []
        for _ in range(args.max_tokens):
            logits = logits[0] / max(args.temp, 1e-2)
            if args.top_k > 0:
                v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                logits[logits < v[-1]] = -float("inf")
            nxt = torch.multinomial(torch.softmax(logits, -1), 1).item()
            if nxt == tok.eos_id:
                break
            out.append(nxt)
            buf.append(nxt)
            logits, states = model.step(torch.tensor([nxt]),
                                        torch.tensor([min(len(buf) - 1, seq_len - 1)]), states)
            if "вопрос:" in tok.decode(out):
                out.pop()
                buf.pop()
                break
        print("спутник:", tok.decode(out).strip() or "(тишина)", flush=True)


if __name__ == "__main__":
    main()
