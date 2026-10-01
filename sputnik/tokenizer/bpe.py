import heapq
import json
from collections import Counter, defaultdict

from .morph import BOW, MID, pretokenize

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"


def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


BYTE_ENC = bytes_to_unicode()
BYTE_DEC = {c: b for b, c in BYTE_ENC.items()}


class BPETokenizer:
    def __init__(self):
        self.base_vocab = {PAD: 0, BOS: 1, EOS: 2}
        for ch in BYTE_ENC.values():
            self.base_vocab[ch] = len(self.base_vocab)
        self.vocab = dict(self.base_vocab)
        self.ranks = {}
        self.merges = []
        self.id_sym = {i: s for s, i in self.vocab.items()}
        self.cache = {}
        self.bos_id = 1
        self.eos_id = 2

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _refresh(self):
        self.id_sym = {i: s for s, i in self.vocab.items()}
        self.cache = {}

    def train(self, text, vocab_size):
        freqs = Counter(pretokenize(text))
        words = {w: [BYTE_ENC[b] for b in w.encode("utf-8")] for w in freqs}
        self.vocab = dict(self.base_vocab)
        self.ranks = {}
        self.merges = []
        pairs = Counter()
        index = defaultdict(set)
        for w, syms in words.items():
            f = freqs[w]
            for pr in zip(syms, syms[1:]):
                pairs[pr] += f
                index[pr].add(w)
        heap = [(-c, a, b) for (a, b), c in pairs.items()]
        heapq.heapify(heap)
        while heap and len(self.merges) < vocab_size - len(self.base_vocab):
            neg, a, b = heapq.heappop(heap)
            c = pairs.get((a, b), 0)
            if c <= 0 or -neg != c:
                continue
            new = a + b
            for w in list(index.pop((a, b), ())):
                f = freqs[w]
                old = words[w]
                for pr in zip(old, old[1:]):
                    pairs[pr] -= f
                merged, j = [], 0
                while j < len(old):
                    if j + 1 < len(old) and old[j] == a and old[j + 1] == b:
                        merged.append(new)
                        j += 2
                    else:
                        merged.append(old[j])
                        j += 1
                words[w] = merged
                for pr in zip(merged, merged[1:]):
                    pairs[pr] += f
                    index[pr].add(w)
                    heapq.heappush(heap, (-pairs[pr], pr[0], pr[1]))
            pairs.pop((a, b), None)
            self.merges.append((a, b))
            self.vocab[new] = len(self.vocab)
            self.ranks[(a, b)] = len(self.ranks)
        self._refresh()

    def _merge_word(self, tok):
        syms = [BYTE_ENC[b] for b in tok.encode("utf-8")]
        while len(syms) > 1:
            best, br = None, 1 << 30
            for pr in zip(syms, syms[1:]):
                r = self.ranks.get(pr)
                if r is not None and r < br:
                    br, best = r, pr
            if best is None:
                break
            a, b = best
            out, j = [], 0
            while j < len(syms):
                if j + 1 < len(syms) and syms[j] == a and syms[j + 1] == b:
                    out.append(a + b)
                    j += 2
                else:
                    out.append(syms[j])
                    j += 1
            syms = out
        return [self.vocab[s] for s in syms]

    def encode(self, text, bos=False, eos=False):
        ids = [self.bos_id] if bos else []
        for tok in pretokenize(text):
            hit = self.cache.get(tok)
            if hit is None:
                hit = self._merge_word(tok)
                self.cache[tok] = hit
            ids.extend(hit)
        if eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        chars = "".join(self.id_sym[i] for i in ids if i >= 3)
        text = bytes(BYTE_DEC[c] for c in chars).decode("utf-8", errors="replace")
        return text.replace(BOW, " ").replace(MID, "")

    def save(self, path):
        data = {"version": 1, "merges": [list(m) for m in self.merges]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        t = cls()
        with open(path, encoding="utf-8") as f:
            merges = json.load(f)["merges"]
        for a, b in merges:
            t.vocab[a + b] = len(t.vocab)
            t.ranks[(a, b)] = len(t.ranks)
            t.merges.append((a, b))
        t._refresh()
        return t
