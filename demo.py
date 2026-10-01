import torch

from sputnik.config import SputnikConfig
from sputnik.model.model import SputnikModel

cfg = SputnikConfig(vocab_size=512, dim=64, n_layer=3, heads=4, seq_len=64,
                    n_expert=16, n_heavy=2)
model = SputnikModel(cfg)
x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
logits, loss, info = model(x, torch.randint(0, cfg.vocab_size, (2, cfg.seq_len)))
print(f"параметров: {model.num_params() / 1e6:.2f}M")
print(f"logits: {tuple(logits.shape)} | loss: {loss.item():.3f} | средний k: {info['k']:.2f}")
print("блоки:", [cfg.attn_kind(i) for i in range(cfg.n_layer)])
