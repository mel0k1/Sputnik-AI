from dataclasses import dataclass


@dataclass
class SputnikConfig:
    vocab_size: int = 16384
    dim: int = 512
    n_layer: int = 12
    seq_len: int = 512
    heads: int = 8
    attn_cycle: str = "lin,rwkv,mamba"
    ffn_mult: float = 3.0
    moe_every: int = 3
    n_expert: int = 128
    expert_hidden: int = 64
    n_heavy: int = 8
    heavy_mult: int = 3
    k_min: int = 1
    k_max: int = 4
    aux_coef: float = 0.01
    state_dim: int = 16
    conv_width: int = 4
    dt_min: float = 1e-3
    dt_max: float = 0.1
    dropout: float = 0.0
    tie_weights: bool = True

    def attn_kind(self, i):
        kinds = self.attn_cycle.split(",")
        return kinds[i % len(kinds)]

    def is_moe(self, i):
        return (i + 1) % self.moe_every == 0
