"""Small GPT for Level 2 (nanoGPT-style, written fresh; MIT nanoGPT is the reference design).

T4 constraints: fp16 autocast (no bf16), PyTorch SDPA attention (no FlashAttention-2). Weights tied.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    ctx: int = 512
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.ln2 = nn.LayerNorm(c.d_model)
        self.fc = nn.Linear(c.d_model, 4 * c.d_model, bias=False)
        self.out = nn.Linear(4 * c.d_model, c.d_model, bias=False)
        self.n_head, self.dropout = c.n_head, c.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.proj(y.transpose(1, 2).contiguous().view(B, T, C))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.c = c
        self.vocab_padded = math.ceil(c.vocab_size / 64) * 64
        self.wte = nn.Embedding(self.vocab_padded, c.d_model)
        self.wpe = nn.Embedding(c.ctx, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, self.vocab_padded, bias=False)
        self.head.weight = self.wte.weight  # tied
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("out.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def n_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - (self.wte.weight.numel() + self.wpe.weight.numel()) if non_embedding else n

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.shape[1]
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        return logits[..., : self.c.vocab_size]  # never predict padding rows
