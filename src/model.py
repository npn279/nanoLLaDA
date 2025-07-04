"""
Implementation of GPT2 like model.
Inspired from Kaparthy's tutorial https://github.com/karpathy/nanoGPT/tree/master
To convert a regular GPT2-like model to become a Language Diffusion models,
the causal mask from the self-attention mechanism needs to be removed.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Implementation of multiple head self attention layer.
    """

    def __init__(
        self,
        n_heads: int,
        dim_emb: int,
        max_seq_len: int,
        dropout: float = 0.0,
        flash: Optional[bool] = None,
    ) -> None:
        super().__init__()
        assert dim_emb % n_heads == 0

        # key, query, value projections for all heads, but in a batch
        self.att_weights = nn.Linear(dim_emb, 3 * dim_emb, bias=False)
        self.output_proj = nn.Linear(dim_emb, dim_emb)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.n_heads = n_heads
        self.dim_emb = dim_emb
        self.flash = flash
        if flash is None:
            self.flash = hasattr(F, "scaled_dot_product_attention")

        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size()[:-1]
        x = x.view(batch_size, seq_len, self.n_heads, self.dim_emb // self.n_heads)
        return x.permute(0, 2, 1, 3)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size()[:-2] + (self.dim_emb,))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.att_weights(x).split(self.dim_emb, dim=2)
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0,
                is_causal=False,
            )
        else:
            # manual implementation of attention
            att = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(k.size(-1)))
            att = self.softmax(att)
            att = self.attn_dropout(att)
            y = torch.matmul(att, v)

        y = self.merge_heads(y)
        # output projection
        y = self.resid_dropout(self.output_proj(y))
        return y


class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        out_dim: int,
        pointwise_mid_modules: list[nn.Module],
    ) -> None:
        super().__init__()
        self.first_layer = nn.Linear(in_dim, mid_dim, bias=False)
        self.mid = nn.ModuleList(pointwise_mid_modules)
        self.second_layer = nn.Linear(mid_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_layer(x)
        for layer in self.mid:
            x = layer(x)
        x = self.second_layer(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_heads: int,
        dim_emb: int,
        max_seq_len: int,
        dropout: float = 0.0,
        flash: Optional[bool] = None,
    ):
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            n_heads, dim_emb, max_seq_len, dropout, flash
        )
        self.norm1 = nn.RMSNorm(dim_emb)
        self.mlp = FeedForwardBlock(dim_emb, 4 * dim_emb, dim_emb, [nn.SiLU()])

        self.norm2 = nn.RMSNorm(dim_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        dim_emb: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.0,
        flash: Optional[bool] = None,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, dim_emb)
        self.pos_emb = nn.Embedding(block_size, dim_emb)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(n_heads, dim_emb, block_size, dropout, flash)
                for _ in range(n_layers)
            ]
        )
        self.emb_drop = nn.Dropout(dropout)
        self.lm_head = nn.Linear(dim_emb, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.shape[1] <= self.block_size
        ), f"""
            Cannot forward sequence of length {x.shape[0]},
            block size is only {self.block_size}
            """
        token_emb = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(x.shape[1], device=x.device))
        x = self.emb_drop(token_emb + pos_emb)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)
