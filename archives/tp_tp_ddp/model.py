#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn


class FeedForwardBlock(nn.Module):
    def __init__(
        self, in_dim: int, mid_dim: int, out_dim: int, pointwise_mid_modules: list[nn.Module],
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


class ResNorm(nn.Module):
    def __init__(self, submodule: nn.Module, hidden_dim: int, dropout_prob: float) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.sub = submodule
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        y = self.layernorm(x)
        y = self.sub(y, *args, **kwargs)
        y = self.dropout(y)
        return x + y


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout_prob: float) -> None:
        super().__init__()
        
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.softmax = nn.Softmax(dim=-1)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size()[:-2] + (self.hidden_dim,))
        return x

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        q = self.separate_heads(self.q(x))
        k = self.separate_heads(self.k(x))
        v = self.separate_heads(self.v(x))
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.hidden_dim) + attention_mask
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)
        y = torch.matmul(attention_probs, v)
        y = self.merge_heads(y)

        output = self.output(y)
        return output


class Block(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, num_heads: int, dropout_prob: float) -> None:
        super().__init__()
        self.attention = ResNorm(MultiHeadSelfAttention(hidden_dim, num_heads, dropout_prob), hidden_dim, dropout_prob)
        self.ff = ResNorm(FeedForwardBlock(hidden_dim, intermediate_dim, hidden_dim, [nn.GELU()]), hidden_dim, dropout_prob)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.attention(x, attention_mask=attention_mask)
        x = self.ff(x)
        return x


class Classifier(nn.Module):
    def __init__(self, hidden_dim: int, dropout_prob: float) -> None:
        super().__init__()
        self.network = FeedForwardBlock(
            in_dim=hidden_dim,
            mid_dim=hidden_dim,
            out_dim=2,
            pointwise_mid_modules=[nn.GELU(), nn.Dropout(dropout_prob)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 0, :] # only take the output of the [CLS] token
        x = self.network(x)
        return x


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.mod = nn.Embedding(vocab_size, hidden_dim)

    def forward(self, x):
        x = self.mod(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self, vocab_size: int,
        num_layers: int,
        hidden_dim: int,
        intermediate_dim: int,
        num_heads: int,
        dropout_prob: float,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embedding = Embedding(vocab_size, hidden_dim)
        self.transformer_blocks = nn.ModuleList([
            Block(hidden_dim, intermediate_dim, num_heads, dropout_prob)
            for _ in range(num_layers)
        ])
        self.classifier = Classifier(hidden_dim, dropout_prob)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        attention_mask = attention_mask[:, None, None, :].to(x.dtype)
        attention_mask = (1.0 - attention_mask) * torch.finfo(x.dtype).min
        for block in self.transformer_blocks:
            x = block(x, attention_mask=attention_mask)
        x = self.classifier(x)
        return x
