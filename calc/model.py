"""A compact decoder-only Transformer language model."""

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_size: int, num_heads: int, dropout: float):
        super().__init__()
        if embedding_size % num_heads:
            raise ValueError("embedding_size must be divisible by num_heads")
        self.attention = nn.MultiheadAttention(
            embedding_size, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1
        )
        result, _ = self.attention(x, x, x, attn_mask=mask, need_weights=False)
        return self.dropout(result)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = CausalSelfAttention(embedding_size, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.GELU(),
            nn.Linear(4 * embedding_size, embedding_size),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class LanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        embedding_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(block_size, embedding_size)
        self.blocks = nn.Sequential(
            *(TransformerBlock(embedding_size, num_heads, dropout) for _ in range(num_layers))
        )
        self.norm = nn.LayerNorm(embedding_size)
        self.output = nn.Linear(embedding_size, vocab_size)

    def forward(
        self, tokens: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, length = tokens.shape
        if length > self.block_size:
            raise ValueError(f"sequence length {length} exceeds block size {self.block_size}")
        positions = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        logits = self.output(self.norm(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            context = tokens[:, -self.block_size :]
            logits, _ = self(context)
            next_token = torch.multinomial(logits[:, -1].softmax(dim=-1), num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens
