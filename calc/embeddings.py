"""Positional embedding implementations for language models.

Includes:
- Learned positional embeddings (GPT-2 style lookup)
- Sinusoidal positional encodings (Vaswani et al., 2017)
- Rotary positional embeddings (RoPE; Su et al., 2021)
"""

import torch
from torch import nn


class LearnedPositionalEmbedding(nn.Module):
    """Learned absolute positional embeddings via a lookup table (GPT-1/GPT-2 style)."""

    def __init__(self, block_size: int, embedding_size: int):
        super().__init__()
        self.block_size = block_size
        self.embedding_size = embedding_size
        self.weight = nn.Embedding(block_size, embedding_size)

    def forward(self, length: int, device: torch.device | None = None) -> torch.Tensor:
        """Return positional embeddings of shape [length, embedding_size]."""
        if length > self.block_size:
            raise ValueError(f"sequence length {length} exceeds maximum block size {self.block_size}")
        positions = torch.arange(length, device=device)
        return self.weight(positions)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding as introduced in 'Attention Is All You Need' (Vaswani et al., 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i / embedding_size))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / embedding_size))
    """

    def __init__(self, embedding_size: int, max_length: int = 2048, base: float = 10000.0):
        super().__init__()
        if embedding_size % 2 != 0:
            raise ValueError("embedding_size must be even for sinusoidal encoding")
        self.embedding_size = embedding_size
        self.max_length = max_length
        self.base = base

        pe = self._build_encoding(max_length, embedding_size, base)
        self.register_buffer("pe", pe, persistent=False)

    @staticmethod
    def _build_encoding(max_length: int, embedding_size: int, base: float) -> torch.Tensor:
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        # 2i for i in [0, embedding_size // 2 - 1]
        dim_indices = torch.arange(0, embedding_size, 2, dtype=torch.float32)
        div_term = torch.exp(dim_indices * (-torch.log(torch.tensor(base)) / embedding_size))

        pe = torch.zeros(max_length, embedding_size, dtype=torch.float32)
        angles = positions * div_term
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        return pe

    def forward(self, length: int, device: torch.device | None = None) -> torch.Tensor:
        """Return positional encodings of shape [length, embedding_size]."""
        if length > self.max_length:
            # Dynamically generate if requested length exceeds cached buffer
            pe = self._build_encoding(length, self.embedding_size, self.base)
            return pe.to(device=device)
        pe = self.pe[:length]
        if device is not None and pe.device != device:
            pe = pe.to(device=device)
        return pe


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE; Su et al., 2021).

    Applies rotation to Query and Key representations so that their inner product
    depends solely on relative token displacement (m - n).
    """

    def __init__(self, head_size: int, max_length: int = 2048, base: float = 10000.0):
        super().__init__()
        if head_size % 2 != 0:
            raise ValueError("head_size must be even for RoPE")
        self.head_size = head_size
        self.max_length = max_length
        self.base = base

        cos, sin = self._build_cache(max_length, head_size, base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _build_cache(max_length: int, head_size: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
        # theta_i = base^(-2i / head_size) for i in [0, head_size // 2 - 1]
        dim_indices = torch.arange(0, head_size, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (dim_indices / head_size))
        positions = torch.arange(max_length, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)  # [max_length, head_size // 2]
        return angles.cos(), angles.sin()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary transformation to tensor x.

        Supports x shapes:
        - [batch, num_heads, length, head_size]
        - [batch, length, num_heads, head_size]
        - [batch, length, head_size]

        Assumes the sequence length dimension is at index -2 and head_size is at index -1.
        """
        length = x.shape[-2]
        if length > self.max_length:
            cos, sin = self._build_cache(length, self.head_size, self.base)
            cos = cos.to(device=x.device, dtype=x.dtype)
            sin = sin.to(device=x.device, dtype=x.dtype)
        else:
            cos = self.cos[:length].to(device=x.device, dtype=x.dtype)
            sin = self.sin[:length].to(device=x.device, dtype=x.dtype)

        # Match dimensions of x: [1, ..., 1, length, head_size // 2]
        ndim = x.ndim
        view_shape = [1] * (ndim - 2) + [length, self.head_size // 2]
        cos = cos.view(view_shape)
        sin = sin.view(view_shape)

        # Pair adjacent components along the last dimension
        x_paired = x.view(*x.shape[:-1], self.head_size // 2, 2)
        x1 = x_paired[..., 0]
        x2 = x_paired[..., 1]

        # 2D rotation matrix:
        # [ cos  -sin ] [ x1 ] = [ x1*cos - x2*sin ]
        # [ sin   cos ] [ x2 ]   [ x1*sin + x2*cos ]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        return torch.stack((y1, y2), dim=-1).flatten(-2)
