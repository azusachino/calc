"""Positional embedding implementations for language models.

Why positional encodings are necessary:
---------------------------------------
Standard self-attention computes query-key similarity solely via pairwise dot
products: Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V.
Because matrix multiplication has no notion of sequential order, attention is
permutation-equivariant: rearranging input tokens rearranges the output
identically without altering pairwise interactions. Order must be explicitly
injected into the representations.

Why simple addition `x = tok_emb + pos_emb` works:
--------------------------------------------------
In high dimensions (e.g. d=128 to 4096), space is vast and random vectors are
nearly orthogonal. By the distributive property of linear projections:
    W_q (e_i + p_i) = W_q e_i + W_q p_i
This allows downstream projections to separate semantic content from positional
coordinates. When computing the attention score between token i and token j:
    q_i^T k_j = (e_i + p_i)^T (e_j + p_j)
              = e_i^T e_j           (content-to-content: semantic similarity)
              + e_i^T p_j           (content-to-position: e.g. look for words at pos 0)
              + p_i^T e_j           (position-to-content: syntax-position bias)
              + p_i^T p_j           (position-to-position: pure spatial distance)

References:
-----------
- Vaswani et al. (2017): "Attention Is All You Need" (Sinusoidal encodings)
  https://arxiv.org/abs/1706.03762
- Su et al. (2021): "RoFormer: Enhanced Transformer with Rotary Position Embedding"
  https://arxiv.org/abs/2104.09864
- Ke et al. (ICLR 2021): "Rethinking Positional Encoding in Language Pre-training" (TUPE)
  https://arxiv.org/abs/2006.15595
- Press et al. (ICLR 2022): "Train Short, Test Long: Attention with Linear Biases (ALiBi)"
  https://arxiv.org/abs/2108.12409
"""

import torch
from torch import nn


class LearnedPositionalEmbedding(nn.Module):
    """Learned absolute positional embeddings via a lookup table (GPT-1/GPT-2 style).

    Each discrete position index t in [0, block_size - 1] is mapped to a trainable
    d-dimensional continuous vector.

    Properties:
    - Optimization: Entirely data-driven through backpropagation.
    - Limitation 1 (Discrete labels): Position 5 and position 6 are initialized as
      independent random Gaussian vectors. The network must spend optimization steps
      learning that adjacent positions are correlated.
    - Limitation 2 (No extrapolation): Strict context ceiling. Any input sequence
      length exceeding `block_size` triggers a runtime error.
    """

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

    Formulas:
        PE(pos, 2i)   = sin(pos / base^(2i / d))
        PE(pos, 2i+1) = cos(pos / base^(2i / d))

    where:
        pos: token position index in the sequence (0, 1, 2, ...)
        i: dimension index (0, 1, ..., d/2 - 1)
        base: frequency scale (typically 10000.0)

    Properties:
    - Wavelengths form a geometric progression from 2*pi to base*2*pi.
      Low dimensions oscillate rapidly (capturing local token order), while
      high dimensions oscillate slowly (capturing long-range global position).
    - Relative shift property: by the angle sum trigonometric identities:
          cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
          sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
      any offset PE(pos + k) can be represented as a linear transformation of
      PE(pos) via a 2x2 rotation matrix dependent only on k.
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
        # div_term = 1 / (base^(2i / d)) = exp(-2i/d * ln(base))
        div_term = torch.exp(dim_indices * (-torch.log(torch.tensor(base)) / embedding_size))

        pe = torch.zeros(max_length, embedding_size, dtype=torch.float32)
        angles = positions * div_term
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        return pe

    def forward(self, length: int, device: torch.device | None = None) -> torch.Tensor:
        """Return positional encodings of shape [length, embedding_size]."""
        if length > self.max_length:
            # Dynamically compute without reallocating if sequence exceeds cache
            pe = self._build_encoding(length, self.embedding_size, self.base)
            return pe.to(device=device)
        pe = self.pe[:length]
        if device is not None and pe.device != device:
            pe = pe.to(device=device)
        return pe


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE; Su et al., 2021).

    RoPE injects position by rotating Query and Key representations in 2D sub-planes
    such that their inner product depends strictly on relative displacement (m - n):

        < R_{m, theta} q, R_{n, theta} k > = q^T R_{n - m, theta} k

    Why RoPE is standard in modern LLMs (LLaMA, Mistral, Qwen, DeepSeek):
    1. Operates inside attention heads: avoids polluting or diluting the residual stream.
    2. Zero trained parameters: purely geometric transformation.
    3. Natural long-term decay: orthogonal rotations across geometric frequencies
       produce natural attention score decay as relative distance |m - n| grows.
    4. Flexible length extrapolation: context window can be extended via NTK-aware
       scaling, YaRN, or position interpolation without changing model weights.

    Implementation:
        For a 2D vector [x1, x2] at position m with angle theta:
            [ y1 ] = [ cos(m*theta)  -sin(m*theta) ] [ x1 ]
            [ y2 ]   [ sin(m*theta)   cos(m*theta) ] [ x2 ]
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
        angles = torch.outer(positions, inv_freq)  # shape: [max_length, head_size // 2]
        return angles.cos(), angles.sin()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary transformation to tensor x.

        Supports x shapes:
        - [batch, num_heads, length, head_size]
        - [batch, length, num_heads, head_size]
        - [batch, length, head_size]

        Assumes sequence length dimension is at index -2 and head_size is at index -1.
        """
        length = x.shape[-2]
        if length > self.max_length:
            cos, sin = self._build_cache(length, self.head_size, self.base)
            cos = cos.to(device=x.device, dtype=x.dtype)
            sin = sin.to(device=x.device, dtype=x.dtype)
        else:
            cos = self.cos[:length].to(device=x.device, dtype=x.dtype)
            sin = self.sin[:length].to(device=x.device, dtype=x.dtype)

        # Broadcast dimensions to match x: [1, ..., 1, length, head_size // 2]
        ndim = x.ndim
        view_shape = [1] * (ndim - 2) + [length, self.head_size // 2]
        cos = cos.view(view_shape)
        sin = sin.view(view_shape)

        # Pair adjacent components along the last dimension: [..., head_size // 2, 2]
        x_paired = x.view(*x.shape[:-1], self.head_size // 2, 2)
        x1 = x_paired[..., 0]
        x2 = x_paired[..., 1]

        # 2D orthogonal rotation:
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        return torch.stack((y1, y2), dim=-1).flatten(-2)
