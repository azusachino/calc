# Positional embeddings: Learned, Sinusoidal, and RoPE

This exercise explores how sequence order is represented in language models, implemented in `calc/embeddings.py`.

## Background

Self-attention computes token interactions solely through pairwise dot products ($QK^T$). Without positional information, self-attention is permutation-equivariant: rearranging words produces the exact same representations, merely permuted. Positional information must be injected explicitly.

## Implementations in `calc`

### 1. Learned absolute position embeddings (`LearnedPositionalEmbedding`)

```python
from calc.embeddings import LearnedPositionalEmbedding

pos_emb = LearnedPositionalEmbedding(block_size=64, embedding_size=128)
positions = pos_emb(16)  # shape: [16, 128]
```

- **Mechanism**: A trainable lookup table (`nn.Embedding(block_size, embedding_size)`), matching the baseline `calc` model and GPT-1/GPT-2.
- **Tradeoff**: Simple and fully data-driven, but strictly limited to `block_size` tokens. Sequences beyond `block_size` cannot be processed without modifying the architecture or retraining.

### 2. Sinusoidal positional encodings (`SinusoidalPositionalEncoding`)

Vaswani et al. (2017) introduced deterministic sinusoidal waveforms across geometric frequencies:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / D}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / D}}\right)$$

```python
from calc.embeddings import SinusoidalPositionalEncoding

sin_emb = SinusoidalPositionalEncoding(embedding_size=128, max_length=64)
pe = sin_emb(16)  # shape: [16, 128]
```

- **Mechanism**: Low dimensions oscillate at high frequencies; high dimensions oscillate at low frequencies.
- **Tradeoff**: Requires no parameters and enables relative shift awareness through trigonometric addition formulas ($PE_{pos+k} = M_k PE_{pos}$).

### 3. Rotary Position Embedding (`RotaryPositionalEmbedding` / RoPE)

Su et al. (2021) designed RoPE to rotate Query and Key representations in 2D sub-planes:

$$\mathbf{q}_m = \mathbf{R}_{m\theta} \mathbf{q}, \quad \mathbf{k}_n = \mathbf{R}_{n\theta} \mathbf{k}$$

```python
import torch
from calc.embeddings import RotaryPositionalEmbedding

rope = RotaryPositionalEmbedding(head_size=32, max_length=64)
q = torch.randn(1, 4, 16, 32)  # [batch, heads, length, head_size]
rotated_q = rope(q)
```

- **Mechanism**: Applied to Query and Key tensors inside attention heads, rather than added to input token embeddings.
- **Property**: The inner product $\langle \mathbf{q}_m, \mathbf{k}_n \rangle$ depends strictly on the relative distance $(m - n)$, preserving decay over distance and enabling context extension.

---

## Runnable comparison script

Run the standalone comparison script:

```bash
uv run calc/compare_embeddings.py
```

This script inspects:
1. **Lookup & broadcasting**: how `token_embeddings [B, T, D]` and `position_embeddings [T, D]` fuse into `[B, T, D]`.
2. **Boundary properties**: verifies $\sin(0) = 0$ and $\cos(0) = 1$.
3. **Distance decay**: computes inner products between position 0 and position $t$ across relative distances.
