# Positional embeddings: Learned, Sinusoidal, and RoPE

This document explores how sequence order is represented in language models, covering the mathematical intuition, implementation in `calc/core/embeddings.py`, and primary research papers.

---

## 1. The Core Intuition: Why Vector Addition Works

When looking at the standard injection line:

```python
x = self.token_embedding(tokens) + self.position_embedding(positions)
```

a natural question arises: **Why does adding a position vector to a word vector not scramble and corrupt the word's meaning?**

### The High-Dimensional Subspace Perspective

In 2D or 3D space, adding two vectors changes their directions completely. However, language models operate in high-dimensional vector spaces ($D = 128$ in `calc`, $768$ in GPT-2, $4096$ in LLaMA):

1. **High-dimensional space is overwhelmingly sparse**: Any two randomly initialized high-dimensional vectors are almost strictly orthogonal ($\mathbf{u} \cdot \mathbf{v} \approx 0$).
2. **Distributive linear projections**: Downstream attention heads project `x` via projection matrices $W_Q, W_K, W_V$. By linearity:
   $$W_Q (\mathbf{e}_{\text{word}} + \mathbf{p}_{\text{pos}}) = W_Q \mathbf{e}_{\text{word}} + W_Q \mathbf{p}_{\text{pos}}$$
   The network easily learns to project semantic features into one subspace and positional coordinates into another.

### The 4-Term Attention Expansion

When self-attention computes query-key similarity between token $i$ and token $j$, expand $(\mathbf{e}_i + \mathbf{p}_i)^T (\mathbf{e}_j + \mathbf{p}_j)$:

$$\mathbf{q}_i^T \mathbf{k}_j = \underbrace{\mathbf{e}_i^T \mathbf{e}_j}_{\text{1. Content-to-Content}} + \underbrace{\mathbf{e}_i^T \mathbf{p}_j}_{\text{2. Content-to-Position}} + \underbrace{\mathbf{p}_i^T \mathbf{e}_j}_{\text{3. Position-to-Content}} + \underbrace{\mathbf{p}_i^T \mathbf{p}_j}_{\text{4. Position-to-Position}}$$

This single addition gives attention four distinct perceptual abilities simultaneously:
- **Content-to-Content**: Semantics-to-semantics (e.g., does the verb "eat" attend to the noun "apple"?).
- **Content-to-Position**: Content querying a specific slot (e.g., pronouns attending to the sentence subject at position 0).
- **Position-to-Content**: Position querying syntax (e.g., does position 2 usually host an auxiliary verb?).
- **Position-to-Position**: Pure spatial displacement (e.g., attending to immediately preceding tokens).

---

## 2. Implementations in `calc`

### 1. Learned absolute position embeddings (`LearnedPositionalEmbedding`)

```python
from calc.core.embeddings import LearnedPositionalEmbedding

pos_emb = LearnedPositionalEmbedding(block_size=64, embedding_size=128)
positions = pos_emb(16)  # shape: [16, 128]
```

- **Mechanism**: A raw lookup table (`nn.Embedding(block_size, embedding_size)`), matching the baseline `calc` model and GPT-1/GPT-2.
- **Why it feels crude**: To an untrained model, position 5 and position 6 are independent Gaussian random vectors. The network must spend training steps learning that 5 and 6 are adjacent.
- **Hard context limit**: If `block_size = 64`, sequence length 65 causes an immediate runtime error.

### 2. Sinusoidal positional encodings (`SinusoidalPositionalEncoding`)

Vaswani et al. (2017) introduced deterministic sinusoidal waveforms across geometric frequencies:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / D}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / D}}\right)$$

```python
from calc.core.embeddings import SinusoidalPositionalEncoding

sin_emb = SinusoidalPositionalEncoding(embedding_size=128, max_length=64)
pe = sin_emb(16)  # shape: [16, 128]
```

- **Mechanism**: Low dimensions oscillate rapidly (capturing local token order); high dimensions oscillate slowly (capturing global sequence position).
- **Relative shift property**: By the trigonometric angle-sum identities, $PE_{pos+k} = M_k PE_{pos}$. Any relative offset $k$ is a linear rotation of $PE_{pos}$.

### 3. Rotary Position Embedding (`RotaryPositionalEmbedding` / RoPE)

Su et al. (2021) observed that instead of adding position vectors at layer 0 (where deep LayerNorm and MLP layers dilute them), we can rotate Query and Key vectors inside attention heads:

$$\mathbf{q}_m = \mathbf{R}_{m\theta} \mathbf{q}, \quad \mathbf{k}_n = \mathbf{R}_{n\theta} \mathbf{k}$$

```python
import torch
from calc.core.embeddings import RotaryPositionalEmbedding

rope = RotaryPositionalEmbedding(head_size=32, max_length=64)
q = torch.randn(1, 4, 16, 32)  # [batch, heads, length, head_size]
rotated_q = rope(q)
```

- **Mathematical beauty**:
  $$\langle \mathbf{q}_m, \mathbf{k}_n \rangle = (\mathbf{R}_{m\theta} \mathbf{q})^T (\mathbf{R}_{n\theta} \mathbf{k}) = \mathbf{q}^T \mathbf{R}_{(n - m)\theta} \mathbf{k}$$
  The inner product depends strictly on relative distance $(n - m)$, not absolute positions $m$ or $n$.
- **Modern industry standard**: Used by LLaMA, Mistral, Qwen, DeepSeek, and Gemma.

---

## 3. Runnable Verification

Run the comparison script:

```bash
uv run calc/exercises/compare_embeddings.py
```

Observe the relative distance inner-product decay:
```text
Distance (t)    RoPE Dot-Product     Sinusoidal Cos-Sim  
0               1.0000               1.0000              
1               0.9571               0.9662              
2               0.8581               0.8845              
4               0.7288               0.7479              
8               0.6576               0.7004              
16              0.4953               0.6054              
32              0.6016               0.6111              
```

---

## 4. Deep Dive & Primary References

To explore the mathematical theory and historical progression further:

1. **Foundational Architecture**:
   - [Vaswani et al. (2017) "Attention Is All You Need"](https://arxiv.org/abs/1706.03762) — Section 3.5 introduces sinusoidal positional encodings and the relative linear transformation conjecture.
2. **Rotary Position Embedding**:
   - [Su et al. (2021/2024) "RoFormer: Enhanced Transformer with Rotary Position Embedding"](https://arxiv.org/abs/2104.09864) — Full derivation in 2D complex coordinates and multi-dimensional block diagonals.
   - [Jianlin Su's Blog (Transformer升级之路：2、博采众长的旋转式位置编码)](https://spaces.ac.cn/archives/8265) — The original author's intuitive mathematical exposition that led to RoPE.
3. **Deconstructing Attention Cross-Terms**:
   - [Ke et al. (ICLR 2021) "Rethinking Positional Encoding in Language Pre-training" (TUPE)](https://arxiv.org/abs/2006.15595) — Proves why content-to-position and position-to-content terms should be disentangled.
4. **Alternative Modern Encodings**:
   - [Press et al. (ICLR 2022) "Train Short, Test Long: Attention with Linear Biases (ALiBi)"](https://arxiv.org/abs/2108.12409) — Demonstrates extrapolation by adding a static linear penalty directly to attention logits.
5. **Context Window Extension**:
   - [RoPE NTK-Aware Scaling](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/) and [YaRN (Peng et al., 2023)](https://arxiv.org/abs/2309.00071) — How rotating frequencies differently enables extending 4K context models to 128K+.
