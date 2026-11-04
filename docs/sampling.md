# Softmax, temperature scaling, and nucleus sampling

This document explores the bridge between continuous unnormalized logits and discrete next-token probability distributions in autoregressive generation.

Run the companion benchmark:

```bash
uv run calc/math/sampling.py
```

---

## 1. Softmax and numerical stability

An autoregressive language model ends with a linear unembedding layer that produces raw unconstrained real values called **logits** $\mathbf{z} \in \mathbb{R}^V$ for a vocabulary of size $V$.

To convert logits into a valid probability distribution $P$ where $\sum_{i=1}^V p_i = 1$ and $p_i \ge 0$, we apply the **Softmax function** (Boltzmann distribution from statistical mechanics):

$$p_i = \text{Softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^V e^{z_j}}$$

### The numerical overflow problem
In 32-bit floating-point arithmetic, $\exp(89.0) \approx 10^{38}$ approaches the maximum representable float (`float32` max $\approx 3.4 \times 10^{38}$). If a logit reaches $100.0$, $\exp(100.0)$ produces `inf`, and dividing by `inf` yields `NaN`.

### The exact shift trick
Notice that shifting all logits by a constant $C$ cancels out mathematically:

$$\frac{e^{z_i - C}}{\sum_j e^{z_j - C}} = \frac{e^{z_i} e^{-C}}{\sum_j e^{z_j} e^{-C}} = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

By setting $C = \max_k(z_k)$, the largest exponent is $e^0 = 1.0$, completely preventing numerical overflow without altering probabilities.

---

## 2. Temperature scaling ($T$) and Shannon entropy

By dividing logits by a temperature parameter $T > 0$ before Softmax:

$$p_i(T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

### Physical and geometric effects
- **$T \to 0$ (Cold / Greedy)**: Sharpens differences exponentially. The largest logit dominates with probability $\approx 1.0$. The distribution becomes deterministic ($\text{argmax}$).
- **$T = 1.0$ (Calibrated)**: The native probability distribution learned during cross-entropy pretraining.
- **$T \gg 1.0$ (Hot / Chaotic)**: Diminishes logit differences ($z_i / T \to 0$, so $\exp(0) \to 1$). The distribution flattens toward a uniform distribution across all $V$ tokens, inducing high-entropy hallucinations.

### Shannon entropy as confidence metric
The uncertainty of the next token distribution is measured in bits by **Shannon Entropy**:

$$H(P) = -\sum_{i=1}^V p_i \log_2(p_i)$$

- If the model is 100% confident in a single token: $H(P) = 0$ bits.
- If the model is completely uncertain across $V$ tokens: $H(P) = \log_2(V)$ bits.

---

## 3. Truncation sampling: Top-k vs. Top-p (Nucleus)

Unconstrained sampling from the raw Softmax distribution risks selecting tokens from the long tail of low-probability words, which often contain syntactically invalid or incoherent choices.

### Top-$k$ filtering (Fan et al., 2018)
- Sorts tokens by probability and keeps only the top $k$ candidates, zeroing out the rest.
- **Limitation**: Fixed $k$ is rigid. If the distribution is peaked ($p_{\text{top}} = 0.95$), keeping $k=50$ includes noisy tail tokens. If the distribution is flat, $k=50$ might cut off viable synonyms.

### Top-$p$ / Nucleus filtering (Holtzman et al., 2019)
- Dynamically selects the smallest set of tokens whose cumulative probability exceeds threshold $p \in (0, 1]$:

$$\sum_{i \in V^{(p)}} p_i \ge p$$

- When the model is confident, the nucleus $V^{(p)}$ shrinks to 1 or 2 tokens.
- When the model is uncertain, the nucleus expands automatically to accommodate all plausible candidates.
- The remaining tokens are renormalized so probabilities sum to 1.0.
