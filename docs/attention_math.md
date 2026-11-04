# Dot product geometry and scaled attention variance

This document explores the mathematical justification for dot product similarity and the $1/\sqrt{d_k}$ scaling factor in modern transformer attention mechanisms.

Run the companion benchmark:

```bash
uv run calc/math/attention.py
```

---

## 1. The dot product as a similarity flashlight

For two vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$, the dot product reconciles an algebraic sum with a geometric projection:

$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^d u_i v_i = \|\mathbf{u}\| \|\mathbf{v}\| \cos \theta$$

- **Length of projection of $\mathbf{u}$ onto $\mathbf{v}$**: $\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) = \|\mathbf{u}\| \cos \theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|}$.
- If two vectors are parallel ($\theta = 0^\circ$), the dot product reaches its maximum positive value.
- If two vectors are perpendicular ($\theta = 90^\circ$), the dot product is $0$.
- If two vectors point in opposite directions ($\theta = 180^\circ$), the dot product is maximally negative.

Attention uses this property as a continuous "soft address lookup": a Query vector projects onto all Key vectors to score their mutual relevance.

---

## 2. Why divide by $\sqrt{d_k}$? (The variance explosion)

Assume query and key components are independent random variables with zero mean and unit variance:

$$q_i \sim \mathcal{N}(0, 1), \quad k_i \sim \mathcal{N}(0, 1)$$

The product of two independent standard normal variables has:

$$\mathbb{E}[q_i k_i] = \mathbb{E}[q_i] \mathbb{E}[k_i] = 0$$

$$\operatorname{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i k_i])^2 = \mathbb{E}[q_i^2] \mathbb{E}[k_i^2] = 1 \times 1 = 1$$

When computing the dot product across all $d_k$ dimensions:

$$s = \mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$$

Because the $d_k$ terms are independent, variances add:

$$\operatorname{Var}(s) = \sum_{i=1}^{d_k} \operatorname{Var}(q_i k_i) = d_k, \quad \sigma_s = \sqrt{d_k}$$

| Dimension ($d_k$) | Raw Variance $\operatorname{Var}(s)$ | Raw Standard Deviation $\sigma_s$ | Scaled Std $\sigma / \sqrt{d_k}$ |
| --- | --- | --- | --- |
| **16** | $16$ | $4.00$ | $1.00$ |
| **64** | $64$ | $8.00$ | $1.00$ |
| **128** | $128$ | $11.31$ | $1.00$ |
| **256** | $256$ | $16.00$ | $1.00$ |
| **768** | $768$ | $27.71$ | $1.00$ |

---

## 3. The Softmax saturation problem

When logits $s$ grow large (e.g. $\pm 25$), the Softmax function:

$$\operatorname{Softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

exponentiates the dominant value into $\approx 1.0$, pushing all others toward $0.0$.

The gradient of Softmax with respect to its inputs is:

$$\frac{\partial \operatorname{Softmax}_i}{\partial z_j} = \operatorname{Softmax}_i (\delta_{ij} - \operatorname{Softmax}_j)$$

When $\operatorname{Softmax}_i \approx 1.0$, this gradient evaluates to $1.0 \times (1.0 - 1.0) = 0.0$.
**Gradients vanish, and training stalls.**

### Empirical benchmark results (`calc/math/attention.py`)

Simulating $d_k = 64$ and context length $16$:
- **Raw Attention (No $1/\sqrt{d_k}$)**: Peak probability is **84.5%**, entropy collapses to **0.404** (saturated).
- **Scaled Attention (With $1/\sqrt{d_k}$)**: Peak probability is **25.2%**, entropy is **2.345** (smooth, uniform max is $2.773$).

Dividing by $\sqrt{d_k}$ preserves the unit variance of attention logits, ensuring healthy gradient flow across all tokens during training.
