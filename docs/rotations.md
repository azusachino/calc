# 2D Rotations and Rotary Position Embeddings (RoPE)

This document explores the geometry of 2D rotation matrices and how modern LLMs (LLaMA, Mistral, Gemma, Qwen) inject relative position information without adding learnable position vectors or altering token vector norms.

Run the companion benchmark:

```bash
uv run calc/rotations_rope.py
```

---

## 1. 2D rotation as an isometry

A 2D counterclockwise rotation by angle $\theta$ is represented by the matrix:

$$R_\theta = \begin{pmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{pmatrix}$$

### Geometric property: Isometry
Rotation matrices are orthogonal ($R_\theta^T R_\theta = I$). For any vector $\mathbf{v}$:

$$\|R_\theta \mathbf{v}\|^2 = (R_\theta \mathbf{v})^T (R_\theta \mathbf{v}) = \mathbf{v}^T R_\theta^T R_\theta \mathbf{v} = \mathbf{v}^T \mathbf{v} = \|\mathbf{v}\|^2$$

**Rotation preserves length perfectly.** Unlike learned absolute position embeddings ($x + p_m$), where adding $p_m$ perturbs the token's semantic norm and direction in an unconstrained way, rotating a token vector preserves its exact norm and energy.

---

## 2. The RoPE relative distance proof

Suppose token $A$ at position $m$ produces query vector $\mathbf{q}$, and token $B$ at position $n$ produces key vector $\mathbf{k}$.

In RoPE (Su et al., 2021), we rotate $\mathbf{q}$ by $m\theta$ and $\mathbf{k}$ by $n\theta$:

$$\tilde{\mathbf{q}}_m = R_{m\theta} \mathbf{q}, \quad \tilde{\mathbf{k}}_n = R_{n\theta} \mathbf{k}$$

Their attention inner product is:

$$\langle \tilde{\mathbf{q}}_m, \tilde{\mathbf{k}}_n \rangle = (R_{m\theta} \mathbf{q})^T (R_{n\theta} \mathbf{k}) = \mathbf{q}^T R_{m\theta}^T R_{n\theta} \mathbf{k}$$

Because $R_\theta$ is orthogonal, the transpose is simply the inverse rotation:

$$R_{m\theta}^T = R_{-m\theta}$$

Since rotating backwards by $m\theta$ and forward by $n\theta$ is equivalent to a single rotation by $(n - m)\theta$:

$$R_{-m\theta} R_{n\theta} = R_{(n - m)\theta}$$

Thus:

$$\langle \tilde{\mathbf{q}}_m, \tilde{\mathbf{k}}_n \rangle = \mathbf{q}^T R_{(n - m)\theta} \mathbf{k}$$

### The key insight
The inner product depends **only on the relative distance $(n - m)$**, not on the absolute positions $m$ or $n$.
Token positions enter attention purely as relative angular offsets.

---

## 3. High-dimensional RoPE: 2D clock pairs

In a $d$-dimensional head (e.g. $d = 64$ or $128$), RoPE does not attempt high-dimensional spherical rotations. Instead, it pairs adjacent dimensions into $d/2$ independent 2D planes:

$$\mathbf{x} = \begin{pmatrix} x_0 \\ x_1 \end{pmatrix}, \begin{pmatrix} x_2 \\ x_3 \end{pmatrix}, \dots, \begin{pmatrix} x_{d-2} \\ x_{d-1} \end{pmatrix}$$

Each plane $i$ rotates at an exponentially decreasing angular frequency:

$$\theta_i = 10000^{-2i/d}, \quad i \in \{0, 1, \dots, d/2 - 1\}$$

- **Lower planes (high frequencies, large $\theta_0 \approx 1$)**: Rotate rapidly with token index, capturing fine-grained local word order.
- **Higher planes (low frequencies, small $\theta \approx 0.0001$)**: Rotate very slowly, capturing macro sequence-level positioning.

When summed across all planes, the inner product exhibits natural long-distance decay, biasing attention towards nearby tokens while retaining relative distance across long contexts.
