# Linear transformations and QKV projection geometry

This document explores how matrices function as space transformations and why modern self-attention requires three separate projection matrices ($W_Q, W_K, W_V$).

Run the companion benchmark:

```bash
uv run calc/linear_transformations.py
```

---

## 1. Matrices as geometric space transformations

In standard programming, a matrix is often treated as a 2D array of floats.
In geometry (Grant Sanderson / 3Blue1Brown view), **a matrix is a dynamic coordinate transformation**:

$$\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} a & b \\ c & d \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = x \begin{pmatrix} a \\ c \end{pmatrix} + y \begin{pmatrix} b \\ d \end{pmatrix}$$

- The first column $\begin{pmatrix} a \\ c \end{pmatrix}$ is the new landing spot of standard basis vector $\hat{i} = (1, 0)^T$.
- The second column $\begin{pmatrix} b \\ d \end{pmatrix}$ is the new landing spot of standard basis vector $\hat{j} = (0, 1)^T$.
- Matrix-vector multiplication scales and adds these transformed axes, shearing, stretching, or rotating the entire coordinate space while preserving straight, parallel grid lines.

---

## 2. Why attention requires $W_Q$ and $W_K$: breaking symmetry

If a model computed attention directly on raw token embeddings $X \in \mathbb{R}^{N \times d}$:

$$S_{\text{raw}} = X X^T$$

The resulting attention score matrix is **strictly symmetric**:

$$(S_{\text{raw}})_{i, j} = \mathbf{x}_i \cdot \mathbf{x}_j = \mathbf{x}_j \cdot \mathbf{x}_i = (S_{\text{raw}})_{j, i}$$

### The linguistic flaw of symmetric attention

In natural language syntax, grammatical dependencies are **directed**:
- The verb *"kicked"* must attend strongly to its direct object *"ball"*.
- But the word *"ball"* should not be forced to attend equally back to *"kicked"*; it might need to attend to its determiner *"the"* or modifier *"red"*.

By projecting $X$ through two independent learned matrices:

$$Q = X W_Q, \quad K = X W_K$$
$$S = Q K^T = X (W_Q W_K^T) X^T$$

Because $W_Q W_K^T$ is not symmetric in general, $S_{i, j} \neq S_{j, i}$.
**Independent $W_Q$ and $W_K$ break symmetry and enable rich, directional syntax.**

---

## 3. Decoupling roles: Query, Key, and Value

A single word must act in three distinct roles:
1. **Query ($Q$)**: The active search criteria ("What token can satisfy my role?").
2. **Key ($K$)**: The published index tag ("What properties do I offer?").
3. **Value ($V$)**: The substantive payload ("If selected, what semantic information do I transfer?").

`calc/linear_transformations.py` demonstrates that starting from a single 768D token embedding, three independent linear projections produce $q, k, v$ vectors with near-zero mutual correlation ($\cos \approx 0$). Each role operates in its own dedicated semantic sub-space.
