# Spectral graph Laplacians, Personalized PageRank, and GNNs

This document accompanies `calc/math/graphs.py` (paired with **Stanford CS224W: Machine Learning with Graphs**) and explores the Graph Laplacian Dirichlet energy, Personalized PageRank (`PPR`), and over-smoothing in Message-Passing Graph Neural Networks.

Run the companion benchmark:

```bash
uv run calc/math/graphs.py
```

---

## 1. The Graph Laplacian and Dirichlet smoothness energy

For an undirected graph with adjacency matrix $A$ and degree matrix $D = \operatorname{diag}(\sum_j A_{ij})$, the unnormalized Graph Laplacian is $L = D - A$. For any node signal $x \in \mathbb{R}^n$, its quadratic form measures total squared variation across edges:

$$x^\top L x = \frac{1}{2} \sum_{i, j} A_{ij} (x_i - x_j)^2 \ge 0$$

- The smallest eigenvalue is $\lambda_1 = 0$ with constant eigenvector $\mathbf{1}$.
- The second-smallest eigenvalue $\lambda_2$ (**algebraic connectivity / Fiedler value**) is strictly positive if and only if the graph is connected.

---

## 2. Personalized PageRank (`PPR`) for structural code ranking

Given seed nodes $S \subseteq V$ (for example, files or symbols currently in context), `PPR` teleports back to the seed distribution $p_S$ with probability $\alpha$:

$$r^{(t+1)} = (1 - \alpha) M r^{(t)} + \alpha p_S, \qquad M = A^\top D^{-1}$$

---

## 3. Kipf-Welling GCN propagation and over-smoothing

Each GCN layer applies degree-normalized neighborhood averaging $\tilde{D}^{-1/2}(A + I)\tilde{D}^{-1/2}$, which acts as a low-pass spectral graph filter. Stacking too many message-passing layers without residual connections attenuates high-frequency eigenvalues and drives the normalized Dirichlet energy toward zero (**over-smoothing**).
