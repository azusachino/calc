# Vector geometry and high-dimensional intuition

This document explores the fundamental geometric mental models underpinning token embeddings, vector addition, and attention in transformer models.

Run the companion benchmark:

```bash
uv run calc/vector_geometry.py
```

---

## 1. Vector addition: journeys vs. overwrites

In standard programming, variables overwrite: `x = x + pos`.
In geometry, vector addition is **journey composition** via the parallelogram law:

$$\mathbf{x} = \mathbf{e}_{\text{token}} + \mathbf{e}_{\text{pos}}$$

- $\mathbf{e}_{\text{token}}$ points to the semantic coordinates of the word in concept space.
- $\mathbf{e}_{\text{pos}}$ points to the location offset for that token's position index.
- $\mathbf{x}$ preserves geometric information about both journeys.

---

## 2. High-dimensional near-orthogonality

In 2D or 3D, adding an arbitrary offset easily bumps into neighboring points because space is crowded (only 2 or 3 mutually perpendicular directions exist).

In $d$-dimensional space with independent coordinates drawn from $\mathcal{N}(0, 1)$, the cosine similarity between any two random vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$ behaves as:

$$\mathbb{E}[\cos \theta] = 0, \quad \operatorname{Var}(\cos \theta) = \frac{1}{d}, \quad \sigma = \frac{1}{\sqrt{d}}$$

| Dimension ($d$) | Theoretical $\sigma = 1/\sqrt{d}$ | Empirical Mean $|\cos \theta|$ | Overlap Behavior |
| --- | --- | --- | --- |
| **2** | $0.7071$ | $\sim 0.64$ | Frequent collisions; directions cannot decouple |
| **8** | $0.3536$ | $\sim 0.28$ | Significant alignment jitter |
| **32** | $0.1768$ | $\sim 0.14$ | Moderate separation |
| **128** | $0.0884$ | $\sim 0.07$ | Noticeable orthogonality |
| **768** | **$0.0361$** | **$\sim 0.028$** | **Near-perfect orthogonality ($\approx 90^\circ \pm 2^\circ$)** |
| **4096** | **$0.0156$** | **$\sim 0.012$** | **Virtually orthogonal** |

### Why this matters for LLMs

Because $d = 768$ or $4096$ provides hundreds of nearly orthogonal directions, embedding spaces have distinct "hallways":
- Semantic features live in certain subspaces.
- Positional encodings live in others.
- Vector addition shifts the arrow along an independent coordinate axis without destroying the semantic identity of the word.

---

## 3. Subspace superposition benchmark

`calc/vector_geometry.py` simulates adding a positional vector to normalized token embeddings:

$$\text{sim}(\mathbf{e}_{\text{cat}}, \mathbf{e}_{\text{cat}} + \mathbf{p}_0) \approx 0.895$$
$$\text{sim}(\mathbf{e}_{\text{cat}}, \mathbf{e}_{\text{dog}} + \mathbf{p}_1) \approx 0.005$$

The semantic signal margin exceeds $+0.88$, showing that high-dimensional superposition retains clean ranking separation.
