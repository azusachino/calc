# Strided NDArray views and reverse-mode automatic differentiation

This document accompanies `calc/math/autograd.py` (paired with **CMU 10-414 / 10-714: Deep Learning Systems**) and demonstrates how modern tensor engines implement zero-copy memory views and reverse-mode automatic differentiation with broadcast reduction.

Run the companion benchmark:

```bash
uv run calc/math/autograd.py
```

---

## 1. Zero-copy strided tensor views (`shape`, `strides`, `offset`)

A $k$-dimensional tensor view over a flat 1D backing buffer `data` is completely specified by `(shape, strides, offset)`. Element $(i_0, \dots, i_{k-1})$ maps to flat storage index:

$$\text{index}(i_0, \dots, i_{k-1}) = \text{offset} + \sum_{d=0}^{k-1} i_d \cdot \text{strides}[d]$$

- **Axis permutation (`permute`)**: reorders `shape` and `strides` in $\mathcal{O}(1)$ metadata time without copying `data`.
- **Zero-copy broadcasting (`broadcast_to`)**: expands a singleton dimension ($s_d = 1 \to t_d = N$) by setting $\text{strides}[d] = 0$, so every index along axis $d$ reads the same memory slot.
- **Compaction (`compact`)**: materializes a non-contiguous view into row-major contiguous strides before calling hardware GEMM kernels.

---

## 2. Reverse-mode VJP adjoints and `sum_to_shape`

Given forward operation $H = X W + b$ where $X \in \mathbb{R}^{B \times D_{\text{in}}}$, $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$, and bias $b \in \mathbb{R}^{1 \times D_{\text{out}}}$ broadcasts across batch size $B$:

1. Matrix multiplication adjoints:
   $$\bar{X} = \bar{H} W^\top, \qquad \bar{W} = X^\top \bar{H}$$
2. Broadcast addition adjoint (`sum_to_shape`): because forward broadcasting replicates $b$ across all $B$ rows (a $1 \to B$ fan-out in the computation graph), the multivariable chain rule sums incoming gradients across all broadcasted axes:
   $$\bar{b} = \operatorname{sum\_to\_shape}(\bar{H}, (1, D_{\text{out}})) = \sum_{i=1}^B \bar{H}_{i, :}$$
