# Online Softmax and tiled FlashAttention-2 recurrence

This document accompanies `calc/math/flash_attention.py` (paired with **Stanford CS336: Language Modeling from Scratch**) and proves how tiled SRAM attention computes exact scaled dot-product attention without materializing the $\mathcal{O}(N^2)$ attention matrix in HBM.

Run the companion benchmark:

```bash
uv run calc/math/flash_attention.py
```

---

## 1. The single-pass Online Softmax recurrence

Standard numerically stable softmax requires two passes over a row $s \in \mathbb{R}^N$: pass 1 finds $m = \max_j s_j$, and pass 2 accumulates $\ell = \sum_j e^{s_j - m}$.

**Online Softmax** fuses both into a single streaming pass across blocks $j = 1, 2, \dots, T_c$:

$$m^{(j)} = \max\!\left(m^{(j-1)}, \max_{k \in B_j} s_k\right)$$

$$\ell^{(j)} = e^{m^{(j-1)} - m^{(j)}} \ell^{(j-1)} + \sum_{k \in B_j} e^{s_k - m^{(j)}}$$

---

## 2. FlashAttention-2 unnormalized output accumulation

For query tile $Q_i \in \mathbb{R}^{B_r \times d}$ and key/value tile $K_j, V_j \in \mathbb{R}^{B_c \times d}$:

1. Compute tile scores $S_{ij} = Q_i K_j^\top / \sqrt{d}$ in on-chip SRAM.
2. Update running row maximum $m_i^{(j)} = \max(m_i^{(j-1)}, \operatorname{rowmax}(S_{ij}))$.
3. Rescale previous unnormalized output $O_i^{(j-1)}$ by $\alpha_i = e^{m_i^{(j-1)} - m_i^{(j)}}$ and add the new tile contribution $\tilde{P}_{ij} V_j = e^{S_{ij} - m_i^{(j)}} V_j$:
   $$O_i^{(j)} = \operatorname{diag}\!\left(e^{m_i^{(j-1)} - m_i^{(j)}}\right) O_i^{(j-1)} + \tilde{P}_{ij} V_j$$
4. Divide once by $\ell_i^{(T_c)}$ after the inner loop completes: $O_i = \operatorname{diag}(\ell_i^{(T_c)})^{-1} O_i^{(T_c)}$.

At $N = 4096, d = 128$ in BF16, eliminating the $N \times N$ intermediate reads/writes reduces HBM traffic by **33.0x**.
