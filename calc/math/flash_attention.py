"""Online Softmax and tiled FlashAttention-2 forward recurrence.

Companion lab for Stanford CS336 (Language Modeling from Scratch):
1. Single-pass Online Softmax using running maximum m^(j) and normalizer l^(j).
2. Tiled FlashAttention-2 forward pass in SRAM blocks without materializing the N x N matrix.
3. Exact 6ND training FLOPs and HBM traffic comparison.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def online_softmax_1d(x: torch.Tensor, block_size: int = 4) -> tuple[torch.Tensor, float, float]:
    """Compute exact softmax in blocks using the Online Softmax rescaling recurrence.

    Returns:
        (probabilities, final_max, final_normalizer)
    """
    if x.ndim != 1:
        raise ValueError("online_softmax_1d expects a 1D tensor")

    m_prev = -float("inf")
    l_prev = 0.0
    n = x.numel()

    for start in range(0, n, block_size):
        block = x[start : min(start + block_size, n)]
        m_block = float(torch.max(block).item())
        m_new = max(m_prev, m_block)
        l_new = math.exp(m_prev - m_new) * l_prev + float(torch.sum(torch.exp(block - m_new)).item())
        m_prev, l_prev = m_new, l_new

    probs = torch.exp(x - m_prev) / l_prev
    return probs, m_prev, l_prev


def tiled_flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_q: int = 4,
    block_kv: int = 4,
    causal: bool = False,
) -> torch.Tensor:
    """Compute exact scaled dot-product attention via FlashAttention-2 SRAM tiling.

    Args:
        q: Query tensor of shape (N, d)
        k: Key tensor of shape (N, d)
        v: Value tensor of shape (N, d)
        block_q: Query tile size B_r
        block_kv: Key/Value tile size B_c
        causal: Whether to apply a causal mask (col > row set to -inf)

    Returns:
        Output tensor O of shape (N, d) matching standard softmax(Q K^T / sqrt(d)) V.
    """
    if q.ndim != 2 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k, v must be 2D tensors of identical shape (N, d)")

    n, d = q.shape
    scale = 1.0 / math.sqrt(d)
    out = torch.zeros_like(q)

    for q_start in range(0, n, block_q):
        q_end = min(q_start + block_q, n)
        q_tile = q[q_start:q_end]  # (B_r, d)
        b_r = q_end - q_start

        m_i = torch.full((b_r,), -float("inf"), dtype=q.dtype, device=q.device)
        l_i = torch.zeros((b_r,), dtype=q.dtype, device=q.device)
        o_i = torch.zeros((b_r, d), dtype=q.dtype, device=q.device)

        for kv_start in range(0, n, block_kv):
            kv_end = min(kv_start + block_kv, n)
            if causal and kv_start >= q_end:
                break

            k_tile = k[kv_start:kv_end]  # (B_c, d)
            v_tile = v[kv_start:kv_end]  # (B_c, d)

            s_ij = (q_tile @ k_tile.transpose(0, 1)) * scale  # (B_r, B_c)
            if causal:
                q_idx = torch.arange(q_start, q_end, device=q.device).unsqueeze(1)
                k_idx = torch.arange(kv_start, kv_end, device=q.device).unsqueeze(0)
                s_ij = torch.where(k_idx <= q_idx, s_ij, torch.tensor(-float("inf"), dtype=q.dtype))

            row_max = torch.max(s_ij, dim=1).values
            m_new = torch.maximum(m_i, row_max)

            alpha = torch.exp(m_i - m_new)
            p_tilde = torch.exp(s_ij - m_new.unsqueeze(1))

            l_i = alpha * l_i + torch.sum(p_tilde, dim=1)
            o_i = alpha.unsqueeze(1) * o_i + (p_tilde @ v_tile)
            m_i = m_new

        out[q_start:q_end] = o_i / l_i.unsqueeze(1)

    return out


def standard_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Materialized O(N^2) scaled dot-product attention reference."""
    n, d = q.shape
    scores = (q @ k.transpose(0, 1)) / math.sqrt(d)
    if causal:
        mask = torch.triu(torch.ones((n, n), dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask, -float("inf"))
    probs = F.softmax(scores, dim=-1)
    return probs @ v


def attention_hbm_bytes_comparison(
    seq_len: int,
    d_head: int,
    bytes_per_elem: int = 2,
) -> tuple[int, int, float]:
    """Compare HBM read/write bytes for materialized O(N^2) attention vs FlashAttention-2.

    Standard attention reads Q, K, V, writes S (N^2), reads S, writes P (N^2), reads P, V, writes O.
    FlashAttention-2 streams Q, K, V and writes O without reading/writing N^2 matrices to HBM.

    Returns:
        (standard_bytes, flash_bytes, reduction_factor)
    """
    qkv_o_bytes = 4 * seq_len * d_head * bytes_per_elem
    n2_matrix_bytes = 4 * (seq_len * seq_len) * bytes_per_elem  # write/read S and P
    standard_bytes = qkv_o_bytes + n2_matrix_bytes
    flash_bytes = qkv_o_bytes
    return standard_bytes, flash_bytes, standard_bytes / flash_bytes


def main() -> None:
    print("=" * 72)
    print("Stanford CS336 Lab: Online Softmax & Tiled FlashAttention-2 Recurrence")
    print("=" * 72)

    gen = torch.Generator().manual_seed(1337)
    q = torch.randn(32, 16, generator=gen)
    k = torch.randn(32, 16, generator=gen)
    v = torch.randn(32, 16, generator=gen)

    for causal in (False, True):
        out_flash = tiled_flash_attention_forward(q, k, v, block_q=8, block_kv=8, causal=causal)
        out_ref = standard_attention_reference(q, k, v, causal=causal)
        max_err = float(torch.max(torch.abs(out_flash - out_ref)).item())
        print(f"Causal={str(causal):<5} | Max |O_flash - O_ref|: {max_err:.2e}")

    std_b, flash_b, ratio = attention_hbm_bytes_comparison(seq_len=4096, d_head=128)
    print(
        f"\nHBM Traffic (N=4096, d=128, BF16): Standard={std_b / 1e6:.2f} MB vs "
        f"Flash={flash_b / 1e6:.2f} MB ({ratio:.1f}x reduction)"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
