"""Spectral graph Laplacians, Personalized PageRank, and Message-Passing GNNs.

Companion lab for Stanford CS224W (Machine Learning with Graphs):
1. Graph Laplacian Dirichlet energy identity: x^T L x = 0.5 * sum_{i,j} A_{ij} (x_i - x_j)^2.
2. Personalized PageRank (PPR) random walk with restart for repository symbol ranking.
3. Kipf-Welling GCN degree-normalized message passing and over-smoothing decay.
"""

from __future__ import annotations

import torch


def laplacian_dirichlet_energy(
    adj: torch.Tensor,
    signal: torch.Tensor,
) -> tuple[float, float, float]:
    """Verify the Graph Laplacian quadratic form x^T L x == 0.5 * sum_{i,j} A_{ij} (x_i - x_j)^2.

     Also computes the algebraic connectivity lambda_2 (Fiedler eigenvalue) of L = D - A.

    Returns:
        (quadratic_form_energy, pairwise_diff_energy, fiedler_eigenvalue)
    """
    deg = torch.sum(adj, dim=1)
    lap = torch.diag(deg) - adj

    quad_energy = float((signal @ (lap @ signal)).item())
    diff = signal.unsqueeze(1) - signal.unsqueeze(0)
    pair_energy = 0.5 * float(torch.sum(adj * (diff * diff)).item())

    eigvals = torch.linalg.eigvalsh(lap)
    fiedler_val = float(eigvals[1].item()) if eigvals.numel() > 1 else 0.0
    return quad_energy, pair_energy, fiedler_val


def personalized_pagerank(
    adj: torch.Tensor,
    seed_indices: list[int] | tuple[int, ...],
    alpha: float = 0.15,
    steps: int = 100,
) -> torch.Tensor:
    """Compute Personalized PageRank (PPR) via random walk with restart on seed_indices.

    Recurrence:
        r^{(t+1)} = (1 - alpha) * M @ r^{(t)} + alpha * p_seed
    where M = A^T D^{-1} is the column-stochastic transition matrix.
    """
    n = adj.shape[0]
    if not seed_indices:
        raise ValueError("seed_indices must be non-empty")

    p_seed = torch.zeros(n, dtype=adj.dtype, device=adj.device)
    for idx in seed_indices:
        p_seed[idx] += 1.0 / len(seed_indices)

    out_deg = torch.sum(adj, dim=1)
    inv_deg = torch.where(out_deg > 0, 1.0 / out_deg, torch.zeros_like(out_deg))
    m_trans = adj.transpose(0, 1) @ torch.diag(inv_deg)

    r = p_seed.clone()
    for _ in range(steps):
        r = (1.0 - alpha) * (m_trans @ r) + alpha * p_seed
        # Re-inject leaked mass from any dangling nodes (out_deg == 0)
        missing = 1.0 - float(torch.sum(r).item())
        if abs(missing) > 1e-7:
            r = r + missing * p_seed

    return r / torch.sum(r)


def measure_gcn_oversmoothing(
    adj: torch.Tensor,
    features: torch.Tensor,
    num_layers: int = 8,
) -> list[float]:
    """Measure node feature Dirichlet energy across repeated Kipf-Welling GCN propagation steps.

    Uses the symmetric normalized adjacency with self-loops:
        S = D_tilde^{-1/2} (A + I) D_tilde^{-1/2}
    and tracks the normalized Dirichlet energy ||(I - S) H^(l)||_F^2 / ||H^(l)||_F^2.
    """
    n = adj.shape[0]
    eye = torch.eye(n, dtype=adj.dtype, device=adj.device)
    a_tilde = adj + eye
    deg_tilde = torch.sum(a_tilde, dim=1)
    d_inv_sqrt = torch.diag(torch.pow(deg_tilde, -0.5))
    s_prop = d_inv_sqrt @ a_tilde @ d_inv_sqrt
    l_sym = eye - s_prop

    h = features.clone()
    energies: list[float] = []
    for _ in range(num_layers + 1):
        norm_sq = float(torch.sum(h * h).item())
        lh = l_sym @ h
        dirichlet_ratio = float(torch.sum(h * lh).item()) / max(norm_sq, 1e-12)
        energies.append(dirichlet_ratio)
        h = s_prop @ h

    return energies


def main() -> None:
    print("=" * 72)
    print("Stanford CS224W Lab: Spectral Laplacians, Personalized PageRank & GNNs")
    print("=" * 72)

    # Barbell-like two-cluster graph (nodes 0,1,2 and 3,4,5 connected by bridge 2-3)
    adj = torch.tensor(
        [
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        ]
    )
    signal = torch.tensor([1.0, 1.0, 0.8, -0.8, -1.0, -1.0])
    q_e, p_e, lam2 = laplacian_dirichlet_energy(adj, signal)
    print(f"Dirichlet Energy x^T L x = {q_e:.4f} | Pairwise = {p_e:.4f} | Fiedler lambda_2 = {lam2:.4f}")

    ppr = personalized_pagerank(adj, seed_indices=[0], alpha=0.2)
    print(f"PPR from seed node 0: {[round(x, 4) for x in ppr.tolist()]}")

    gen = torch.Generator().manual_seed(7)
    feats = torch.randn(6, 4, generator=gen)
    energies = measure_gcn_oversmoothing(adj, feats, num_layers=6)
    print(f"GCN Dirichlet Energy Decay (L=0..6): {[round(e, 4) for e in energies]}")
    print("=" * 72)


if __name__ == "__main__":
    main()
