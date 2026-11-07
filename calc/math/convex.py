"""Convex optimization, KKT duality, and ADMM proximal splitting.

Companion lab for Stanford EE364a (Convex Optimization):
1. Closed-form soft-thresholding proximal operator for the L1 norm.
2. Alternating Direction Method of Multipliers (ADMM) for Lasso regression.
3. KKT subgradient stationarity and ridge-regression strong duality verification.
"""

from __future__ import annotations

import torch


def soft_threshold(v: torch.Tensor, kappa: float) -> torch.Tensor:
    """Apply the elementwise L1 proximal soft-thresholding operator S_kappa(v)."""
    if kappa < 0.0:
        raise ValueError("Threshold kappa must be non-negative")
    return torch.sign(v) * torch.clamp(torch.abs(v) - kappa, min=0.0)


def solve_lasso_admm(
    a_mat: torch.Tensor,
    b_vec: torch.Tensor,
    lam: float = 0.5,
    rho: float = 1.0,
    steps: int = 150,
) -> tuple[torch.Tensor, float, float, float]:
    """Solve Lasso min_x 0.5 * ||A x - b||_2^2 + lam * ||x||_1 via scaled ADMM.

    Splits into:
        min_{x, z} 0.5 * ||A x - b||_2^2 + lam * ||z||_1   subject to   x - z = 0

    Returns:
        (z_solution, objective_value, primal_residual_norm, kkt_subgrad_violation)
    """
    _, n = a_mat.shape
    x = torch.zeros(n, dtype=a_mat.dtype, device=a_mat.device)
    z = torch.zeros(n, dtype=a_mat.dtype, device=a_mat.device)
    u = torch.zeros(n, dtype=a_mat.dtype, device=a_mat.device)

    eye = torch.eye(n, dtype=a_mat.dtype, device=a_mat.device)
    lhs = a_mat.transpose(0, 1) @ a_mat + rho * eye
    at_b = a_mat.transpose(0, 1) @ b_vec

    for _ in range(steps):
        rhs = at_b + rho * (z - u)
        x = torch.linalg.solve(lhs, rhs)
        z = soft_threshold(x + u, lam / rho)
        u = u + (x - z)

    primal_res = float(torch.linalg.norm(x - z).item())
    residual = a_mat @ z - b_vec
    obj = 0.5 * float(torch.sum(residual * residual).item()) + lam * float(torch.sum(torch.abs(z)).item())

    # KKT subgradient check: A^T (b - A z) in lam * partial ||z||_1
    grad_neg = a_mat.transpose(0, 1) @ (b_vec - a_mat @ z)
    support = torch.abs(z) > 1e-4
    inactive = ~support

    viol_active = (
        float(torch.max(torch.abs(grad_neg[support] - lam * torch.sign(z[support]))).item())
        if torch.any(support)
        else 0.0
    )
    viol_inactive = (
        float(torch.max(torch.clamp(torch.abs(grad_neg[inactive]) - lam, min=0.0)).item())
        if torch.any(inactive)
        else 0.0
    )
    kkt_violation = max(viol_active, viol_inactive)

    return z, obj, primal_res, kkt_violation


def verify_ridge_strong_duality(
    a_mat: torch.Tensor,
    b_vec: torch.Tensor,
    lam: float = 0.5,
) -> tuple[float, float, float]:
    """Verify strong duality (p* == d*, zero duality gap) for strictly convex Ridge QP.

    Primal:
        min_x  0.5 * ||A x - b||_2^2 + 0.5 * lam * ||x||_2^2
    Lagrangian Dual (over residual variable nu = b - A x):
        max_nu -0.5 * ||nu||_2^2 - (1 / (2 * lam)) * ||A^T nu||_2^2 + b^T nu

    Returns:
        (primal_opt, dual_opt, duality_gap)
    """
    _, n = a_mat.shape
    eye = torch.eye(n, dtype=a_mat.dtype, device=a_mat.device)
    x_star = torch.linalg.solve(a_mat.transpose(0, 1) @ a_mat + lam * eye, a_mat.transpose(0, 1) @ b_vec)
    nu_star = b_vec - a_mat @ x_star

    primal_opt = 0.5 * float(torch.sum(nu_star * nu_star).item()) + 0.5 * lam * float(
        torch.sum(x_star * x_star).item()
    )
    at_nu = a_mat.transpose(0, 1) @ nu_star
    dual_opt = (
        -0.5 * float(torch.sum(nu_star * nu_star).item())
        - (0.5 / lam) * float(torch.sum(at_nu * at_nu).item())
        + float(torch.dot(b_vec, nu_star).item())
    )
    gap = abs(primal_opt - dual_opt)
    return primal_opt, dual_opt, gap


def main() -> None:
    print("=" * 72)
    print("Stanford EE364a Lab: Strong Duality, KKT Optimality, and ADMM Lasso")
    print("=" * 72)

    gen = torch.Generator().manual_seed(42)
    a_mat = torch.randn(20, 8, generator=gen)
    true_x = torch.tensor([2.5, 0.0, -1.8, 0.0, 0.0, 1.2, 0.0, 0.0])
    b_vec = a_mat @ true_x + 0.05 * torch.randn(20, generator=gen)

    p_val, d_val, gap = verify_ridge_strong_duality(a_mat, b_vec, lam=0.5)
    print(f"Ridge QP Strong Duality: p*={p_val:.6f} | d*={d_val:.6f} | Gap={gap:.2e}")

    z_sol, obj, p_res, kkt_viol = solve_lasso_admm(a_mat, b_vec, lam=0.8, rho=2.0, steps=200)
    nonzeros = int(torch.sum(torch.abs(z_sol) > 1e-3).item())
    print(
        f"ADMM Lasso: Obj={obj:.4f} | Nonzeros={nonzeros}/8 | "
        f"Primal Res={p_res:.2e} | KKT Subgrad Viol={kkt_viol:.2e}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
