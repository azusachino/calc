"""Tests for Convex Optimization: strong duality, KKT stationarity, and ADMM Lasso."""

import unittest
import torch
from calc.math.convex import (
    soft_threshold,
    solve_lasso_admm,
    verify_ridge_strong_duality,
)


class ConvexOptimizationTests(unittest.TestCase):
    def test_soft_threshold_proximal_operator(self):
        v = torch.tensor([-3.0, -0.5, 0.0, 0.4, 2.5])
        res = soft_threshold(v, kappa=1.0)
        expected = torch.tensor([-2.0, 0.0, 0.0, 0.0, 1.5])
        self.assertTrue(torch.allclose(res, expected))

    def test_ridge_strong_duality_zero_gap(self):
        gen = torch.Generator().manual_seed(17)
        a_mat = torch.randn(16, 6, generator=gen)
        b_vec = torch.randn(16, generator=gen)
        p_opt, d_opt, gap = verify_ridge_strong_duality(a_mat, b_vec, lam=0.7)

        self.assertGreater(p_opt, 0.0)
        self.assertAlmostEqual(p_opt, d_opt, places=5)
        self.assertLess(gap, 1e-5)

    def test_lasso_admm_kkt_convergence(self):
        gen = torch.Generator().manual_seed(42)
        a_mat = torch.randn(24, 8, generator=gen)
        x_true = torch.tensor([3.0, 0.0, -2.0, 0.0, 0.0, 1.5, 0.0, 0.0])
        b_vec = a_mat @ x_true

        z_sol, _, primal_res, kkt_viol = solve_lasso_admm(
            a_mat, b_vec, lam=0.6, rho=2.0, steps=250
        )
        self.assertLess(primal_res, 1e-4)
        self.assertLess(kkt_viol, 1e-3)
        # Largest support indices should match true nonzeros (0, 2, 5)
        top3 = sorted(torch.topk(torch.abs(z_sol), k=3).indices.tolist())
        self.assertEqual(top3, [0, 2, 5])


if __name__ == "__main__":
    unittest.main()
