"""Tests for gradient descent, autograd equivalence, and residual gradient highway."""

import unittest
from calc.gradient_descent import (
    quadratic_loss_and_analytical_grad,
    compute_autograd,
    run_gradient_descent,
    measure_residual_gradient_highway,
)


class GradientDescentTests(unittest.TestCase):
    def test_analytical_vs_autograd_equivalence(self):
        test_points = [(1.0, 1.0), (-3.5, 2.8), (0.0, 0.0), (10.0, -15.0)]
        for x, y in test_points:
            loss_ana, (gx_ana, gy_ana) = quadratic_loss_and_analytical_grad(x, y)
            loss_auto, (gx_auto, gy_auto) = compute_autograd(x, y)

            self.assertAlmostEqual(loss_ana, loss_auto, places=5)
            self.assertAlmostEqual(gx_ana, gx_auto, places=5)
            self.assertAlmostEqual(gy_ana, gy_auto, places=5)

    def test_gradient_descent_convergence(self):
        traj = run_gradient_descent(start_pos=(5.0, 5.0), lr=0.1, steps=10)
        initial_loss = traj[0][3]
        final_loss = traj[-1][3]

        # Loss should decrease monotonically
        self.assertLess(final_loss, initial_loss)
        # After 10 steps with lr=0.1, loss should decrease by more than 80%
        self.assertLess(final_loss, initial_loss * 0.2)

    def test_residual_gradient_highway(self):
        plain_norm, res_norm = measure_residual_gradient_highway(depth=8)
        # Residual connection should retain an order of magnitude more gradient norm
        self.assertGreater(res_norm, plain_norm * 5.0)


if __name__ == "__main__":
    unittest.main()
