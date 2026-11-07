"""Tests for MDPs, Bellman gamma-contraction, GAE, and GRPO."""

import unittest
import torch
from calc.math.rl import (
    bellman_optimality_step,
    compute_gae,
    grpo_group_loss,
    solve_mdp_value_iteration,
)


class ReinforcementLearningMathTests(unittest.TestCase):
    def test_bellman_gamma_contraction_in_sup_norm(self):
        transitions = torch.tensor(
            [
                [[0.7, 0.3], [0.2, 0.8]],
                [[0.4, 0.6], [0.9, 0.1]],
            ],
            dtype=torch.float32,
        )
        rewards = torch.tensor([[1.0, 0.5], [0.2, 2.0]], dtype=torch.float32)
        gamma = 0.85

        v1 = torch.tensor([10.0, -4.0])
        v2 = torch.tensor([-2.0, 6.0])
        tv1, _ = bellman_optimality_step(transitions, rewards, v1, gamma=gamma)
        tv2, _ = bellman_optimality_step(transitions, rewards, v2, gamma=gamma)

        dist_in = float(torch.max(torch.abs(v1 - v2)).item())
        dist_out = float(torch.max(torch.abs(tv1 - tv2)).item())
        self.assertLessEqual(dist_out, gamma * dist_in + 1e-5)

        v_star, _, steps = solve_mdp_value_iteration(transitions, rewards, gamma=gamma)
        tv_star, _ = bellman_optimality_step(transitions, rewards, v_star, gamma=gamma)
        self.assertLess(steps, 200)
        self.assertLess(float(torch.max(torch.abs(tv_star - v_star)).item()), 1e-5)

    def test_gae_boundary_cases(self):
        rewards = torch.tensor([1.0, 1.0, 1.0])
        values = torch.tensor([0.5, 0.5, 0.5])

        # lam = 0.0 reduces strictly to 1-step TD error delta_t = r_t + gamma * V_{t+1} - V_t
        adv_td0, _ = compute_gae(rewards, values, next_value=0.5, gamma=0.9, lam=0.0)
        expected_delta = 1.0 + 0.9 * 0.5 - 0.5
        for val in adv_td0:
            self.assertAlmostEqual(float(val.item()), expected_delta, places=5)

    def test_grpo_zero_mean_advantages_and_non_negative_kl(self):
        log_probs = torch.tensor([-1.0, -2.0, -0.5, -1.5])
        old_log_probs = torch.tensor([-1.1, -1.9, -0.6, -1.4])
        ref_log_probs = torch.tensor([-1.5, -1.5, -1.0, -1.2])
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])

        _, adv, k3_kl = grpo_group_loss(log_probs, old_log_probs, ref_log_probs, rewards)
        self.assertAlmostEqual(float(torch.mean(adv).item()), 0.0, places=5)
        self.assertTrue(torch.all(k3_kl >= 0.0))


if __name__ == "__main__":
    unittest.main()
