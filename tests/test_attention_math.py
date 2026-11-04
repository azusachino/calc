"""Tests for dot products and scaled attention mathematics."""

import unittest
import torch
from calc.attention_math import (
    dot_product_and_projection,
    measure_attention_variance,
    measure_softmax_saturation,
)


class AttentionMathTests(unittest.TestCase):
    def test_dot_product_and_projection(self):
        u = torch.tensor([3.0, 4.0])
        v_same = torch.tensor([3.0, 4.0])
        v_ortho = torch.tensor([-4.0, 3.0])

        dot_same, proj_same, angle_same = dot_product_and_projection(u, v_same)
        self.assertAlmostEqual(dot_same, 25.0, places=5)
        self.assertAlmostEqual(proj_same, 5.0, places=5)
        self.assertAlmostEqual(angle_same, 0.0, places=5)

        dot_ortho, proj_ortho, angle_ortho = dot_product_and_projection(u, v_ortho)
        self.assertAlmostEqual(dot_ortho, 0.0, places=5)
        self.assertAlmostEqual(proj_ortho, 0.0, places=5)
        self.assertAlmostEqual(angle_ortho, 90.0, places=5)

    def test_variance_scaling(self):
        stats = measure_attention_variance(d_k_list=(64,), num_samples=5_000)
        raw_var, raw_std, scaled_var, scaled_std = stats[64]

        # Raw variance should be around 64, std around 8
        self.assertTrue(50.0 < raw_var < 80.0)
        self.assertTrue(7.0 < raw_std < 9.0)

        # Scaled variance and std should be strictly around 1.0
        self.assertTrue(0.85 < scaled_var < 1.15)
        self.assertTrue(0.90 < scaled_std < 1.10)

    def test_softmax_saturation(self):
        raw_max, raw_ent, scaled_max, scaled_ent = measure_softmax_saturation(d_k=64, seq_len=16)

        # Raw attention should be far peakier (higher max prob, lower entropy)
        self.assertGreater(raw_max, scaled_max)
        self.assertLess(raw_ent, scaled_ent)


if __name__ == "__main__":
    unittest.main()
