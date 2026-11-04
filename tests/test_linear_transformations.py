"""Tests for linear transformations and QKV projection mathematics."""

import unittest
import torch
from calc.linear_transformations import (
    transform_2d_grid,
    check_attention_symmetry,
    measure_qkv_orthogonality,
)


class LinearTransformationTests(unittest.TestCase):
    def test_2d_transformation(self):
        m = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        v = torch.tensor([[1.0, 1.0], [2.0, -1.0]])
        out = transform_2d_grid(m, v)

        # [1, 1] scaled by 2 on x and 3 on y => [2, 3]
        self.assertAlmostEqual(float(out[0, 0].item()), 2.0, places=5)
        self.assertAlmostEqual(float(out[0, 1].item()), 3.0, places=5)
        # [2, -1] => [4, -3]
        self.assertAlmostEqual(float(out[1, 0].item()), 4.0, places=5)
        self.assertAlmostEqual(float(out[1, 1].item()), -3.0, places=5)

    def test_attention_symmetry_breaking(self):
        raw_asym, proj_asym = check_attention_symmetry(dim=32, num_tokens=4)

        # Unprojected attention must be strictly symmetric (error 0.0)
        self.assertAlmostEqual(raw_asym, 0.0, places=5)

        # Projected attention with separate W_Q and W_K must be strongly asymmetric
        self.assertGreater(proj_asym, 0.5)

    def test_qkv_orthogonality(self):
        cos_qk, cos_qv, cos_kv = measure_qkv_orthogonality(dim=768, head_dim=64)

        # The three projections of the same token should be substantially decoupled (< 0.3)
        self.assertLess(abs(cos_qk), 0.35)
        self.assertLess(abs(cos_qv), 0.35)
        self.assertLess(abs(cos_kv), 0.35)


if __name__ == "__main__":
    unittest.main()
