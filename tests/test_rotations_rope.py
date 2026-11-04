"""Tests for 2D rotations and Rotary Position Embeddings (RoPE) mathematics."""

import math
import unittest
import torch
from calc.rotations_rope import (
    rotation_matrix_2d,
    apply_rotation_2d,
    check_rope_relative_identity,
    simulate_multi_head_rope_decay,
)


class RotationsRoPETests(unittest.TestCase):
    def test_rotation_isometry_norm_preservation(self):
        v = torch.tensor([3.0, 4.0])
        orig_norm = float(torch.linalg.norm(v).item())
        v_rot = apply_rotation_2d(v, math.pi / 4)
        rot_norm = float(torch.linalg.norm(v_rot).item())

        self.assertAlmostEqual(orig_norm, 5.0, places=5)
        self.assertAlmostEqual(orig_norm, rot_norm, places=5)

    def test_rope_relative_distance_identity(self):
        # Test across multiple token pairs
        test_pairs = [(0, 1), (5, 12), (100, 150), (42, 42)]
        for m, n in test_pairs:
            lhs, rhs, diff = check_rope_relative_identity(m=m, n=n, theta=0.05)
            self.assertAlmostEqual(lhs, rhs, places=5)
            self.assertLess(diff, 1e-5)

    def test_multi_head_rope_structure(self):
        distances = (0, 1, 5, 20)
        decay = simulate_multi_head_rope_decay(dim=32, distances=distances)
        self.assertEqual(len(decay), 4)
        # Distance 0 should yield a valid float
        self.assertIsInstance(decay[0][1], float)


if __name__ == "__main__":
    unittest.main()
