"""Tests for strided NDArray memory views and reverse-mode automatic differentiation."""

import unittest
import torch
from calc.math.autograd import (
    StridedView,
    reverse_mode_linear_gelu_step,
    sum_to_shape,
)


class AutogradAndStridedViewTests(unittest.TestCase):
    def test_zero_copy_permute_and_compact(self):
        view = StridedView.from_flat([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], (2, 3))
        self.assertTrue(view.is_compact())
        self.assertEqual(view.strides, (3, 1))

        perm = view.permute((1, 0))
        self.assertEqual(perm.shape, (3, 2))
        self.assertEqual(perm.strides, (1, 3))
        self.assertFalse(perm.is_compact())
        self.assertEqual(perm.get((2, 1)), 5.0)

        compacted = perm.compact()
        self.assertTrue(compacted.is_compact())
        self.assertEqual(compacted.shape, (3, 2))
        self.assertEqual(compacted.strides, (2, 1))
        self.assertEqual(compacted.data, (0.0, 3.0, 1.0, 4.0, 2.0, 5.0))

    def test_zero_copy_broadcast_and_sum_to_shape(self):
        col = StridedView.from_flat([10.0, 20.0], (2, 1))
        bcast = col.broadcast_to((2, 4))
        self.assertEqual(bcast.strides, (1, 0))
        for j in range(4):
            self.assertEqual(bcast.get((0, j)), 10.0)
            self.assertEqual(bcast.get((1, j)), 20.0)

        grad = torch.ones(3, 2, 4)
        reduced = sum_to_shape(grad, (2, 1))
        self.assertEqual(tuple(reduced.shape), (2, 1))
        self.assertTrue(torch.allclose(reduced, torch.full((2, 1), 12.0)))

    def test_reverse_mode_adjoints_match_pytorch(self):
        gen = torch.Generator().manual_seed(42)
        x = torch.randn(4, 5, generator=gen)
        w = torch.randn(5, 3, generator=gen)
        b = torch.randn(1, 3, generator=gen)

        _, _, errors = reverse_mode_linear_gelu_step(x, w, b)
        for name, err in errors.items():
            self.assertLess(err, 1e-5, f"Gradient error for {name} too large: {err}")


if __name__ == "__main__":
    unittest.main()
