"""Tests for vector geometry intuitions."""

import unittest
import torch
from calc.math.vectors import (
    cosine_similarity,
    measure_dimension_orthogonality,
    demonstrate_subspace_superposition,
)


class VectorGeometryTests(unittest.TestCase):
    def test_cosine_similarity_basic(self):
        u = torch.tensor([1.0, 0.0])
        v = torch.tensor([0.0, 1.0])
        # Orthogonal vectors have cos 0
        self.assertAlmostEqual(cosine_similarity(u, v), 0.0, places=5)

        # Collinear vectors have cos 1
        w = torch.tensor([2.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(u, w), 1.0, places=5)

    def test_dimension_orthogonality_scaling(self):
        stats = measure_dimension_orthogonality(dimensions=(2, 768), num_pairs=500)
        _, std_2d, _ = stats[2]
        _, std_768d, _ = stats[768]

        # Standard deviation in 768D should be an order of magnitude smaller than 2D
        self.assertLess(std_768d, 0.1)
        self.assertGreater(std_2d, 0.5)

    def test_subspace_superposition_preserves_semantics(self):
        self_sim, max_other = demonstrate_subspace_superposition(dim=768)
        # The word combined with position should still strongly match its original self
        self.assertGreater(self_sim, 0.8)
        # And be vastly more similar than other distinct words
        self.assertGreater(self_sim - max_other, 0.7)


if __name__ == "__main__":
    unittest.main()
