"""Tests for Spectral Graph Laplacians, Personalized PageRank, and GNN over-smoothing."""

import unittest
import torch
from calc.math.graphs import (
    laplacian_dirichlet_energy,
    measure_gcn_oversmoothing,
    personalized_pagerank,
)


class GraphMathematicsTests(unittest.TestCase):
    def setUp(self):
        self.adj = torch.tensor(
            [
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )

    def test_laplacian_dirichlet_quadratic_identity(self):
        signal = torch.tensor([2.0, -1.0, 0.5, 3.0])
        quad_e, pair_e, lam2 = laplacian_dirichlet_energy(self.adj, signal)

        self.assertAlmostEqual(quad_e, pair_e, places=5)
        # Connected graph must have strictly positive Fiedler eigenvalue lambda_2 > 0
        self.assertGreater(lam2, 0.1)

        # Constant signal has zero Dirichlet energy
        const_q, const_p, _ = laplacian_dirichlet_energy(self.adj, torch.ones(4))
        self.assertAlmostEqual(const_q, 0.0, places=6)
        self.assertAlmostEqual(const_p, 0.0, places=6)

    def test_personalized_pagerank_distribution(self):
        ppr = personalized_pagerank(self.adj, seed_indices=[0], alpha=0.25)
        self.assertAlmostEqual(float(ppr.sum().item()), 1.0, places=5)
        self.assertTrue(torch.all(ppr > 0.0))
        # Seed node 0 and its immediate clique neighbors (1, 2) outrank leaf node 3
        self.assertGreater(float(ppr[0].item()), float(ppr[3].item()))

    def test_gcn_oversmoothing_energy_decay(self):
        gen = torch.Generator().manual_seed(11)
        features = torch.randn(4, 3, generator=gen)
        energies = measure_gcn_oversmoothing(self.adj, features, num_layers=6)
        # Dirichlet energy ratio should decrease after repeated Laplacian smoothing
        self.assertLess(energies[-1], energies[0])


if __name__ == "__main__":
    unittest.main()
