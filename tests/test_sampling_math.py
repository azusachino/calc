"""Tests for Softmax, temperature scaling, entropy, and nucleus sampling."""

import math
import unittest
import torch
from calc.sampling_math import (
    stable_softmax,
    shannon_entropy,
    top_k_filter,
    top_p_filter,
    sample_categorical,
)


class SamplingMathTests(unittest.TestCase):
    def test_softmax_numerical_stability(self):
        # Huge logits that would overflow exp(1000.0) in standard float32
        logits = torch.tensor([1000.0, 1001.0, 999.0])
        probs = stable_softmax(logits, temperature=1.0)

        # Probs must sum to 1.0 and not contain NaNs
        self.assertFalse(torch.isnan(probs).any())
        self.assertAlmostEqual(float(probs.sum().item()), 1.0, places=5)
        # 1001.0 should have the highest probability
        self.assertEqual(int(torch.argmax(probs).item()), 1)

    def test_temperature_limits(self):
        logits = torch.tensor([5.0, 3.0, 1.0])

        # Cold temperature -> peak probability approaches 1.0
        p_cold = stable_softmax(logits, temperature=0.01)
        self.assertGreater(float(p_cold[0].item()), 0.99)

        # Hot temperature -> probabilities approach uniform (1/3 each)
        p_hot = stable_softmax(logits, temperature=100.0)
        for val in p_hot:
            self.assertAlmostEqual(float(val.item()), 1.0 / 3.0, places=1)

    def test_shannon_entropy_bounds(self):
        # Deterministic distribution -> Entropy = 0
        p_det = torch.tensor([1.0, 0.0, 0.0, 0.0])
        h_det = shannon_entropy(p_det)
        self.assertAlmostEqual(h_det, 0.0, places=4)

        # Uniform distribution over 4 choices -> Entropy = log2(4) = 2.0 bits
        p_unif = torch.tensor([0.25, 0.25, 0.25, 0.25])
        h_unif = shannon_entropy(p_unif)
        self.assertAlmostEqual(h_unif, 2.0, places=4)

    def test_top_k_filter(self):
        probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
        filtered = top_k_filter(probs, k=2)

        # Should only keep first two items and renormalize
        self.assertAlmostEqual(float(filtered[2].item()), 0.0, places=5)
        self.assertAlmostEqual(float(filtered[3].item()), 0.0, places=5)
        self.assertAlmostEqual(float(filtered.sum().item()), 1.0, places=5)
        # Relative ratio between 0.5 and 0.3 should be preserved: 0.5 / 0.8 = 0.625
        self.assertAlmostEqual(float(filtered[0].item()), 0.625, places=5)

    def test_top_p_nucleus_filter(self):
        probs = torch.tensor([0.6, 0.25, 0.1, 0.05])
        # Cumulative: 0.6 (<= 0.8), +0.25 = 0.85 (>= 0.8) -> stops here!
        filtered = top_p_filter(probs, p=0.8)

        self.assertGreater(float(filtered[0].item()), 0.0)
        self.assertGreater(float(filtered[1].item()), 0.0)
        self.assertAlmostEqual(float(filtered[2].item()), 0.0, places=5)
        self.assertAlmostEqual(float(filtered[3].item()), 0.0, places=5)
        self.assertAlmostEqual(float(filtered.sum().item()), 1.0, places=5)

    def test_sample_categorical(self):
        probs = torch.tensor([0.0, 1.0, 0.0])
        idx = sample_categorical(probs, seed=42)
        self.assertEqual(idx, 1)


if __name__ == "__main__":
    unittest.main()
