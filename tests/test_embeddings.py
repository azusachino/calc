import unittest

import torch

from calc.embeddings import (
    LearnedPositionalEmbedding,
    RotaryPositionalEmbedding,
    SinusoidalPositionalEncoding,
)


class PositionalEmbeddingTests(unittest.TestCase):
    def test_learned_positional_embedding_shapes(self):
        emb = LearnedPositionalEmbedding(block_size=16, embedding_size=32)
        pos = emb(8)
        self.assertEqual(tuple(pos.shape), (8, 32))

        with self.assertRaises(ValueError):
            emb(17)

    def test_sinusoidal_encoding_properties(self):
        with self.assertRaises(ValueError):
            SinusoidalPositionalEncoding(embedding_size=15)  # odd dimension

        sin_emb = SinusoidalPositionalEncoding(embedding_size=32, max_length=16)
        pe = sin_emb(8)
        self.assertEqual(tuple(pe.shape), (8, 32))
        self.assertTrue(torch.all(pe >= -1.0) and torch.all(pe <= 1.0))

        # At pos 0: sin(0) = 0, cos(0) = 1
        self.assertAlmostEqual(pe[0, 0].item(), 0.0, places=5)
        self.assertAlmostEqual(pe[0, 1].item(), 1.0, places=5)

        # Dynamic generation for length > max_length
        extended_pe = sin_emb(32)
        self.assertEqual(tuple(extended_pe.shape), (32, 32))

    def test_rope_shapes_and_dimensions(self):
        with self.assertRaises(ValueError):
            RotaryPositionalEmbedding(head_size=31)  # odd head_size

        rope = RotaryPositionalEmbedding(head_size=16, max_length=32)

        # 4D: [batch, heads, length, head_size]
        x_4d = torch.randn(2, 4, 10, 16)
        out_4d = rope(x_4d)
        self.assertEqual(tuple(out_4d.shape), (2, 4, 10, 16))

        # 3D: [batch, length, head_size]
        x_3d = torch.randn(2, 10, 16)
        out_3d = rope(x_3d)
        self.assertEqual(tuple(out_3d.shape), (2, 10, 16))

        # Dynamic extension beyond max_length
        x_long = torch.randn(1, 1, 64, 16)
        out_long = rope(x_long)
        self.assertEqual(tuple(out_long.shape), (1, 1, 64, 16))

    def test_rope_orthogonality(self):
        """RoPE is an orthogonal transformation; norm and dot products at the same position are invariant."""
        rope = RotaryPositionalEmbedding(head_size=16, max_length=32)
        x = torch.randn(1, 1, 1, 16)
        y = torch.randn(1, 1, 1, 16)

        # Both at position 0
        rx = rope(x)
        ry = rope(y)

        orig_dot = torch.dot(x.flatten(), y.flatten())
        rotated_dot = torch.dot(rx.flatten(), ry.flatten())
        torch.testing.assert_close(orig_dot, rotated_dot, rtol=1e-4, atol=1e-4)

    def test_rope_relative_shift_invariance(self):
        """Inner product <R_{m+k} x, R_{n+k} y> equals <R_m x, R_n y> for any shift k."""
        rope = RotaryPositionalEmbedding(head_size=16, max_length=64)
        x = torch.randn(16)
        y = torch.randn(16)

        # Pair 1: positions m=2 and n=5 (shift 0)
        seq_base = torch.zeros(1, 1, 10, 16)
        seq_base[0, 0, 2] = x
        seq_base[0, 0, 5] = y
        rot_base = rope(seq_base)
        dot_base = torch.dot(rot_base[0, 0, 2], rot_base[0, 0, 5])

        # Pair 2: positions (2+k) and (5+k) with shift k=4 -> positions 6 and 9
        k = 4
        seq_shifted = torch.zeros(1, 1, 15, 16)
        seq_shifted[0, 0, 2 + k] = x
        seq_shifted[0, 0, 5 + k] = y
        rot_shifted = rope(seq_shifted)
        dot_shifted = torch.dot(rot_shifted[0, 0, 2 + k], rot_shifted[0, 0, 5 + k])

        torch.testing.assert_close(dot_base, dot_shifted, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
