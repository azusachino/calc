"""Vector geometry intuitions for machine learning and LLM foundations.

Demonstrates:
1. 2D vector journey and parallelogram addition.
2. High-dimensional near-orthogonality (why 768D random vectors have cos(theta) ≈ 0).
3. Subspace superposition (why token_embedding + position_embedding works without collision).
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def cosine_similarity(u: torch.Tensor, v: torch.Tensor) -> float:
    """Compute cosine similarity between two 1D vectors."""
    dot = torch.dot(u, v)
    norm_u = torch.linalg.norm(u)
    norm_v = torch.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return float((dot / (norm_u * norm_v)).item())


def measure_dimension_orthogonality(
    dimensions: list[int] | tuple[int, ...] = (2, 8, 32, 128, 768, 4096),
    num_pairs: int = 1000,
    seed: int = 42,
) -> dict[int, tuple[float, float, float]]:
    """Empirically measure cosine similarities between random vectors across dimensions.

    Returns:
        Mapping from dimension -> (mean_abs_cos, std_cos, theoretical_std).
    """
    generator = torch.Generator().manual_seed(seed)
    results: dict[int, tuple[float, float, float]] = {}

    for d in dimensions:
        u = torch.randn(num_pairs, d, generator=generator)
        v = torch.randn(num_pairs, d, generator=generator)
        # Cosine similarity along dimension 1
        cos = F.cosine_similarity(u, v, dim=1)
        mean_abs = float(torch.mean(torch.abs(cos)).item())
        std = float(torch.std(cos).item())
        theoretical_std = 1.0 / math.sqrt(d)
        results[d] = (mean_abs, std, theoretical_std)

    return results


def demonstrate_subspace_superposition(
    dim: int = 768,
    num_tokens: int = 5,
    seed: int = 42,
) -> tuple[float, float]:
    """Demonstrate how high-dimensional vectors preserve semantic ranking after position addition.

    Returns:
        (semantic_retention_ratio, max_positional_drift).
    """
    generator = torch.Generator().manual_seed(seed)

    # Simulated semantic word embeddings
    tokens = torch.randn(num_tokens, dim, generator=generator)
    tokens = F.normalize(tokens, dim=1)

    # Position embeddings (e.g. learned vectors scaled down to typical norm)
    positions = torch.randn(num_tokens, dim, generator=generator)
    positions = F.normalize(positions, dim=1) * 0.5  # Typical ratio in trained models

    # Composite vectors: x = token + position
    x = tokens + positions

    # Check pairwise semantic ranking retention:
    # Does token 0 remain closest to itself in composite space compared to other tokens?
    original_self_sim = 1.0
    composite_self_sim = cosine_similarity(tokens[0], x[0])

    cross_sims = [cosine_similarity(tokens[0], tokens[i]) for i in range(1, num_tokens)]
    max_cross = max(cross_sims) if cross_sims else 0.0

    return composite_self_sim, max_cross


def main() -> None:
    print("=" * 68)
    print("Lesson 01 Exploratory Benchmark: High-Dimensional Vector Geometry")
    print("=" * 68)

    print("\n1. Measuring Near-Orthogonality Across Dimensions (N=1,000 random pairs):")
    print("-" * 68)
    print(f"{'Dimension (d)':<15} | {'Mean |cos θ|':<14} | {'Empirical σ':<14} | {'1/√d (Theory)':<14}")
    print("-" * 68)

    stats = measure_dimension_orthogonality()
    for d, (mean_abs, std, th_std) in stats.items():
        print(f"{d:<15} | {mean_abs:<14.4f} | {std:<14.4f} | {th_std:<14.4f}")

    print("-" * 68)
    print("Intuition: As d increases from 2 to 768, two random vectors are virtually")
    print("guaranteed to be perpendicular (dot product collapses to 0.00 ± 0.036).")

    print("\n2. Subspace Superposition in 768 Dimensions:")
    print("-" * 68)
    self_sim, max_other = demonstrate_subspace_superposition(dim=768)
    print(f"Similarity between 'cat' and ('cat' + pos_0):     {self_sim:.4f}")
    print(f"Similarity between 'cat' and other distinct words:  {max_other:.4f}")
    print(f"Semantic Signal Margin (Self vs Others):            +{self_sim - max_other:.4f}")
    print("-" * 68)
    print("Conclusion: Adding position vectors does NOT scramble token semantics;")
    print("it merely adds an orthogonal positional coordinate.")
    print("=" * 68)


if __name__ == "__main__":
    main()
