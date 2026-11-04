"""Mathematical foundations of dot products and scaled attention.

Demonstrates:
1. Geometric projection and angular alignment (the similarity flashlight).
2. Variance explosion in high dimensions: Var(q · k) = d_k.
3. Softmax saturation and vanishing gradients without 1/√d_k scaling.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def dot_product_and_projection(u: torch.Tensor, v: torch.Tensor) -> tuple[float, float, float]:
    """Calculate dot product, length of projection of u onto v, and angle in degrees.

    Returns:
        (dot_product, projection_length, angle_degrees)
    """
    dot = float(torch.dot(u, v).item())
    norm_u = float(torch.linalg.norm(u).item())
    norm_v = float(torch.linalg.norm(v).item())

    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0, 0.0, 90.0

    cos_theta = max(-1.0, min(1.0, dot / (norm_u * norm_v)))
    angle_deg = math.degrees(math.acos(cos_theta))
    proj_len = dot / norm_v  # Length of u projected onto v's direction

    return dot, proj_len, angle_deg


def measure_attention_variance(
    d_k_list: list[int] | tuple[int, ...] = (16, 64, 128, 256, 768),
    num_samples: int = 10_000,
    seed: int = 42,
) -> dict[int, tuple[float, float, float, float]]:
    """Measure empirical variance of raw vs scaled dot products across key dimensions.

    Returns:
        Mapping d_k -> (raw_var, raw_std, scaled_var, scaled_std).
    """
    generator = torch.Generator().manual_seed(seed)
    results: dict[int, tuple[float, float, float, float]] = {}

    for d_k in d_k_list:
        # q, k ~ N(0, 1) independently
        q = torch.randn(num_samples, d_k, generator=generator)
        k = torch.randn(num_samples, d_k, generator=generator)

        # Raw dot products: sum(q_i * k_i) along feature dimension
        raw_dot = torch.sum(q * k, dim=1)
        raw_var = float(torch.var(raw_dot).item())
        raw_std = float(torch.std(raw_dot).item())

        # Scaled dot products: (q · k) / sqrt(d_k)
        scaled_dot = raw_dot / math.sqrt(d_k)
        scaled_var = float(torch.var(scaled_dot).item())
        scaled_std = float(torch.std(scaled_dot).item())

        results[d_k] = (raw_var, raw_std, scaled_var, scaled_std)

    return results


def measure_softmax_saturation(
    d_k: int = 64,
    seq_len: int = 16,
    num_trials: int = 1_000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Compare softmax peak probability and entropy between raw and scaled attention scores.

    Returns:
        (raw_max_prob, raw_entropy, scaled_max_prob, scaled_entropy).
    """
    generator = torch.Generator().manual_seed(seed)

    # Simulate query and keys
    # q: [trials, 1, d_k], k: [trials, seq_len, d_k]
    q = torch.randn(num_trials, 1, d_k, generator=generator)
    k = torch.randn(num_trials, seq_len, d_k, generator=generator)

    # Raw attention logits: [trials, 1, seq_len]
    raw_logits = torch.matmul(q, k.transpose(1, 2))
    raw_probs = F.softmax(raw_logits, dim=-1)
    raw_max_prob = float(torch.mean(torch.max(raw_probs, dim=-1).values).item())
    raw_entropy = float(torch.mean(-torch.sum(raw_probs * torch.log(raw_probs + 1e-12), dim=-1)).item())

    # Scaled attention logits: logits / sqrt(d_k)
    scaled_logits = raw_logits / math.sqrt(d_k)
    scaled_probs = F.softmax(scaled_logits, dim=-1)
    scaled_max_prob = float(torch.mean(torch.max(scaled_probs, dim=-1).values).item())
    scaled_entropy = float(torch.mean(-torch.sum(scaled_probs * torch.log(scaled_probs + 1e-12), dim=-1)).item())

    return raw_max_prob, raw_entropy, scaled_max_prob, scaled_entropy


def main() -> None:
    print("=" * 72)
    print("Lesson 02 Benchmark: Dot Products, Variance Explosion, and Scaled Attention")
    print("=" * 72)

    print("\n1. Dot Product as a Similarity Flashlight (2D Alignment):")
    print("-" * 72)
    u = torch.tensor([3.0, 4.0])  # norm 5
    v_same = torch.tensor([3.0, 4.0])
    v_ortho = torch.tensor([-4.0, 3.0])  # perpendicular
    v_oppo = torch.tensor([-3.0, -4.0])

    for label, v in [("Parallel (Identical)", v_same), ("Orthogonal (Perpendicular)", v_ortho), ("Opposite (Reverse)", v_oppo)]:
        dot, proj, deg = dot_product_and_projection(u, v)
        print(f"u · {label:<24} => Dot: {dot:+6.1f} | Proj on v: {proj:+5.1f} | Angle: {deg:5.1f}°")

    print("\n2. The Variance Explosion in Raw Dot Products (N=10,000 pairs):")
    print("-" * 72)
    print(f"{'Key Dim (d_k)':<14} | {'Raw Var (≈ d_k)':<16} | {'Raw Std (≈ √d_k)':<18} | {'Scaled Std (1/√d_k)':<16}")
    print("-" * 72)

    stats = measure_attention_variance()
    for d_k, (r_var, r_std, s_var, s_std) in stats.items():
        print(f"{d_k:<14} | {r_var:<16.2f} | {r_std:<18.2f} | {s_std:<16.2f}")

    print("-" * 72)
    print("Mathematical Truth: Summing d_k random products multiplies variance by d_k.")
    print("For d_k=768, raw dot products swing between ±27! Softmax would immediately freeze.")

    print("\n3. Softmax Saturation and Entropy Collapse (d_k=64, seq_len=16):")
    print("-" * 72)
    r_max, r_ent, s_max, s_ent = measure_softmax_saturation(d_k=64, seq_len=16)
    uniform_entropy = math.log(16)
    print(f"Max Uniform Entropy (All keys equal):          {uniform_entropy:.3f}")
    print(f"Raw Attention (No 1/√d_k):   Peak Prob: {r_max*100:5.1f}%  | Entropy: {r_ent:.3f} (Severely saturated!)")
    print(f"Scaled Attention (With 1/√d_k): Peak Prob: {s_max*100:5.1f}%  | Entropy: {s_ent:.3f} (Smooth, healthy gradients)")
    print("-" * 72)
    print("Conclusion: Dividing by √d_k keeps logits in the linear, gradient-rich")
    print("regime of Softmax, enabling the model to learn multiple relevant tokens.")
    print("=" * 72)


if __name__ == "__main__":
    main()
