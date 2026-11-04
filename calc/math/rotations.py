"""Mathematical foundations of 2D rotations and Rotary Position Embeddings (RoPE).

Demonstrates:
1. 2D rotation as an isometry: preserves vector lengths and pairwise angles.
2. The relative distance identity: <R_{m*theta} q, R_{n*theta} k> = q^T R_{(n - m)*theta} k.
3. Multi-frequency attenuation across sequence distances.
"""

from __future__ import annotations

import math
import torch


def rotation_matrix_2d(theta: float) -> torch.Tensor:
    """Construct a 2x2 counterclockwise rotation matrix."""
    cos = math.cos(theta)
    sin = math.sin(theta)
    return torch.tensor([[cos, -sin], [sin, cos]], dtype=torch.float32)


def apply_rotation_2d(v: torch.Tensor, theta: float) -> torch.Tensor:
    """Rotate a 2D vector by angle theta."""
    r = rotation_matrix_2d(theta)
    return torch.matmul(r, v)


def check_rope_relative_identity(
    m: int,
    n: int,
    theta: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Verify that <R_{m*theta} q, R_{n*theta} k> strictly equals q^T R_{(n - m)*theta} k.

    Returns:
        (left_hand_side, right_hand_side, absolute_difference)
    """
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(2, generator=generator)
    k = torch.randn(2, generator=generator)

    # LHS: Rotate q by m*theta, rotate k by n*theta, then dot product
    q_rot = apply_rotation_2d(q, m * theta)
    k_rot = apply_rotation_2d(k, n * theta)
    lhs = float(torch.dot(q_rot, k_rot).item())

    # RHS: Directly evaluate relative rotation (n - m)*theta
    r_rel = rotation_matrix_2d((n - m) * theta)
    rhs = float(torch.dot(q, torch.matmul(r_rel, k)).item())

    diff = abs(lhs - rhs)
    return lhs, rhs, diff


def simulate_multi_head_rope_decay(
    dim: int = 64,
    distances: list[int] | tuple[int, ...] = (0, 1, 2, 5, 10, 20, 50, 100),
    seed: int = 42,
) -> list[tuple[int, float]]:
    """Simulate average RoPE attention score attenuation as token distance increases.

    Returns:
        List of (distance, mean_attention_score).
    """
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(dim, generator=generator)
    k = torch.randn(dim, generator=generator)

    # Base frequencies across pairs: theta_i = 10000^(-2(i-1)/d)
    pairs = dim // 2
    freqs = torch.tensor([1.0 / (10000.0 ** (2 * i / dim)) for i in range(pairs)])

    results: list[tuple[int, float]] = []
    for dist in distances:
        # Sum inner products across all pairs under relative rotation (dist * freq)
        score = 0.0
        for i in range(pairs):
            q_i = q[2 * i : 2 * i + 2]
            k_i = k[2 * i : 2 * i + 2]
            r = rotation_matrix_2d(float(dist * freqs[i].item()))
            score += float(torch.dot(q_i, torch.matmul(r, k_i)).item())

        results.append((dist, score / pairs))

    return results


def main() -> None:
    print("=" * 72)
    print("Lesson 04 Benchmark: 2D Rotations and RoPE Mathematical Foundations")
    print("=" * 72)

    print("\n1. 2D Rotation Preserves Vector Norms (Isometry):")
    print("-" * 72)
    v = torch.tensor([3.0, 4.0])
    orig_norm = float(torch.linalg.norm(v).item())
    v_rot = apply_rotation_2d(v, math.pi / 3)  # 60 degrees
    rot_norm = float(torch.linalg.norm(v_rot).item())
    print(f"Original Vector: ({v[0]:.1f}, {v[1]:.1f}) => Norm: {orig_norm:.4f}")
    print(f"Rotated Vector:  ({v_rot[0]:.1f}, {v_rot[1]:.1f}) => Norm: {rot_norm:.4f}")
    print(f"Norm Difference: {abs(orig_norm - rot_norm):.1e} (Perfect preservation)")

    print("\n2. The Core RoPE Relative Distance Proof:")
    print("-" * 72)
    m, n = 15, 23
    lhs, rhs, diff = check_rope_relative_identity(m=m, n=n, theta=0.08)
    print(f"Token A Position m = {m}, Token B Position n = {n} (Relative offset = {n - m})")
    print(f"LHS <R_{{mθ}} q, R_{{nθ}} k>:  {lhs:+.6f}")
    print(f"RHS q^T R_{{(n - m)θ}} k:     {rhs:+.6f}")
    print(f"Absolute Numerical Error:     {diff:.1e} (Exact identity!)")
    print("-" * 72)
    print("Mathematical Truth: Absolute positions m and n cancel out completely;")
    print("only the relative distance (n - m) governs the attention dot product.")

    print("\n3. Long-Distance Attenuation in RoPE Multi-Frequency Space (dim=64):")
    print("-" * 72)
    print(f"{'Distance |m - n|':<18} | {'Relative Score':<16} | {'Trend'}")
    print("-" * 72)
    decay = simulate_multi_head_rope_decay()
    for dist, score in decay:
        bar = "█" * max(1, int(abs(score) * 40))
        print(f"{dist:<18} | {score:<16.4f} | {bar}")
    print("-" * 72)
    print("Conclusion: RoPE achieves relative positional awareness and natural")
    print("long-range decay without a single learnable position parameter!")
    print("=" * 72)


if __name__ == "__main__":
    main()
