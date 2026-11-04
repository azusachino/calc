"""Mathematical foundations of gradients, loss landscapes, and backpropagation.

Demonstrates:
1. Analytical gradient vs PyTorch autograd equivalence on an anisotropic 2D loss surface.
2. Gradient descent trajectory and convergence along negative gradient direction.
3. The gradient highway: why residual connections prevent exponential vanishing gradients.
"""

from __future__ import annotations

import torch


def quadratic_loss_and_analytical_grad(
    x: float,
    y: float,
    a: float = 1.0,
    b: float = 4.0,
) -> tuple[float, tuple[float, float]]:
    """Compute L(x, y) = a*x^2 + b*y^2 and its analytical gradient [2*a*x, 2*b*y]."""
    loss = a * (x**2) + b * (y**2)
    grad_x = 2.0 * a * x
    grad_y = 2.0 * b * y
    return loss, (grad_x, grad_y)


def compute_autograd(
    x_val: float,
    y_val: float,
    a: float = 1.0,
    b: float = 4.0,
) -> tuple[float, tuple[float, float]]:
    """Compute loss and gradient using PyTorch reverse-mode autograd."""
    x = torch.tensor(x_val, requires_grad=True, dtype=torch.float32)
    y = torch.tensor(y_val, requires_grad=True, dtype=torch.float32)

    loss = a * (x**2) + b * (y**2)
    loss.backward()

    assert x.grad is not None and y.grad is not None
    return float(loss.item()), (float(x.grad.item()), float(y.grad.item()))


def run_gradient_descent(
    start_pos: tuple[float, float] = (5.0, 5.0),
    lr: float = 0.1,
    steps: int = 10,
    a: float = 1.0,
    b: float = 4.0,
) -> list[tuple[int, float, float, float]]:
    """Simulate 2D gradient descent trajectory.

    Returns:
        List of (step, x, y, loss).
    """
    x, y = start_pos
    trajectory: list[tuple[int, float, float, float]] = []

    for step in range(steps + 1):
        loss, (gx, gy) = quadratic_loss_and_analytical_grad(x, y, a=a, b=b)
        trajectory.append((step, x, y, loss))
        # Parameter update along negative gradient
        x -= lr * gx
        y -= lr * gy

    return trajectory


def measure_residual_gradient_highway(depth: int = 8) -> tuple[float, float]:
    """Compare gradient norm reaching the input with and without residual connections.

    Returns:
        (plain_input_grad_norm, residual_input_grad_norm)
    """
    torch.manual_seed(42)

    # 1. Plain chain: x_{l+1} = W_l * x_l with small weights (spectral radius < 1)
    x_plain = torch.ones(1, 16, requires_grad=True)
    curr = x_plain
    for _ in range(depth):
        # Weight with singular values ~ 0.6
        w = torch.randn(16, 16) * 0.15
        curr = torch.matmul(curr, w)

    loss_plain = curr.sum()
    loss_plain.backward()
    plain_norm = float(torch.linalg.norm(x_plain.grad).item())

    # 2. Residual chain: x_{l+1} = x_l + F(x_l)
    x_res = torch.ones(1, 16, requires_grad=True)
    curr = x_res
    for _ in range(depth):
        w = torch.randn(16, 16) * 0.15
        curr = curr + torch.matmul(curr, w)

    loss_res = curr.sum()
    loss_res.backward()
    res_norm = float(torch.linalg.norm(x_res.grad).item())

    return plain_norm, res_norm


def main() -> None:
    print("=" * 72)
    print("Lesson 05 Benchmark: Gradients, Loss Surfaces, and Backpropagation")
    print("=" * 72)

    print("\n1. Analytical Gradient vs. PyTorch Autograd Equivalence:")
    print("-" * 72)
    x_val, y_val = 3.5, -2.0
    a, b = 1.0, 4.0
    loss_ana, (gx_ana, gy_ana) = quadratic_loss_and_analytical_grad(x_val, y_val, a, b)
    loss_auto, (gx_auto, gy_auto) = compute_autograd(x_val, y_val, a, b)

    diff_loss = abs(loss_ana - loss_auto)
    diff_gx = abs(gx_ana - gx_auto)
    diff_gy = abs(gy_ana - gy_auto)

    print(f"Position (x, y): ({x_val}, {y_val}) on surface L(x, y) = {a}x^2 + {b}y^2")
    print(f"Analytical: Loss = {loss_ana:.4f}, Grad = ({gx_ana:.4f}, {gy_ana:.4f})")
    print(f"Autograd:   Loss = {loss_auto:.4f}, Grad = ({gx_auto:.4f}, {gy_auto:.4f})")
    print(f"Max Gradient Error: {max(diff_gx, diff_gy):.1e} (Reverse-mode exactness)")

    print("\n2. Descent Trajectory Down the Anisotropic Valley (lr=0.1):")
    print("-" * 72)
    print(f"{'Step':<6} | {'x':<10} | {'y':<10} | {'Loss L(x, y)':<14} | {'Convergence Bar'}")
    print("-" * 72)
    traj = run_gradient_descent(start_pos=(5.0, 3.0), lr=0.1, steps=8, a=1.0, b=4.0)
    max_loss = traj[0][3]
    for step, x, y, loss in traj:
        bar_len = max(1, int((loss / max_loss) * 30))
        bar = "█" * bar_len
        print(f"{step:<6} | {x:<10.4f} | {y:<10.4f} | {loss:<14.4f} | {bar}")
    print("-" * 72)

    print("\n3. The Gradient Highway: Vanishing Gradient vs. Residual Connection:")
    print("-" * 72)
    plain_norm, res_norm = measure_residual_gradient_highway(depth=10)
    print(f"Chain Depth: 10 layers")
    print(f"Plain Sequential Input Gradient Norm:    {plain_norm:.6f} (Severely vanished!)")
    print(f"Residual Network (ResNet/Transformer): {res_norm:.6f} (Gradient preserved!)")
    ratio = res_norm / max(plain_norm, 1e-12)
    print(f"Gradient Highway Preservation Factor:    {ratio:.1f}x")
    print("=" * 72)


if __name__ == "__main__":
    main()
