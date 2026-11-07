"""Reverse-mode automatic differentiation and strided NDArray memory views.

Companion lab for CMU 10-414 / 10-714 (Deep Learning Systems):
1. Zero-copy strided tensor views (shape, strides, offset) for permute, slice, and broadcast.
2. Reverse-mode automatic differentiation over a dynamic DAG with broadcast gradient reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class StridedView:
    """Zero-copy view metadata over a flat 1D storage buffer."""

    data: tuple[float, ...]
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    offset: int = 0

    @staticmethod
    def contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
        """Compute row-major (C-contiguous) strides for a given shape."""
        if not shape:
            return ()
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)

    @classmethod
    def from_flat(cls, data: list[float] | tuple[float, ...], shape: tuple[int, ...]) -> StridedView:
        expected = math.prod(shape) if shape else 1
        if len(data) != expected:
            raise ValueError(f"Data length {len(data)} does not match shape {shape}")
        return cls(
            data=tuple(float(x) for x in data),
            shape=shape,
            strides=cls.contiguous_strides(shape),
            offset=0,
        )

    def flat_index(self, indices: tuple[int, ...]) -> int:
        """Translate multidimensional index (i_0, ..., i_{k-1}) into flat storage offset."""
        if len(indices) != len(self.shape):
            raise IndexError(f"Expected {len(self.shape)} indices, got {len(indices)}")
        addr = self.offset
        for idx, dim, stride in zip(indices, self.shape, self.strides):
            if idx < 0 or idx >= dim:
                raise IndexError(f"Index {idx} out of bounds for dimension {dim}")
            addr += idx * stride
        return addr

    def get(self, indices: tuple[int, ...]) -> float:
        return self.data[self.flat_index(indices)]

    def is_compact(self) -> bool:
        """Check whether the view is contiguous in row-major order with zero offset."""
        return self.offset == 0 and self.strides == self.contiguous_strides(self.shape)

    def permute(self, axes: tuple[int, ...]) -> StridedView:
        """Zero-copy axis permutation by reordering shape and strides."""
        if sorted(axes) != list(range(len(self.shape))):
            raise ValueError(f"Invalid permutation axes {axes} for shape {self.shape}")
        new_shape = tuple(self.shape[a] for a in axes)
        new_strides = tuple(self.strides[a] for a in axes)
        return StridedView(
            data=self.data,
            shape=new_shape,
            strides=new_strides,
            offset=self.offset,
        )

    def broadcast_to(self, target_shape: tuple[int, ...]) -> StridedView:
        """Zero-copy broadcast by setting stride=0 along singleton dimensions."""
        if len(target_shape) < len(self.shape):
            raise ValueError(f"Cannot broadcast {self.shape} to fewer dims {target_shape}")
        pad = len(target_shape) - len(self.shape)
        padded_shape = (1,) * pad + self.shape
        padded_strides = (0,) * pad + self.strides

        new_strides: list[int] = []
        for s_dim, s_stride, t_dim in zip(padded_shape, padded_strides, target_shape):
            if s_dim == t_dim:
                new_strides.append(s_stride)
            elif s_dim == 1:
                new_strides.append(0)
            else:
                raise ValueError(f"Incompatible broadcast dimension {s_dim} -> {t_dim}")
        return StridedView(
            data=self.data,
            shape=target_shape,
            strides=tuple(new_strides),
            offset=self.offset,
        )

    def compact(self) -> StridedView:
        """Materialize a non-contiguous or broadcasted view into fresh contiguous storage."""
        if not self.shape:
            return StridedView(data=(self.data[self.offset],), shape=(), strides=(), offset=0)

        out: list[float] = []

        def _recurse(prefix: tuple[int, ...], depth: int) -> None:
            if depth == len(self.shape):
                out.append(self.get(prefix))
                return
            for i in range(self.shape[depth]):
                _recurse(prefix + (i,), depth + 1)

        _recurse((), 0)
        return StridedView.from_flat(out, self.shape)


def sum_to_shape(grad: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    """Adjoint of broadcast_to: sum out prepended and singleton broadcasted axes."""
    res = grad
    while res.ndim > len(target_shape):
        res = res.sum(dim=0)
    for axis, (g_dim, t_dim) in enumerate(zip(res.shape, target_shape)):
        if t_dim == 1 and g_dim > 1:
            res = res.sum(dim=axis, keepdim=True)
    return res


def reverse_mode_linear_gelu_step(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> tuple[float, dict[str, torch.Tensor], dict[str, float]]:
    """Compare manual reverse-mode VJP adjoints against PyTorch autograd on L = sum((x @ w + b)^2).

    Here b has shape (1, D_out) and broadcasts across batch dimension B, requiring
    sum_to_shape in the backward pass.

    Returns:
        (loss_value, manual_grads, max_abs_errors)
    """
    # Forward pass
    z = x @ w  # (B, D_out)
    h = z + b  # b (1, D_out) broadcasts to (B, D_out)
    loss = torch.sum(h * h)

    # Manual reverse-mode AD (backward pass)
    v_loss = torch.ones_like(loss)
    v_h = v_loss * (2.0 * h)  # dL/dh = 2 * h
    v_z = v_h
    v_b = sum_to_shape(v_h, tuple(b.shape))
    v_x = v_z @ w.transpose(0, 1)
    v_w = x.transpose(0, 1) @ v_z

    # Reference PyTorch autograd
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    loss_ref = torch.sum((x_ref @ w_ref + b_ref) ** 2)
    loss_ref.backward()

    assert x_ref.grad is not None and w_ref.grad is not None and b_ref.grad is not None
    errors = {
        "x": float(torch.max(torch.abs(v_x - x_ref.grad)).item()),
        "w": float(torch.max(torch.abs(v_w - w_ref.grad)).item()),
        "b": float(torch.max(torch.abs(v_b - b_ref.grad)).item()),
    }
    return float(loss.item()), {"x": v_x, "w": v_w, "b": v_b}, errors


def main() -> None:
    print("=" * 72)
    print("CMU 10-414/10-714 Lab: Strided NDArray Views & Reverse-Mode Autograd")
    print("=" * 72)

    view = StridedView.from_flat(list(range(6)), (2, 3))
    transposed = view.permute((1, 0))
    print(f"Original shape={view.shape}, strides={view.strides}, compact={view.is_compact()}")
    print(
        f"Permuted shape={transposed.shape}, strides={transposed.strides}, "
        f"compact={transposed.is_compact()}"
    )

    col = StridedView.from_flat([10.0, 20.0], (2, 1))
    bcast = col.broadcast_to((2, 4))
    print(f"Broadcast (2, 1) -> (2, 4) strides={bcast.strides}, value[1, 3]={bcast.get((1, 3))}")

    gen = torch.Generator().manual_seed(42)
    x = torch.randn(4, 5, generator=gen)
    w = torch.randn(5, 3, generator=gen)
    b = torch.randn(1, 3, generator=gen)
    loss_val, _, errs = reverse_mode_linear_gelu_step(x, w, b)
    print(f"\nReverse-Mode VJP Loss: {loss_val:.4f} | Max Autograd Diff: {max(errs.values()):.2e}")
    print("=" * 72)


if __name__ == "__main__":
    main()
