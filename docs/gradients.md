# Gradients, loss landscapes, and backpropagation

This document explores the calculus foundations of neural network training: how scalar derivatives generalize to multidimensional gradients, how reverse-mode automatic differentiation computes exact gradients via directed computational graphs, and why residual connections create a "gradient highway".

Run the companion benchmark:

```bash
uv run calc/math/gradients.py
```

---

## 1. From 1D slopes to high-dimensional gradients

In single-variable calculus, the derivative $f'(x)$ is the instantaneous rate of change:

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

In a neural network with billions of parameters $\mathbf{w} \in \mathbb{R}^D$, the loss function $\mathcal{L}(\mathbf{w})$ is a scalar landscape over high-dimensional space.
The **gradient vector** $\nabla \mathcal{L}$ collects the partial derivatives with respect to every single parameter:

$$\nabla \mathcal{L}(\mathbf{w}) = \begin{pmatrix} \frac{\partial \mathcal{L}}{\partial w_1} \\ \frac{\partial \mathcal{L}}{\partial w_2} \\ \vdots \\ \frac{\partial \mathcal{L}}{\partial w_D} \end{pmatrix}$$

### Geometric interpretation
- $\nabla \mathcal{L}(\mathbf{w})$ points in the direction of **steepest ascent** (the sharpest uphill incline).
- The magnitude $\|\nabla \mathcal{L}\|$ is the slope in that steepest direction.
- In **Gradient Descent**, we step in the exact opposite direction:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla \mathcal{L}(\mathbf{w}_t)$$

where $\eta > 0$ is the learning rate.

---

## 2. Reverse-mode automatic differentiation (Backprop)

Modern deep learning does **not** use numerical differentiation (finite differences $\frac{\mathcal{L}(w + \epsilon) - \mathcal{L}(w)}{\epsilon}$ requires $D$ full forward passes, impossible for billions of weights).
Nor does it use symbolic algebraic differentiation (which causes combinatorial expression explosion).

Instead, it evaluates gradients via **Reverse-Mode Automatic Differentiation** on a Directed Acyclic Graph (DAG):
1. **Forward Pass**: Compute intermediate node values from inputs to loss $\mathcal{L}$, storing activations in memory.
2. **Backward Pass**: Traverse the graph in reverse from loss $\mathcal{L}$ to leaf parameters, applying the multivariate Chain Rule:

$$\frac{\partial \mathcal{L}}{\partial x} = \sum_{y \in \text{children}(x)} \frac{\partial \mathcal{L}}{\partial y} \frac{\partial y}{\partial x}$$

Because the backward pass reuses stored forward activations, **all partial derivatives across all $D$ parameters are computed in a single reverse sweep**, costing only $\approx 2\times$ the forward pass compute!

---

## 3. The gradient highway: why residual connections matter

In a deep feedforward network of depth $L$:

$$x_{l+1} = W_l x_l$$

The gradient flowing from the loss to the input is a product of weight matrices:

$$\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} \prod_{l=0}^{L-1} W_l$$

If the singular values of $W_l$ are even slightly less than 1 (e.g., $\sigma \approx 0.8$), after 20 layers $\sigma^{20} \approx 0.01$, and gradients vanish into floating-point underflow.

### The residual solution: $x_{l+1} = x_l + F(x_l)$

$$\frac{\partial x_{l+1}}{\partial x_l} = I + \frac{\partial F(x_l)}{\partial x_l}$$

By expanding across layers:

$$\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} \left( I + \sum \dots \right)$$

The identity matrix $I$ provides an unattenuated, direct path for gradients to propagate straight from the loss to earlier layers without exponential decay. This is the mathematical reason why 100-layer Transformers can be trained stably.
