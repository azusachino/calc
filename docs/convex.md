# Convex optimization, KKT duality, and ADMM Lasso

This document accompanies `calc/math/convex.py` (paired with **Stanford EE364a: Convex Optimization**) and verifies strong Lagrangian duality, KKT subgradient stationarity, and Alternating Direction Method of Multipliers (ADMM) operator splitting.

Run the companion benchmark:

```bash
uv run calc/math/convex.py
```

---

## 1. Strong duality in Ridge quadratic programming

For strictly convex Ridge regression:

$$p^\star = \min_{x \in \mathbb{R}^n} \frac{1}{2}\|Ax - b\|_2^2 + \frac{\lambda}{2}\|x\|_2^2$$

introducing residual $r = Ax - b$ with dual multiplier $\nu \in \mathbb{R}^m$ yields the closed-form concave dual:

$$d^\star = \max_{\nu \in \mathbb{R}^m} -\frac{1}{2}\|\nu\|_2^2 - \frac{1}{2\lambda}\|A^\top \nu\|_2^2 + b^\top \nu$$

Because Slater's condition holds trivially, strong duality guarantees zero duality gap ($p^\star - d^\star = 0$) at $\nu^\star = b - A x^\star$.

---

## 2. Lasso regression via ADMM and soft thresholding

To solve the non-differentiable $\ell_1$-regularized problem:

$$\min_{x, z} \frac{1}{2}\|Ax - b\|_2^2 + \lambda \|z\|_1 \quad \text{subject to} \quad x - z = 0$$

scaled ADMM alternates three closed-form updates with penalty $\rho > 0$:

$$x^{k+1} = (A^\top A + \rho I)^{-1}\!\left(A^\top b + \rho(z^k - u^k)\right)$$

$$z^{k+1} = S_{\lambda/\rho}(x^{k+1} + u^k) = \operatorname{sign}(v)\max(|v| - \lambda/\rho, 0)$$

$$u^{k+1} = u^k + x^{k+1} - z^{k+1}$$

At convergence, $z^\star$ satisfies the exact KKT subgradient optimality condition $A^\top(b - Az^\star) \in \lambda \partial \|z^\star\|_1$.
