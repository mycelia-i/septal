"""
Damped BFGS Hessian approximation (Powell, 1978).

The standard BFGS update is:

    H_{k+1} = H_k - (H_k s_k s_kᵀ H_k) / (s_kᵀ H_k s_k)
                  + (r_k r_kᵀ) / (s_kᵀ r_k)

where s_k = x_{k+1} - x_k and r_k is the (possibly damped) curvature vector.

Powell damping modifies y_k so that the updated Hessian remains positive
definite, which is required for the QP subproblem to be convex:

    θ_k = 1                         if  s_kᵀ y_k >= 0.2 · s_kᵀ H_k s_k
          0.8 s_kᵀ H_k s_k          otherwise
          ─────────────────────
          s_kᵀ H_k s_k - s_kᵀ y_k

    r_k = θ_k y_k + (1 - θ_k) H_k s_k

All operations are pure JAX — no Python control flow on traced values.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def bfgs_update(
    H: jnp.ndarray,
    s: jnp.ndarray,
    y: jnp.ndarray,
    skip_tol: float = 1e-10,
) -> jnp.ndarray:
    """Damped BFGS rank-2 Hessian update.

    Parameters
    ----------
    H:
        Current Hessian approximation, shape ``(n, n)``.  Must be
        symmetric positive definite.
    s:
        Primal step ``x_{k+1} - x_k``, shape ``(n,)``.
    y:
        Lagrangian gradient difference ``∇L_{k+1} - ∇L_k``, shape ``(n,)``.
    skip_tol:
        If ``sᵀs < skip_tol`` the update is skipped (step was numerically zero).

    Returns
    -------
    jnp.ndarray
        Updated Hessian, shape ``(n, n)``.
    """
    Hs = H @ s
    sHs = s @ Hs
    sy = s @ y

    # Powell damping: blend y towards H@s so that sᵀr > 0
    theta = jnp.where(sy >= 0.2 * sHs, 1.0, 0.8 * sHs / (sHs - sy + 1e-30))
    r = theta * y + (1.0 - theta) * Hs
    sr = s @ r

    # BFGS rank-2 update
    H_new = (
        H
        - jnp.outer(Hs, Hs) / (sHs + 1e-30)
        + jnp.outer(r, r) / (sr + 1e-30)
    )

    # Skip update if the step was too small (avoids numerical blow-up)
    skip = (s @ s) < skip_tol
    H_new = jnp.where(skip, H, H_new)

    # Symmetrise to counteract floating-point drift
    H_new = 0.5 * (H_new + H_new.T)

    return H_new


def lagrangian_grad(
    x: jnp.ndarray,
    lam: jnp.ndarray,
    p: jnp.ndarray,
    objective: callable,
    constraints: callable,
    n_g: int,
) -> jnp.ndarray:
    """Compute the Lagrangian gradient ∇_x L(x, λ, p).

    Parameters
    ----------
    x:
        Current iterate, shape ``(n,)``.
    lam:
        Dual variables for general constraints, shape ``(n_g,)``.
    p:
        Parameter vector, shape ``(m,)``.
    objective:
        JAX callable ``f(x, p) -> scalar``.
    constraints:
        JAX callable ``g(x, p) -> (n_g,)`` or ``None``.
    n_g:
        Number of general constraints.

    Returns
    -------
    jnp.ndarray, shape ``(n,)``
        ``∇f(x, p) + J_g(x, p)ᵀ λ``.
    """
    grad_f = jax.grad(objective)(x, p)
    if n_g > 0 and constraints is not None:
        jac_g = jax.jacfwd(constraints)(x, p)  # (n_g, n)
        return grad_f + jac_g.T @ lam
    return grad_f
