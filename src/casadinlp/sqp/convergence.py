"""
KKT residual computation and convergence check for the SQP solver.

Convergence is declared when both:

    ‖∇f(x,p) + J_g(x,p)ᵀ λ‖_∞  ≤  ε_stat   (stationarity)
    max(max(g-rhs, 0), max(lhs-g, 0))_∞      ≤  ε_feas   (primal feasibility)
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from casadinlp.sqp.schema import SQPConfig


def kkt_residuals(
    x: jnp.ndarray,
    lam: jnp.ndarray,
    p: jnp.ndarray,
    objective: Callable,
    constraints: Optional[Callable],
    constraint_lhs: Optional[jnp.ndarray],
    constraint_rhs: Optional[jnp.ndarray],
    n_g: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute KKT stationarity and primal feasibility residuals.

    Parameters
    ----------
    x:
        Current iterate, shape ``(n,)``.
    lam:
        Dual variables for general constraints, shape ``(n_g,)``.
    p:
        Parameter vector, shape ``(m,)``.
    objective:
        ``f(x, p) -> scalar``.
    constraints:
        ``g(x, p) -> (n_g,)`` or ``None``.
    constraint_lhs, constraint_rhs:
        Constraint bounds, each of shape ``(n_g,)`` or ``None``.
    n_g:
        Number of general constraints.

    Returns
    -------
    stationarity : jnp.ndarray
        Scalar ``‖∇_x L(x, λ, p)‖_∞``.
    feasibility : jnp.ndarray
        Scalar constraint violation in the L∞ norm.
    """
    grad_f = jax.grad(objective)(x, p)

    if n_g > 0 and constraints is not None:
        g_val = constraints(x, p).reshape(n_g)
        jac_g = jax.jacfwd(constraints)(x, p).reshape(n_g, x.shape[0])
        grad_lag = grad_f + jac_g.T @ lam

        rhs = jnp.asarray(constraint_rhs).reshape(n_g)
        lhs = jnp.asarray(constraint_lhs).reshape(n_g)
        feas = jnp.maximum(
            jnp.max(jnp.maximum(g_val - rhs, 0.0)),
            jnp.max(jnp.maximum(lhs - g_val, 0.0)),
        )
    else:
        grad_lag = grad_f
        feas = jnp.zeros(())

    stationarity = jnp.max(jnp.abs(grad_lag))
    return stationarity, feas


def is_converged(
    stationarity: jnp.ndarray,
    feasibility: jnp.ndarray,
    cfg: SQPConfig,
) -> jnp.ndarray:
    """Return ``True`` iff both KKT tolerances are satisfied."""
    return (stationarity <= cfg.tol_stationarity) & (feasibility <= cfg.tol_feasibility)
