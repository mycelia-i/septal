"""
L1 exact penalty merit function and backtracking line search.

Merit function
--------------
    φ(x; ρ) = f(x, p) + ρ · Σᵢ [max(gᵢ(x,p) - rhsᵢ, 0) + max(lhsᵢ - gᵢ(x,p), 0)]

This is the standard L1 exact penalty.  For ρ > ‖λ*‖_∞ the merit function
is exact (local minima of the NLP are local minima of the merit function).

Directional derivative estimate (Nocedal & Wright §18.3)
---------------------------------------------------------
For the SQP search direction d_k:

    D_φ(x_k; d_k) ≈ ∇f_kᵀ d_k  −  ρ · v(x_k)

where v(x_k) = Σᵢ max violations.  This is the standard sufficient descent
condition used to check acceptability of the penalty parameter.

Line search
-----------
Armijo backtracking via ``jax.lax.while_loop`` (JIT- and vmap-compatible):

    α ← α · β  until  φ(x + α·d) ≤ φ(x) + c·α·D_φ
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp

from septal.jax.sqp.schema import SQPConfig


# ---------------------------------------------------------------------------
# L1 merit helpers
# ---------------------------------------------------------------------------


def constraint_violation(
    g_val: jnp.ndarray,
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
) -> jnp.ndarray:
    """L1 constraint violation: Σᵢ max(gᵢ-rhsᵢ, 0) + max(lhsᵢ-gᵢ, 0)."""
    upper_viol = jnp.sum(jnp.maximum(g_val - rhs, 0.0))
    lower_viol = jnp.sum(jnp.maximum(lhs - g_val, 0.0))
    return upper_viol + lower_viol


def l1_merit(
    f_val: jnp.ndarray,
    g_val: jnp.ndarray,
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    penalty: jnp.ndarray,
    n_g: int,
) -> jnp.ndarray:
    """Evaluate the L1 exact penalty merit function φ(x; ρ).

    Parameters
    ----------
    f_val:
        Objective value ``f(x, p)``, scalar.
    g_val:
        Constraint value ``g(x, p)``, shape ``(n_g,)``.
        Pass ``jnp.zeros(0)`` when there are no general constraints.
    lhs, rhs:
        Constraint bounds, each of shape ``(n_g,)``.
    penalty:
        Penalty parameter ``ρ``, scalar.
    n_g:
        Number of general constraints.

    Returns
    -------
    jnp.ndarray
        Scalar merit value.
    """
    if n_g > 0:
        viol = constraint_violation(g_val, lhs, rhs)
        return f_val + penalty * viol
    return f_val


def merit_directional_deriv(
    grad_f: jnp.ndarray,
    d: jnp.ndarray,
    g_val: jnp.ndarray,
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    penalty: jnp.ndarray,
    n_g: int,
) -> jnp.ndarray:
    """Estimate the directional derivative of the L1 merit along the SQP step.

    Uses the standard SQP approximation:
        D_φ(x; d) ≈ ∇f(x)ᵀ d  −  ρ · v(x)

    This is negative (descent) when ρ ≥ ‖λ‖_∞.
    """
    gd = grad_f @ d
    if n_g > 0:
        viol = constraint_violation(g_val, lhs, rhs)
        return gd - penalty * viol
    return gd


def update_penalty(
    lam: jnp.ndarray,
    penalty: jnp.ndarray,
    n_g: int,
    eps: float,
) -> jnp.ndarray:
    """Non-decreasing penalty update: ρ_{k+1} = max(‖λ_k‖_∞ + ε, ρ_k).

    Ensures the penalty parameter is large enough for the merit function
    to be an exact penalty (i.e. local NLP optima are merit function optima).
    """
    if n_g > 0:
        lam_inf = jnp.max(jnp.abs(lam))
        return jnp.maximum(lam_inf + eps, penalty)
    return penalty


# ---------------------------------------------------------------------------
# Backtracking line search
# ---------------------------------------------------------------------------


def backtracking_line_search(
    x: jnp.ndarray,
    d: jnp.ndarray,
    p: jnp.ndarray,
    current_merit: jnp.ndarray,
    dir_deriv: jnp.ndarray,
    penalty: jnp.ndarray,
    objective: Callable,
    constraints: Optional[Callable],
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    n_g: int,
    cfg: SQPConfig,
) -> jnp.ndarray:
    """Armijo backtracking line search on the L1 merit function.

    Starts with ``α = cfg.line_search_alpha0`` and multiplies by
    ``cfg.line_search_beta`` until

        φ(x + α·d; ρ)  ≤  φ(x; ρ) + c · α · D_φ(x; d)

    or until ``cfg.max_line_search`` trials are exhausted.

    Parameters
    ----------
    x:
        Current iterate, shape ``(n,)``.
    d:
        Search direction, shape ``(n,)``.
    p:
        Parameter vector, shape ``(m,)``.
    current_merit:
        Merit value at ``x``, scalar.
    dir_deriv:
        Directional derivative estimate at ``x`` along ``d``, scalar.
    penalty:
        Penalty parameter ``ρ``, scalar.
    objective:
        ``f(x, p) -> scalar``.
    constraints:
        ``g(x, p) -> (n_g,)`` or ``None``.
    lhs, rhs:
        Constraint bounds.
    n_g:
        Number of general constraints.
    cfg:
        Solver configuration.

    Returns
    -------
    jnp.ndarray
        Accepted step length ``α``, scalar.
    """
    beta = cfg.line_search_beta
    c = cfg.line_search_c
    alpha0 = jnp.array(cfg.line_search_alpha0, dtype=x.dtype)

    def _merit_at(alpha: jnp.ndarray) -> jnp.ndarray:
        x_new = x + alpha * d
        f_new = objective(x_new, p)
        if n_g > 0 and constraints is not None:
            g_new = constraints(x_new, p)
            return l1_merit(f_new, g_new, lhs, rhs, penalty, n_g)
        return jnp.asarray(f_new)

    def cond_fn(state: tuple) -> jnp.ndarray:
        alpha, i = state
        trial_merit = _merit_at(alpha)
        armijo = trial_merit <= current_merit + c * alpha * dir_deriv
        # Continue if Armijo fails AND we have more trials
        return (~armijo) & (i < cfg.max_line_search)

    def body_fn(state: tuple) -> tuple:
        alpha, i = state
        return (alpha * beta, i + 1)

    alpha_final, _ = jax.lax.while_loop(cond_fn, body_fn, (alpha0, 0))
    return alpha_final
