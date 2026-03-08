"""
QP subproblem formation and solution for the SQP outer loop.

At SQP iterate x_k the subproblem is:

    min_{d}   ½ dᵀ H_k d + ∇f_kᵀ d
    s.t.      lhs - g_k  ≤  J_k d  ≤  rhs - g_k     (linearised constraints)
              lb  - x_k  ≤    d    ≤  ub  - x_k      (box constraints on step)

The two-sided constraints  l ≤ Cd ≤ u  are converted to  Gd ≤ h:

    G = [ C ]     h = [ u_safe ]
        [-C ]         [-l_safe ]

where l_safe / u_safe replace ±inf with ±INF_VAL.

ADMM QP solver (pure JAX, fully vmappable)
------------------------------------------
We solve  min ½ dᵀ Q d + cᵀ d  s.t.  G d ≤ h  via scaled ADMM:

    x-update: (Q + ρ GᵀG) x = ρ Gᵀ(z - u) - c
    z-update: z = min(Gx + u, h)    (projection onto z ≤ h)
    u-update: u = u + Gx - z

at convergence:  λ = ρ · u  is the Lagrange multiplier for G d ≤ h.

The ``lax.scan`` over a fixed number of iterations makes this fully
vmappable and JIT-compilable with no Python control-flow on traced values.
jaxopt.OSQP is NOT used here because it has a known vmap bug for certain
batch sizes.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from casadinlp.sqp.schema import SQPConfig

_INF_VAL = 1e10   # stand-in for ±∞ in ADMM bounds


def _safe_bounds(l: jnp.ndarray, u: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Replace ±inf with ±_INF_VAL so all ADMM arithmetic is finite."""
    l_s = jnp.where(jnp.isinf(l) & (l < 0), -_INF_VAL, l)
    u_s = jnp.where(jnp.isinf(u) & (u > 0), _INF_VAL, u)
    return l_s, u_s


def form_qp_matrices(
    hessian: jnp.ndarray,
    grad_f: jnp.ndarray,
    jac_g: jnp.ndarray,
    g_val: jnp.ndarray,
    x: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    n: int,
    n_g: int,
    reg: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build ADMM QP matrices ``(Q, c, G, h)`` for  min ½dᵀQd+cᵀd, Gd≤h.

    Parameters
    ----------
    hessian : (n, n)  BFGS Hessian approximation.
    grad_f  : (n,)    Objective gradient ∇f(x_k, p).
    jac_g   : (n_g, n) Constraint Jacobian.  Pass zeros((0, n)) if n_g=0.
    g_val   : (n_g,)  Constraint value.  Pass zeros(0) if n_g=0.
    x       : (n,)    Current iterate.
    lb, ub  : (n,)    Decision-variable bounds.
    lhs,rhs : (n_g,)  Constraint bounds.
    n, n_g  : problem dimensions.
    reg     : small diagonal regularisation on Q.

    Returns
    -------
    Q : (n, n),  c : (n,),  G : (2*(n_g+n), n),  h : (2*(n_g+n),)
    """
    Q = 0.5 * (hessian + hessian.T) + reg * jnp.eye(n)
    c = grad_f.reshape(n)

    if n_g > 0:
        l_gc = lhs.reshape(n_g) - g_val.reshape(n_g)
        u_gc = rhs.reshape(n_g) - g_val.reshape(n_g)
        l_gc, u_gc = _safe_bounds(l_gc, u_gc)
        C = jnp.concatenate([jac_g.reshape(n_g, n), jnp.eye(n)], axis=0)
        l_all = jnp.concatenate([l_gc, lb - x])
        u_all = jnp.concatenate([u_gc, ub - x])
    else:
        C = jnp.eye(n)
        l_all = lb - x
        u_all = ub - x

    l_all, u_all = _safe_bounds(l_all, u_all)

    # Two-sided → one-sided
    G = jnp.concatenate([C, -C], axis=0)
    h = jnp.concatenate([u_all, -l_all], axis=0)

    return Q, c, G, h


def admm_qp(
    Q: jnp.ndarray,
    c: jnp.ndarray,
    G: jnp.ndarray,
    h: jnp.ndarray,
    rho: float,
    n_iter: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Solve  min ½ dᵀQd + cᵀd  s.t.  Gd ≤ h  via ADMM.

    Fully vmappable and JIT-compilable (uses ``lax.scan`` with fixed
    ``n_iter`` iterations, no Python control flow on traced values).

    Parameters
    ----------
    Q    : (n, n)  Symmetric PSD cost matrix.
    c    : (n,)    Linear cost.
    G    : (m, n)  Inequality constraint matrix.
    h    : (m,)    Inequality right-hand side.
    rho  : float   ADMM penalty parameter.
    n_iter : int   Fixed number of ADMM iterations.

    Returns
    -------
    d   : (n,)  Primal solution.
    lam : (m,)  Dual variables for  Gd ≤ h  (λ = ρ u at convergence).
    """
    n = Q.shape[0]
    m = G.shape[0]

    # Precompute M = Q + ρ GᵀG (constant across ADMM iterations)
    M = Q + rho * (G.T @ G)

    x0 = jnp.zeros(n)
    z0 = jnp.zeros(m)
    u0 = jnp.zeros(m)

    def admm_step(state: tuple, _: None) -> Tuple[tuple, None]:
        x, z, u = state
        # x-update: M x = ρ Gᵀ(z - u) - c
        rhs_x = rho * (G.T @ (z - u)) - c
        x_new = jnp.linalg.solve(M, rhs_x)
        # z-update: projection onto Gd ≤ h
        Gx = G @ x_new
        z_new = jnp.minimum(Gx + u, h)
        # u-update (scaled dual)
        u_new = u + Gx - z_new
        return (x_new, z_new, u_new), None

    (x_final, _, u_final), _ = jax.lax.scan(
        admm_step, (x0, z0, u0), None, length=n_iter
    )

    # Lagrange multipliers for G d ≤ h
    lam = rho * u_final
    return x_final, lam


def solve_qp_subproblem(
    hessian: jnp.ndarray,
    grad_f: jnp.ndarray,
    jac_g: jnp.ndarray,
    g_val: jnp.ndarray,
    x: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    n: int,
    n_g: int,
    cfg: SQPConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Solve the SQP QP subproblem via pure-JAX ADMM.

    Fully vmappable (no jaxopt dependency).

    Returns
    -------
    d     : (n,)    Primal search direction.
    lam_g : (n_g,)  Dual variables for the general constraints.
                    Positive when upper bound active, negative when lower.
    """
    Q, c, G, h = form_qp_matrices(
        hessian, grad_f, jac_g, g_val, x, lb, ub, lhs, rhs, n, n_g, cfg.osqp_reg
    )

    d, lam_all = admm_qp(Q, c, G, h, rho=cfg.admm_rho, n_iter=cfg.admm_n_iter)

    # Extract general-constraint multipliers.
    # G = [C; -C], dual lam_all = [λ_upper; λ_lower]  where each block is (n_g+n,).
    # Effective multiplier for l ≤ Cd ≤ u: lam_g = λ_upper[:n_g] - λ_lower[:n_g]
    m_half = n_g + n  # size of one block
    if n_g > 0:
        lam_g = lam_all[:n_g] - lam_all[m_half: m_half + n_g]
    else:
        lam_g = jnp.zeros(0)

    # Guard against NaN/Inf from a numerically ill-conditioned QP
    d = jnp.where(jnp.isfinite(d).all(), d, jnp.zeros(n))
    lam_g = jnp.where(jnp.isfinite(lam_g).all(), lam_g, jnp.zeros(n_g))

    return d, lam_g
