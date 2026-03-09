"""
Data structures for the JAX-based SQP solver.

Parametric NLP standard form
-----------------------------
    min_{x}  f(x, p)
    s.t.     lhs <= g(x, p) <= rhs
             lb  <=  x      <= ub

where p is a fixed parameter vector.  The batch use-case solves the same
structural problem for N different values of p simultaneously via jax.vmap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, NamedTuple

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------


@dataclass
class ParametricNLPProblem:
    """Parametric NLP: min f(x, p) s.t. lhs <= g(x, p) <= rhs, lb <= x <= ub.

    Parameters
    ----------
    objective:
        JAX function ``f(x, p) -> scalar``.
    bounds:
        ``[lb, ub]`` — list/tuple of two JAX/NumPy arrays, each of shape
        ``(n_decision,)``.
    n_decision:
        Dimension of ``x``.
    n_params:
        Dimension of ``p``.  Use 0 for parameter-free problems.
    constraints:
        Optional JAX function ``g(x, p) -> (n_constraints,)``.
    constraint_lhs:
        Lower bound on ``g(x, p)``; shape ``(n_constraints,)``.
        Use ``-jnp.inf`` entries for one-sided upper bounds.
    constraint_rhs:
        Upper bound on ``g(x, p)``; shape ``(n_constraints,)``.
        Typically ``0`` for constraints of the form ``g(x, p) <= 0``.
    n_constraints:
        Number of general constraints.  Inferred from *constraint_lhs* /
        *constraint_rhs* if not supplied.
    """

    objective: Callable
    bounds: List[Any]
    n_decision: int
    n_params: int
    constraints: Optional[Callable] = None
    constraint_lhs: Optional[Any] = None
    constraint_rhs: Optional[Any] = None
    n_constraints: Optional[int] = None

    def __post_init__(self) -> None:
        if self.n_constraints is None:
            if self.constraint_lhs is not None:
                self.n_constraints = len(jnp.asarray(self.constraint_lhs).reshape(-1))
            elif self.constraint_rhs is not None:
                self.n_constraints = len(jnp.asarray(self.constraint_rhs).reshape(-1))
            else:
                self.n_constraints = 0

    @property
    def lb(self) -> Any:
        return self.bounds[0]

    @property
    def ub(self) -> Any:
        return self.bounds[1]

    @property
    def has_constraints(self) -> bool:
        return self.constraints is not None and self.n_constraints > 0


# ---------------------------------------------------------------------------
# Solver configuration
# ---------------------------------------------------------------------------


@dataclass
class SQPConfig:
    """Hyper-parameters for the JAX SQP solver.

    Parameters
    ----------
    max_iter:
        Maximum number of SQP outer iterations.
    tol_stationarity:
        KKT stationarity convergence tolerance.
    tol_feasibility:
        KKT primal feasibility convergence tolerance.
    line_search_beta:
        Backtracking factor (0 < beta < 1).  Step is multiplied by *beta*
        at each rejected trial.
    line_search_c:
        Armijo sufficient-decrease constant (typically 1e-4).
    line_search_alpha0:
        Initial trial step length.
    max_line_search:
        Maximum backtracking iterations.
    penalty_eps:
        Additive margin above ``‖λ‖_∞`` when updating the penalty parameter.
    penalty_init:
        Initial penalty parameter ``ρ₀``.
    bfgs_init_scale:
        Diagonal scale of the initial Hessian approximation ``H₀ = scale·I``.
    bfgs_skip_tol:
        Skip BFGS update when ``|sᵀy| < tol * ‖s‖²`` (prevents rank-deficiency).
    osqp_tol:
        OSQP convergence tolerance (combined primal/dual).
    osqp_max_iter:
        Maximum OSQP iterations per QP subproblem.
    osqp_reg:
        Small regularisation added to QP Hessian diagonal for PSD guarantee.
    """

    max_iter: int = 100
    tol_stationarity: float = 1e-6
    tol_feasibility: float = 1e-6
    line_search_beta: float = 0.5
    line_search_c: float = 1e-4
    line_search_alpha0: float = 1.0
    max_line_search: int = 30
    penalty_eps: float = 1e-3
    penalty_init: float = 1.0
    bfgs_init_scale: float = 1.0
    bfgs_skip_tol: float = 1e-10
    osqp_tol: float = 1e-7       # kept for API compatibility (unused by ADMM)
    osqp_max_iter: int = 4000   # kept for API compatibility (unused by ADMM)
    osqp_reg: float = 1e-7      # Hessian diagonal regularisation
    admm_rho: float = 1.0       # ADMM penalty parameter
    admm_n_iter: int = 500      # fixed ADMM iterations (scan-based, vmappable)


# ---------------------------------------------------------------------------
# Solver state  (JAX pytree via NamedTuple)
# ---------------------------------------------------------------------------


class SQPState(NamedTuple):
    """Internal SQP iterate.  All fields are JAX arrays so the state is a
    valid JAX pytree compatible with ``lax.scan`` and ``jax.vmap``.

    Parameters
    ----------
    x:
        Current primal iterate, shape ``(n,)``.
    params_p:
        Parameter vector frozen into the state, shape ``(m,)``.
    lam:
        Dual variables for general constraints, shape ``(n_g,)``.
    hessian:
        BFGS Hessian approximation, shape ``(n, n)``.
    grad_lag:
        Lagrangian gradient ``∇_x L(x, λ, p)`` at current iterate, shape ``(n,)``.
    f_val:
        Current objective value ``f(x, p)``, scalar.
    penalty:
        L1 merit penalty parameter ``ρ``, scalar.
    merit:
        Current L1 merit function value ``φ(x; ρ)``, scalar.
    stationarity:
        KKT stationarity residual at current iterate, scalar.
    feasibility:
        KKT primal feasibility residual at current iterate, scalar.
    iteration:
        Current iteration counter (integer scalar).
    converged:
        Boolean scalar — ``True`` when KKT tolerances are satisfied.
    """

    x: jnp.ndarray
    params_p: jnp.ndarray
    lam: jnp.ndarray
    hessian: jnp.ndarray
    grad_lag: jnp.ndarray
    f_val: jnp.ndarray
    penalty: jnp.ndarray
    merit: jnp.ndarray
    stationarity: jnp.ndarray
    feasibility: jnp.ndarray
    iteration: jnp.ndarray
    converged: jnp.ndarray


# ---------------------------------------------------------------------------
# Solve result
# ---------------------------------------------------------------------------


@dataclass
class SQPResult:
    """Result returned by the JAX SQP solver.

    Parameters
    ----------
    success:
        ``True`` if the solver converged within ``max_iter`` iterations.
    objective:
        Optimal objective value ``f(x*, p)``.
    decision_variables:
        Optimal primal solution ``x*``, shape ``(n,)`` or ``(N, n)`` for batches.
    multipliers:
        Optimal dual variables ``λ*`` for general constraints.
    constraints:
        Value of ``g(x*, p)``; ``None`` when the problem has no constraints.
    iterations:
        Number of SQP outer iterations performed.
    kkt_stationarity:
        Final KKT stationarity residual.
    kkt_feasibility:
        Final KKT primal feasibility residual.
    timing:
        Wall-clock solve time in seconds.
    message:
        Human-readable status string.
    """

    success: Any
    objective: Any
    decision_variables: Any
    multipliers: Any = None
    constraints: Optional[Any] = None
    iterations: Any = 0
    kkt_stationarity: Any = 0.0
    kkt_feasibility: Any = 0.0
    timing: float = 0.0
    message: str = ""
