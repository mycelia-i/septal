"""
Core JAX SQP solver.

Two variants are provided:

sqp_solve_single(problem, x0, p, cfg)
    Uses ``jax.lax.while_loop`` — exits as soon as KKT tolerances are met.
    JIT-compilable but **not vmappable** (variable iteration count).
    Suitable for single-problem solves.

sqp_solve_scan(problem, x0, p, cfg)
    Uses ``jax.lax.scan`` — always runs ``cfg.max_iter`` steps with a
    ``converged`` mask that freezes already-converged iterates.
    Fully **vmappable** and GPU-efficient for batched parametric solves.

make_solver(problem, cfg) -> solve_fn
    Returns a JIT-compiled ``solve_fn(x0, p)`` built with the scan-based
    variant.  Pass it to ``jax.vmap`` to batch over parameters.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from septal.jax.sqp.schema import ParametricNLPProblem, SQPConfig, SQPResult, SQPState
from septal.jax.sqp.convergence import is_converged
from septal.jax.sqp.hessian import bfgs_update, lagrangian_grad
from septal.jax.sqp.line_search import (
    backtracking_line_search,
    l1_merit,
    merit_directional_deriv,
    update_penalty,
)
from septal.jax.sqp.qp_subproblem import solve_qp_subproblem


# ---------------------------------------------------------------------------
# State initialisation
# ---------------------------------------------------------------------------


def init_sqp_state(
    x0: jnp.ndarray,
    p: jnp.ndarray,
    problem: ParametricNLPProblem,
    cfg: SQPConfig,
) -> SQPState:
    """Build the initial ``SQPState`` from a starting guess and parameter.

    Parameters
    ----------
    x0:
        Initial primal guess, shape ``(n,)``.
    p:
        Parameter vector, shape ``(m,)``.  Pass ``jnp.zeros(0)`` for
        parameter-free problems.
    problem:
        Parametric NLP definition.
    cfg:
        Solver configuration.

    Returns
    -------
    SQPState
    """
    n = problem.n_decision
    n_g = problem.n_constraints or 0
    x0 = jnp.asarray(x0).reshape(n)
    p = jnp.asarray(p).reshape(problem.n_params) if problem.n_params > 0 else jnp.zeros(0)

    f0 = jnp.asarray(problem.objective(x0, p)).reshape(())
    lam0 = jnp.zeros(n_g)
    H0 = cfg.bfgs_init_scale * jnp.eye(n)
    gl0 = lagrangian_grad(x0, lam0, p, problem.objective, problem.constraints, n_g)

    lhs = jnp.asarray(problem.constraint_lhs).reshape(n_g) if n_g > 0 else jnp.zeros(0)
    rhs = jnp.asarray(problem.constraint_rhs).reshape(n_g) if n_g > 0 else jnp.zeros(0)
    g0 = problem.constraints(x0, p).reshape(n_g) if n_g > 0 and problem.constraints is not None else jnp.zeros(0)

    merit0 = l1_merit(f0, g0, lhs, rhs, jnp.array(cfg.penalty_init), n_g)
    lb_arr = jnp.asarray(problem.lb).reshape(n)
    ub_arr = jnp.asarray(problem.ub).reshape(n)

    # Dual-free initial stationarity: projected gradient of f onto [lb, ub].
    # Consistent with the QP-step stationarity used in make_sqp_step.
    grad_f0 = jax.grad(problem.objective)(x0, p)
    stat0 = jnp.max(jnp.abs(jnp.clip(x0 - grad_f0, lb_arr, ub_arr) - x0))

    # Initial feasibility
    if n_g > 0 and problem.constraints is not None:
        feas0 = jnp.maximum(
            jnp.max(jnp.maximum(g0 - rhs, 0.0)),
            jnp.max(jnp.maximum(lhs - g0, 0.0)),
        )
    else:
        feas0 = jnp.zeros(())

    return SQPState(
        x=x0,
        params_p=p,
        lam=lam0,
        hessian=H0,
        grad_lag=gl0,
        f_val=f0,
        penalty=jnp.array(cfg.penalty_init),
        merit=merit0,
        stationarity=stat0,
        feasibility=feas0,
        iteration=jnp.array(0, dtype=jnp.int32),
        converged=is_converged(stat0, feas0, cfg),
    )


# ---------------------------------------------------------------------------
# Single SQP step (scan-body signature: (state, _) -> (state, None))
# ---------------------------------------------------------------------------


def make_sqp_step(
    problem: ParametricNLPProblem,
    cfg: SQPConfig,
) -> Callable[[SQPState, None], tuple[SQPState, None]]:
    """Return a scan-compatible SQP step function closed over *problem* and *cfg*.

    The returned function is pure JAX — no Python conditionals on traced
    values — so it can be used inside ``lax.scan`` and ``lax.while_loop``.
    Already-converged states are masked out (identity update).
    """
    n = problem.n_decision
    n_g = problem.n_constraints or 0
    lb = jnp.asarray(problem.lb).reshape(n)
    ub = jnp.asarray(problem.ub).reshape(n)
    lhs = jnp.asarray(problem.constraint_lhs).reshape(n_g) if n_g > 0 else jnp.zeros(0)
    rhs = jnp.asarray(problem.constraint_rhs).reshape(n_g) if n_g > 0 else jnp.zeros(0)

    def _do_step(state: SQPState) -> SQPState:
        x = state.x
        p = state.params_p

        # 1. Gradients and constraint evaluation
        f_val, grad_f = jax.value_and_grad(problem.objective)(x, p)
        f_val = jnp.asarray(f_val).reshape(())
        grad_f = grad_f.reshape(n)

        if n_g > 0 and problem.constraints is not None:
            g_val = problem.constraints(x, p).reshape(n_g)
            jac_g = jax.jacfwd(problem.constraints)(x, p).reshape(n_g, n)
        else:
            g_val = jnp.zeros(0)
            jac_g = jnp.zeros((0, n))

        # 2. Solve QP subproblem
        d, lam_new = solve_qp_subproblem(
            state.hessian, grad_f, jac_g, g_val, x, lb, ub, lhs, rhs, n, n_g, cfg
        )

        # 3. Penalty update (non-decreasing)
        penalty_new = update_penalty(lam_new, state.penalty, n_g, cfg.penalty_eps)

        # 4. Directional derivative estimate and line search
        dir_deriv = merit_directional_deriv(grad_f, d, g_val, lhs, rhs, penalty_new, n_g)
        alpha = backtracking_line_search(
            x, d, p,
            state.merit, dir_deriv, penalty_new,
            problem.objective, problem.constraints,
            lhs, rhs, n_g, cfg,
        )

        # 5. Primal update
        x_new = x + alpha * d

        # 6. New function values
        f_new = jnp.asarray(problem.objective(x_new, p)).reshape(())
        if n_g > 0 and problem.constraints is not None:
            g_new = problem.constraints(x_new, p).reshape(n_g)
        else:
            g_new = jnp.zeros(0)
        merit_new = l1_merit(f_new, g_new, lhs, rhs, penalty_new, n_g)

        # 7. Lagrangian gradient at new iterate (for BFGS)
        grad_lag_new = lagrangian_grad(
            x_new, lam_new, p, problem.objective, problem.constraints, n_g
        )

        # 8. Damped BFGS Hessian update
        s = x_new - x
        y = grad_lag_new - state.grad_lag
        H_new = bfgs_update(state.hessian, s, y, cfg.bfgs_skip_tol)

        # 9. Stationarity: use QP-step norm ‖d‖_∞ (dual-free).
        #    At a KKT point the QP yields d=0 regardless of multiplier accuracy.
        #    Feasibility is computed from the already-evaluated g_new.
        stat_new = jnp.max(jnp.abs(d))
        if n_g > 0:
            feas_new = jnp.maximum(
                jnp.max(jnp.maximum(g_new - rhs, 0.0)),
                jnp.max(jnp.maximum(lhs - g_new, 0.0)),
            )
        else:
            feas_new = jnp.zeros(())
        conv_new = is_converged(stat_new, feas_new, cfg)

        return SQPState(
            x=x_new,
            params_p=p,
            lam=lam_new,
            hessian=H_new,
            grad_lag=grad_lag_new,
            f_val=f_new,
            penalty=penalty_new,
            merit=merit_new,
            stationarity=stat_new,
            feasibility=feas_new,
            iteration=state.iteration + 1,
            converged=conv_new,
        )

    def sqp_step(state: SQPState, _: None) -> tuple[SQPState, None]:
        new_state = _do_step(state)
        # Mask: freeze state once converged
        final_state = jax.tree.map(
            lambda n_val, o_val: jnp.where(state.converged, o_val, n_val),
            new_state,
            state,
        )
        # Ensure converged flag is monotone (True stays True)
        return final_state._replace(
            converged=state.converged | new_state.converged
        ), None

    return sqp_step


# ---------------------------------------------------------------------------
# Full solvers
# ---------------------------------------------------------------------------


def sqp_solve_scan(
    problem: ParametricNLPProblem,
    x0: jnp.ndarray,
    p: jnp.ndarray,
    cfg: SQPConfig,
) -> SQPState:
    """SQP solve using ``lax.scan`` — vmappable over parameters.

    Runs exactly ``cfg.max_iter`` steps; converged iterates are frozen
    after convergence via the masked update in :func:`make_sqp_step`.

    Parameters
    ----------
    problem:
        Parametric NLP definition (static — not a JAX array).
    x0:
        Initial guess, shape ``(n,)``.
    p:
        Parameter vector, shape ``(m,)``.
    cfg:
        Solver configuration (static).

    Returns
    -------
    SQPState
        Final solver state after ``cfg.max_iter`` iterations.
    """
    state = init_sqp_state(x0, p, problem, cfg)
    step_fn = make_sqp_step(problem, cfg)
    final_state, _ = jax.lax.scan(step_fn, state, None, length=cfg.max_iter)
    return final_state


def sqp_solve_single(
    problem: ParametricNLPProblem,
    x0: jnp.ndarray,
    p: jnp.ndarray,
    cfg: SQPConfig,
) -> SQPState:
    """SQP solve using ``lax.while_loop`` — exits early on convergence.

    JIT-compilable but **not directly vmappable**.  Use
    :func:`sqp_solve_scan` (or :func:`make_solver`) for batched solves.

    Parameters
    ----------
    problem, x0, p, cfg:
        See :func:`sqp_solve_scan`.

    Returns
    -------
    SQPState
        Final solver state.
    """
    state = init_sqp_state(x0, p, problem, cfg)
    step_fn = make_sqp_step(problem, cfg)

    def cond_fn(s: SQPState) -> jnp.ndarray:
        return (~s.converged) & (s.iteration < cfg.max_iter)

    def body_fn(s: SQPState) -> SQPState:
        new_s, _ = step_fn(s, None)
        return new_s

    return jax.lax.while_loop(cond_fn, body_fn, state)


def make_solver(
    problem: ParametricNLPProblem,
    cfg: SQPConfig,
) -> Callable[[jnp.ndarray, jnp.ndarray], SQPState]:
    """Return a JIT-compiled, vmappable ``solve_fn(x0, p) -> SQPState``.

    The returned callable uses the scan-based solver and can be passed
    directly to ``jax.vmap`` for batched parametric solves::

        solve_one = make_solver(problem, cfg)
        results   = jax.vmap(solve_one)(x0_batch, params_batch)

    Parameters
    ----------
    problem:
        Parametric NLP (captured in closure, treated as static).
    cfg:
        Solver configuration (captured in closure, treated as static).

    Returns
    -------
    Callable
        JIT-compiled ``(x0, p) -> SQPState``.
    """

    @jax.jit
    def _solve(x0: jnp.ndarray, p: jnp.ndarray) -> SQPState:
        return sqp_solve_scan(problem, x0, p, cfg)

    return _solve


# ---------------------------------------------------------------------------
# State → SQPResult conversion
# ---------------------------------------------------------------------------


def state_to_result(
    state: SQPState,
    problem: ParametricNLPProblem,
    timing: float = 0.0,
) -> SQPResult:
    """Convert a final ``SQPState`` to a user-facing ``SQPResult``."""
    n_g = problem.n_constraints or 0
    n = problem.n_decision

    g_final = None
    if n_g > 0 and problem.constraints is not None:
        g_final = problem.constraints(state.x, state.params_p)

    msg = "converged" if bool(state.converged) else "max_iter reached"
    return SQPResult(
        success=bool(state.converged),
        objective=state.f_val,
        decision_variables=state.x,
        multipliers=state.lam,
        constraints=g_final,
        iterations=int(state.iteration),
        kkt_stationarity=float(state.stationarity),
        kkt_feasibility=float(state.feasibility),
        timing=timing,
        message=msg,
    )


def batch_state_to_result(
    state: SQPState,
    problem: ParametricNLPProblem,
    timing: float = 0.0,
) -> SQPResult:
    """Convert a batched ``SQPState`` (from vmap) to a batched ``SQPResult``.

    All array fields will have a leading batch dimension ``N``.
    """
    n_g = problem.n_constraints or 0

    g_final = None
    if n_g > 0 and problem.constraints is not None:
        g_final = jax.vmap(
            lambda x, p: problem.constraints(x, p)
        )(state.x, state.params_p)

    msg = "batch solve complete"
    return SQPResult(
        success=state.converged,
        objective=state.f_val,
        decision_variables=state.x,
        multipliers=state.lam,
        constraints=g_final,
        iterations=state.iteration,
        kkt_stationarity=state.stationarity,
        kkt_feasibility=state.feasibility,
        timing=timing,
        message=msg,
    )
