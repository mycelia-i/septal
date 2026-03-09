"""
septal.jax.sqp — JAX-based SQP solver for parametric NLPs.

Exports the public API needed to formulate and solve parametric NLPs:

    from septal.jax.sqp import (
        ParametricNLPProblem,
        SQPConfig,
        SQPResult,
        SQPState,
        ParametricSQPFactory,
    )
"""

from septal.jax.sqp.schema import (
    ParametricNLPProblem,
    SQPConfig,
    SQPResult,
    SQPState,
)
from septal.jax.sqp.factory import ParametricSQPFactory
from septal.jax.sqp.solver import (
    make_solver,
    sqp_solve_scan,
    sqp_solve_single,
    init_sqp_state,
    state_to_result,
    batch_state_to_result,
)
from septal.jax.sqp.convergence import kkt_residuals, is_converged
from septal.jax.sqp.hessian import bfgs_update, lagrangian_grad
from septal.jax.sqp.line_search import (
    l1_merit,
    backtracking_line_search,
    update_penalty,
    constraint_violation,
)
from septal.jax.sqp.qp_subproblem import form_qp_matrices, solve_qp_subproblem

__all__ = [
    # Schema
    "ParametricNLPProblem",
    "SQPConfig",
    "SQPResult",
    "SQPState",
    # Factory
    "ParametricSQPFactory",
    # Solver
    "make_solver",
    "sqp_solve_scan",
    "sqp_solve_single",
    "init_sqp_state",
    "state_to_result",
    "batch_state_to_result",
    # Convergence
    "kkt_residuals",
    "is_converged",
    # Hessian
    "bfgs_update",
    "lagrangian_grad",
    # Line search
    "l1_merit",
    "backtracking_line_search",
    "update_penalty",
    "constraint_violation",
    # QP subproblem
    "form_qp_matrices",
    "solve_qp_subproblem",
]
