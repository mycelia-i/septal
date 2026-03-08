"""
casadinlp.sqp — JAX-based SQP solver for parametric NLPs.

Exports the public API needed to formulate and solve parametric NLPs:

    from casadinlp.sqp import (
        ParametricNLPProblem,
        SQPConfig,
        SQPResult,
        SQPState,
        ParametricSQPFactory,
    )
"""

from casadinlp.sqp.schema import (
    ParametricNLPProblem,
    SQPConfig,
    SQPResult,
    SQPState,
)
from casadinlp.sqp.factory import ParametricSQPFactory
from casadinlp.sqp.solver import (
    make_solver,
    sqp_solve_scan,
    sqp_solve_single,
    init_sqp_state,
    state_to_result,
    batch_state_to_result,
)
from casadinlp.sqp.convergence import kkt_residuals, is_converged
from casadinlp.sqp.hessian import bfgs_update, lagrangian_grad
from casadinlp.sqp.line_search import (
    l1_merit,
    backtracking_line_search,
    update_penalty,
    constraint_violation,
)
from casadinlp.sqp.qp_subproblem import form_qp_matrices, solve_qp_subproblem

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
