"""
casadinlp — JAX-CasADi NLP solver library.

Provides a standard schema for nonlinear programmes and two solver backends:

* CasADi / IPOPT multi-start  (``CasadiSolver``)
* JAXopt L-BFGS-B multi-start (``JaxSolver``)
* Pure-JAX SQP with batched parametric support (``casadinlp.sqp``)

JAX functions are connected to CasADi via efficient AD callbacks
(forward-mode ``jvp`` or reverse-mode ``vjp``) exported as
``casadify_forward`` / ``casadify_reverse`` / ``casadify``.

Typical usage (IPOPT)::

    from casadinlp import NLPProblem, SolverFactory, casadify_reverse

    objective_cb = casadify_reverse(my_jax_objective, n_d)
    problem = NLPProblem(objective=objective_cb, bounds=[lb, ub])
    factory = SolverFactory.from_problem(cfg, "general_constrained_nlp", problem)
    result = factory.solve(factory.initial_guess())

Typical usage (parametric SQP, GPU-batched)::

    from casadinlp.sqp import ParametricNLPProblem, SQPConfig, ParametricSQPFactory

    problem = ParametricNLPProblem(objective=f, bounds=[lb, ub],
                                   n_decision=n, n_params=m,
                                   constraints=g,
                                   constraint_lhs=lhs, constraint_rhs=rhs)
    factory = ParametricSQPFactory(problem, SQPConfig())
    results = factory.solve_batch(x0_batch, params_batch)
"""

from casadinlp.callbacks import (
    JaxCasADiEvaluator,
    JaxCallbackForward,
    JaxCallbackReverse,
    casadify,
    casadify_forward,
    casadify_reverse,
)
from casadinlp.factory import SolverFactory
from casadinlp.schema import NLPProblem, SolveResult
from casadinlp.solvers import BaseSolver, CasadiSolver, JaxSolver
from casadinlp.utilities import generate_initial_guess
from casadinlp import sqp

__all__ = [
    # Schema
    "NLPProblem",
    "SolveResult",
    # Callbacks
    "JaxCasADiEvaluator",
    "JaxCallbackForward",
    "JaxCallbackReverse",
    "casadify",
    "casadify_forward",
    "casadify_reverse",
    # Solvers
    "BaseSolver",
    "CasadiSolver",
    "JaxSolver",
    # Factory
    "SolverFactory",
    # Utilities
    "generate_initial_guess",
    # SQP sub-package re-exports
    "sqp",
]
