Quick start
===========

JAX SQP — single solve
-----------------------

Solve a constrained parametric NLP with the pure-JAX SQP solver:

.. code-block:: python

    import jax.numpy as jnp
    from septal.jax.sqp import ParametricNLPProblem, SQPConfig, ParametricSQPFactory

    def objective(x, p):
        return jnp.sum((x - p) ** 2)

    def constraints(x, p):
        return jnp.array([jnp.sum(x) - 1.0])   # equality: sum(x) = 1

    problem = ParametricNLPProblem(
        objective=objective,
        bounds=[jnp.full(3, -2.0), jnp.full(3, 2.0)],
        n_decision=3,
        n_params=3,
        constraints=constraints,
        constraint_lhs=jnp.zeros(1),
        constraint_rhs=jnp.zeros(1),
    )

    factory = ParametricSQPFactory(problem, SQPConfig(max_iter=200))

    x0 = jnp.zeros(3)
    p  = jnp.array([0.5, -0.3, 0.8])
    result = factory.solve(x0, p)

    print(result.success)              # True
    print(result.decision_variables)   # x* ≈ [0.467, -0.133, 0.667]
    print(result.kkt_stationarity)     # < 1e-6

JAX SQP — batched parametric solve (GPU)
-----------------------------------------

Solve *N* NLP instances in parallel using ``jax.vmap``:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from septal.jax.sqp import ParametricNLPProblem, SQPConfig, ParametricSQPFactory

    def objective(x, p):
        return 0.5 * jnp.dot(x, x) - jnp.dot(p, x)

    problem = ParametricNLPProblem(
        objective=objective,
        bounds=[jnp.full(4, -5.0), jnp.full(4, 5.0)],
        n_decision=4,
        n_params=4,
    )

    factory = ParametricSQPFactory(problem, SQPConfig())

    N = 1000
    key = jax.random.PRNGKey(0)
    params_batch = jax.random.normal(key, (N, 4))
    x0_batch     = jnp.zeros((N, 4))

    result_batch = factory.solve_batch(x0_batch, params_batch)
    print(result_batch.decision_variables.shape)   # (1000, 4)
    print(jnp.mean(result_batch.success))          # ≈ 1.0

.. note::

   :meth:`~septal.jax.sqp.factory.ParametricSQPFactory.solve_batch` is built on
   :func:`~septal.jax.sqp.solver.sqp_solve_scan` (``lax.scan``-based, fixed
   iteration count) which is fully ``vmap``-compatible.  For a single early-exit
   solve use :func:`~septal.jax.sqp.solver.sqp_solve_single` (``lax.while_loop``).

JAX SQP — lower-level API
--------------------------

When an objective is a live JAX callable (not a serialised surrogate),
use :func:`~septal.jax.sqp.solver.sqp_solve_single` directly together with
:func:`~septal.casadax.utilities.generate_initial_guess` for multi-start:

.. code-block:: python

    import jax.numpy as jnp
    from septal.jax.sqp.schema import SQPConfig
    from septal.jax.sqp.solver import sqp_solve_single
    from septal.casadax.utilities import generate_initial_guess

    # Build ParametricNLPProblem from a plain callable
    # (see the make_parametric_nlp helper in your application code)
    from septal.jax.sqp.schema import ParametricNLPProblem

    objective = lambda x, p: classifier(x, param.reshape(1, -1)).flatten()[0]
    problem   = ParametricNLPProblem(
        objective=objective,
        bounds=[lb, ub],
        n_decision=n_d,
        n_params=0,
    )
    cfg      = SQPConfig(max_iter=500, use_exact_hessian=True)
    x0_batch = generate_initial_guess(n_starts=8, n_d=n_d, bounds=[lb, ub])
    p_empty  = jnp.zeros(0)

    states = [sqp_solve_single(problem, jnp.asarray(x0), p_empty, cfg)
              for x0 in x0_batch]

    # JAX-traceable best-state selection
    f_vals    = jnp.stack([s.f_val for s in states])
    converged = jnp.stack([s.converged for s in states])
    best_f    = f_vals[jnp.argmin(jnp.where(converged, f_vals, f_vals + 1e10))]

CasADi / IPOPT
--------------

Wrap a JAX function for CasADi's IPOPT solver:

.. code-block:: python

    import jax.numpy as jnp
    from septal.casadax import NLPProblem, SolverFactory, casadify_reverse

    def my_objective(x):    # x: (n_d, 1) — CasADi column convention
        return jnp.sum(x ** 2).reshape(1, 1)

    n_d          = 3
    objective_cb = casadify_reverse(my_objective, n_d)

    problem = NLPProblem(
        objective=objective_cb,
        bounds=[jnp.zeros(n_d), jnp.ones(n_d)],
    )

    cfg     = type('C', (), {'n_starts': 5, 'max_solution_time': 60.0})()
    factory = SolverFactory.from_problem(cfg, "general_constrained_nlp", problem)
    result  = factory.solve(factory.initial_guess())

    print(result.success, result.objective, result.decision_variables)

:func:`~septal.casadax.callbacks.casadify` auto-selects forward or reverse mode
based on the Jacobian shape.  Use
:func:`~septal.casadax.callbacks.casadify_forward` (JVP) when ``n_out > n_d``,
:func:`~septal.casadax.callbacks.casadify_reverse` (VJP) otherwise.
