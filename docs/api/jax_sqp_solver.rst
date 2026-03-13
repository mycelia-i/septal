Solver
======

.. automodule:: septal.jax.sqp.solver
   :members:
   :undoc-members:
   :show-inheritance:

Usage notes
-----------

**Choosing between scan and while_loop:**

- Use :func:`sqp_solve_scan` (or :func:`make_solver`) when solving batched problems
  via ``jax.vmap`` — it runs exactly ``cfg.max_iter`` steps and is fully
  vmappable.
- Use :func:`sqp_solve_single` for single or sequential multi-start solves where
  early exit saves compute.

**Multi-start with sqp_solve_single:**

.. code-block:: python

   from septal.jax.sqp.solver import sqp_solve_single
   from septal.casadax.utilities import generate_initial_guess

   x0_batch = generate_initial_guess(n_starts=8, n_d=n_d, bounds=[lb, ub])
   p = jnp.zeros(0)   # non-parametric

   states = [sqp_solve_single(problem, jnp.asarray(x0), p, cfg)
             for x0 in x0_batch]

   # JAX-traceable best-state selection
   f_vals    = jnp.stack([s.f_val for s in states])
   converged = jnp.stack([s.converged for s in states])
   best_f    = f_vals[jnp.argmin(jnp.where(converged, f_vals, f_vals + 1e10))]

**Batched solve with make_solver:**

.. code-block:: python

   from septal.jax.sqp.solver import make_solver

   solve_fn    = make_solver(problem, cfg)          # JIT-compiled
   solve_batch = jax.vmap(solve_fn)                 # vmapped over (x0, p)
   states      = solve_batch(x0_batch, params_batch)
   results     = batch_state_to_result(states, problem)
