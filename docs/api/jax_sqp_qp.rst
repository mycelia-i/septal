QP subproblem
=============

.. automodule:: septal.jax.sqp.qp_subproblem
   :members:
   :undoc-members:
   :show-inheritance:

Pipeline overview
-----------------

:func:`solve_qp_subproblem` is the main entry point.  Internally it chains:

1. :func:`form_qp_matrices` — assembles :math:`(Q, c, A, l, u)` from the SQP iterate
2. :func:`ruiz_equilibration` — diagonal scaling to unit :math:`\infty`-norm
3. :func:`admm_qp` — OSQP-style ADMM (``lax.scan``, fixed iterations)
4. :func:`polish_qp` — active-set KKT refinement
5. Unscale primal/dual back to original space

.. note::

   ``jnp.linalg.solve`` inside ``admm_qp``'s ``lax.scan`` is correct because
   the left-hand matrix :math:`(Q + \sigma I + \rho A^\top A)` is constant
   across ADMM iterations — only the right-hand side changes.  XLA caches the
   factorisation.

.. warning::

   Use ``jnp.where(is_act, b_A, 0.0)`` rather than ``is_act * b_A`` in the
   polishing RHS.  When ``b_A`` contains ``inf`` (unbounded constraints),
   ``0 * inf = NaN`` in JAX/XLA and will corrupt the KKT solve.
