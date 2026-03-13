Convergence
===========

.. automodule:: septal.jax.sqp.convergence
   :members:
   :undoc-members:
   :show-inheritance:

KKT conditions
--------------

The solver checks two KKT residuals at each iterate.

**Stationarity** — projected-gradient condition accounting for active box
constraints:

.. math::

   \text{stat} =
   \left\|\,\text{proj}_{[lb,\,ub]}\!\bigl(x - \nabla_x L(x,\lambda,p)\bigr)
   - x\,\right\|_\infty

**Feasibility** — L∞ constraint violation:

.. math::

   \text{feas} = \max_i\,\max\bigl(g_i(x,p) - r_i,\;
                                   \ell_i - g_i(x,p),\; 0\bigr)

Convergence is declared when both residuals are below their respective
tolerances (``tol_stationarity``, ``tol_feasibility`` in
:class:`~septal.jax.sqp.schema.SQPConfig`).
