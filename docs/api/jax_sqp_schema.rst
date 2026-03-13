Schema
======

.. automodule:: septal.jax.sqp.schema
   :members:
   :undoc-members:
   :show-inheritance:

``ParametricNLPProblem``
------------------------

Defines a parametric NLP instance.  Equality constraints are expressed by
setting ``constraint_lhs[i] == constraint_rhs[i]``.  Infinite bounds
(``-jnp.inf`` / ``jnp.inf``) are supported.

**SQPConfig parameter reference**

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Parameter
     - Default
     - Description
   * - ``max_iter``
     - ``100``
     - Outer SQP iterations
   * - ``tol_stationarity``
     - ``1e-6``
     - Projected-gradient stationarity tolerance
   * - ``tol_feasibility``
     - ``1e-6``
     - L∞ constraint violation tolerance
   * - ``line_search_beta``
     - ``0.5``
     - Armijo backtracking step multiplier
   * - ``line_search_c``
     - ``1e-4``
     - Armijo sufficient-decrease constant
   * - ``line_search_alpha0``
     - ``1.0``
     - Initial step length
   * - ``max_line_search``
     - ``30``
     - Maximum backtracking trials
   * - ``penalty_init``
     - ``1.0``
     - Initial L1 penalty parameter :math:`\rho_0`
   * - ``penalty_eps``
     - ``1e-3``
     - Penalty margin :math:`\varepsilon_\rho`
   * - ``penalty_decrease_factor``
     - ``0.999``
     - Slow shrink of :math:`\rho` when feasible
   * - ``bfgs_init_scale``
     - ``1.0``
     - :math:`H_0 = \sigma_0 I`
   * - ``bfgs_skip_tol``
     - ``1e-10``
     - Skip BFGS update when :math:`\|s\|^2 < \text{tol}`
   * - ``bfgs_max_cond``
     - ``1e8``
     - Reset :math:`H` to :math:`I` when :math:`\text{cond}(H) > \text{tol}`
   * - ``admm_n_iter``
     - ``500``
     - Fixed ADMM inner iterations
   * - ``admm_rho``
     - ``1.0``
     - Initial ADMM penalty :math:`\rho_0`
   * - ``admm_sigma``
     - ``1e-6``
     - ADMM proximal regularisation :math:`\sigma`
   * - ``admm_alpha``
     - ``1.6``
     - ADMM over-relaxation factor
   * - ``admm_adaptive_rho``
     - ``True``
     - Enable adaptive :math:`\rho` scaling
   * - ``admm_rho_eq_scale``
     - ``100.0``
     - Per-row :math:`\rho` multiplier for equality constraints
   * - ``admm_ruiz_iter``
     - ``10``
     - Ruiz equilibration iterations (0 = disable)
   * - ``admm_polish``
     - ``True``
     - OSQP-style active-set polishing
   * - ``admm_polish_reg``
     - ``1e-8``
     - KKT system regularisation for polishing
   * - ``admm_polish_refine``
     - ``3``
     - Iterative refinement steps in polishing
   * - ``use_exact_hessian``
     - ``False``
     - Use ``jax.hessian`` instead of BFGS
   * - ``hess_reg_delta``
     - ``1e-4``
     - Minimum eigenvalue floor for inertia correction
   * - ``hess_reg_min``
     - ``1e-8``
     - Minimum regularisation shift
   * - ``nonmonotone_window``
     - ``5``
     - Non-monotone line search memory :math:`M` (1 = standard Armijo)
   * - ``stagnation_patience``
     - ``30``
     - Reset after N consecutive near-zero steps
   * - ``stagnation_alpha_tol``
     - ``1e-6``
     - Step-size threshold for stagnation detection
   * - ``stagnation_reset_hessian``
     - ``True``
     - Reset :math:`H \to \sigma_0 I` on stagnation
   * - ``stagnation_reset_penalty``
     - ``True``
     - Reset :math:`\rho \to \rho_0` on stagnation

**SQPState field reference**

All fields are JAX arrays.  ``SQPState`` is a ``NamedTuple`` and therefore a
valid JAX pytree, compatible with ``vmap``, ``scan``, and ``pmap``.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Shape
     - Description
   * - ``x``
     - ``(n,)``
     - Current primal iterate
   * - ``params_p``
     - ``(m,)``
     - Parameter vector (fixed throughout solve)
   * - ``lam``
     - ``(n_g,)``
     - Dual variables for :math:`g(x,p)`
   * - ``hessian``
     - ``(n,n)``
     - BFGS approximation or exact Lagrangian Hessian
   * - ``grad_lag``
     - ``(n,)``
     - :math:`\nabla_x L(x,\lambda,p)`
   * - ``f_val``
     - ``()``
     - :math:`f(x,p)`
   * - ``penalty``
     - ``()``
     - L1 penalty parameter :math:`\rho`
   * - ``merit``
     - ``()``
     - L1 merit :math:`\varphi(x;\rho)`
   * - ``stationarity``
     - ``()``
     - Projected-gradient KKT stationarity
   * - ``feasibility``
     - ``()``
     - L∞ constraint violation
   * - ``iteration``
     - ``()``
     - Iteration counter
   * - ``converged``
     - ``()``
     - ``True`` once KKT tolerances satisfied
   * - ``merit_window``
     - ``(M,)``
     - Rolling merit buffer for non-monotone line search
   * - ``stagnation_count``
     - ``()``
     - Consecutive near-zero step count
   * - ``alpha_last``
     - ``()``
     - Last accepted step length
