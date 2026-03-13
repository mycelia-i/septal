Hessian
=======

.. automodule:: septal.jax.sqp.hessian
   :members:
   :undoc-members:
   :show-inheritance:

Choosing between BFGS and exact Hessian
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - BFGS (``use_exact_hessian=False``)
     - Exact (``use_exact_hessian=True``)
   * - Cost per iter
     - :math:`O(n^2)` (rank-2 update)
     - :math:`O(n^2)` forward passes via ``jax.hessian``
   * - Accuracy
     - Approximate (improves over iterations)
     - Exact (full second-order information)
   * - PD guarantee
     - By construction (damped BFGS)
     - Via inertia correction :math:`\tau I`
   * - Recommended for
     - Large :math:`n`, smooth objectives
     - Small :math:`n`, highly non-convex or constrained

.. note::

   ``regularised_lagrangian_hessian`` applies an additive shift
   :math:`\tau I` where :math:`\tau = \max(\delta_\text{min}, -\lambda_\text{min}(H) + \delta)`.
   This guarantees all eigenvalues :math:`\ge \delta`, keeping the QP Hessian PD
   even on non-convex constraint manifolds.
