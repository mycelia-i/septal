Factory
=======

.. automodule:: septal.jax.sqp.factory
   :members:
   :undoc-members:
   :show-inheritance:

SQPResult field reference
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 18 18 39

   * - Field
     - Single solve
     - Batched
     - Description
   * - ``success``
     - ``bool``
     - ``(N,)``
     - KKT tolerances satisfied
   * - ``objective``
     - scalar
     - ``(N,)``
     - :math:`f(x^*, p)`
   * - ``decision_variables``
     - ``(n,)``
     - ``(N, n)``
     - :math:`x^*`
   * - ``multipliers``
     - ``(n_g,)``
     - ``(N, n_g)``
     - :math:`\lambda^*`
   * - ``constraints``
     - ``(n_g,)``
     - ``(N, n_g)``
     - :math:`g(x^*, p)`
   * - ``kkt_stationarity``
     - scalar
     - ``(N,)``
     - Projected-gradient stationarity residual
   * - ``kkt_feasibility``
     - scalar
     - ``(N,)``
     - L∞ constraint violation at :math:`x^*`
   * - ``iterations``
     - int
     - ``(N,)``
     - Iterations used
   * - ``timing``
     - float
     - float
     - Wall-clock seconds
   * - ``message``
     - str
     - str
     - ``"converged"`` or ``"max_iter reached"``
