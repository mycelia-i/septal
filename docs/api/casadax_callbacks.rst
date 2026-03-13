AD callbacks
============

.. automodule:: septal.casadax.callbacks
   :members:
   :undoc-members:
   :show-inheritance:

Choosing AD mode
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Function
     - AD mode
     - Best when
   * - :func:`~septal.casadax.callbacks.casadify_forward`
     - JVP (forward)
     - ``n_out > n_d`` (tall Jacobian)
   * - :func:`~septal.casadax.callbacks.casadify_reverse`
     - VJP (reverse)
     - ``n_out ≤ n_d``, especially scalars
   * - :func:`~septal.casadax.callbacks.casadify`
     - auto-select
     - General use

For scalar objectives (``n_out = 1``), reverse mode (VJP) is always preferred.
