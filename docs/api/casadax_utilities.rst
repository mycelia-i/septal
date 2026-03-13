Utilities
=========

.. automodule:: septal.casadax.utilities
   :members:
   :undoc-members:
   :show-inheritance:

.. note::

   :func:`~septal.casadax.utilities.generate_initial_guess` uses
   ``scipy.stats.qmc.Sobol`` with ``scramble=True, seed=42`` for reproducible
   quasi-random initial points.  The same seed is used across all septal
   test suites.
