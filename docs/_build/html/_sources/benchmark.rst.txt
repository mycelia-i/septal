Benchmark results
=================

Test suite
----------

The benchmark runs 28 Hock-Schittkowski / Rosenbrock problems × 5 Sobol starts
= **140 total solves**.  Sobol initial guesses use ``scramble=True, seed=42``
for reproducibility.

Problems cover unconstrained, box-constrained, inequality-constrained,
equality-constrained, and mixed problems, including several non-convex cases
(Rosenbrock, HS023, HS071).

.. list-table:: Benchmark progression
   :header-rows: 1
   :widths: 55 20 25

   * - Configuration
     - Solved
     - Notes
   * - Baseline BFGS (no extras)
     - 74 / 140
     -
   * - \+ exact Lagrangian Hessian
     - 82 / 140
     - Better curvature on curved manifolds
   * - \+ Ruiz equilibration
     - 94 / 140
     - Fixed scale imbalance (e.g. hs021)
   * - \+ OSQP polishing
     - 95 / 140
     - Higher-accuracy QP solutions
   * - \+ inertia correction + non-monotone LS + stagnation reset
     - **139 / 140**
     - Full robustness suite

Remaining failure
-----------------

**hs023 start 1/5**: five nonlinear inequality constraints form an
exterior-of-ellipse feasible set.  One of the five Sobol starting points lies
outside the feasible region in a basin with no descent direction toward
feasibility.  This is expected for exterior-feasible problems — multi-start with
more starts or a feasibility-restoration phase is the remedy.

Reference configuration
-----------------------

.. code-block:: python

   from septal.jax.sqp import SQPConfig

   cfg = SQPConfig(
       max_iter=1000,
       admm_n_iter=1000,
       tol_stationarity=1e-7,
       tol_feasibility=1e-7,
       use_exact_hessian=True,
       admm_ruiz_iter=10,
       admm_polish=True,
       penalty_init=10.0,
       admm_rho_eq_scale=100.0,
       nonmonotone_window=5,
       stagnation_patience=30,
       stagnation_reset_hessian=True,
       stagnation_reset_penalty=True,
   )

Problem list
------------

.. list-table::
   :header-rows: 1
   :widths: 15 10 15 60

   * - Problem
     - n
     - Type
     - Description
   * - rosenbrock
     - 2
     - unconstrained
     - Classic banana function, ``f* = 0``
   * - beale
     - 2
     - unconstrained
     - Multi-modal, ``f* = 0``
   * - ackley
     - 2
     - box
     - Highly multi-modal, ``f* = 0``
   * - rastrigin
     - 2
     - box
     - Highly multi-modal, ``f* = 0``
   * - hs021
     - 2
     - inequality
     - 10:1 scale imbalance — Ruiz critical
   * - hs023
     - 2
     - inequality
     - Exterior-of-ellipse feasible set
   * - hs071
     - 4
     - mixed
     - Classic NLP benchmark, ``f* ≈ 17.014``
   * - hs007
     - 2
     - equality
     - ``f* = -√3``
   * - hs026
     - 3
     - equality
     - Degenerate Lagrangian Hessian at ``x* = (1,1,1)``
   * - hs063
     - 3
     - equality
     - Concave quadratic on sphere ∩ hyperplane, ``f* ≈ 961.715``
   * - rosen\_suzuki
     - 4
     - inequality
     - 3 nonlinear ineq, 2 active at ``x*``, ``f* = -44``
   * - hs073
     - 4
     - mixed
     - Bilinear ineq + simplex eq, ``f* ≈ 29.894``
   * - \+ 16 standard HS problems
     - 2–4
     - various
     - hs001–hs100 subset
