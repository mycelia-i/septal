CasADi and IPOPT
================

CasADi
------

`CasADi <https://web.casadi.org>`_ is an open-source framework for **symbolic
automatic differentiation** and **numerical optimisation**, with first-class
support for NLP and optimal control problems.

Key capabilities relevant to septal:

- **Symbolic expression graphs** — functions are represented as directed acyclic
  computation graphs.  Derivatives (Jacobians, Hessians) are computed exactly by
  applying the chain rule symbolically, then evaluated numerically.
- **Multiple AD modes** — forward (JVP) and reverse (VJP) mode, and sparse
  Jacobian/Hessian computation exploiting structural zeros.
- **NLP interface** — ``casadi.nlpsol`` wraps any CasADi symbolic objective and
  constraint function and dispatches to a solver backend (IPOPT, SNOPT, etc.)
  via a unified API.
- **JAX interop** — septal's :mod:`~septal.casadax.callbacks` module wraps JAX
  functions as CasADi ``Callback`` objects, routing CasADi's derivative requests
  to ``jax.jvp`` or ``jax.vjp``.  This lets you define objectives in JAX and
  solve with IPOPT without writing any CasADi symbolic code.

For more information see the `CasADi documentation <https://web.casadi.org>`_.

IPOPT
-----

`IPOPT <https://coin-or.github.io/Ipopt/>`_ (Interior Point OPTimizer) is an
open-source **large-scale nonlinear programming** solver developed under the
COIN-OR project.

It implements a **primal-dual interior-point method** with the following
characteristics:

- **Merit function** — augmented Lagrangian with barrier terms for inequality
  constraints.  Barrier parameter :math:`\mu \to 0` drives the iterate toward
  the boundary of the feasible region.
- **Newton step** — at each iteration a large sparse KKT linear system is
  assembled and solved (typically via MA27/MA57 or MUMPS).  The Hessian can be
  provided analytically, approximated by L-BFGS, or computed via finite
  differences.
- **Filter line search** — combines an optimality measure and a feasibility
  measure in a two-objective filter to accept or reject trial steps, giving
  global convergence guarantees under mild assumptions.
- **Second-order corrections** — applied when the filter rejects a step, helping
  avoid the Maratos effect near the solution.
- **Scalability** — designed for problems with thousands to millions of variables
  and sparse constraint Jacobians.  Uses sparse direct linear algebra throughout.

IPOPT is well-suited for **large, sparse, general NLPs** where the structure of
the constraint Jacobian can be exploited.  It runs on CPU only and requires a
Hessian/Jacobian oracle, which CasADi (and septal's callbacks) provide.

For more information see the `IPOPT documentation <https://coin-or.github.io/Ipopt/>`_.

When to use CasADi/IPOPT vs JAX SQP
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Criterion
     - CasADi / IPOPT
     - JAX SQP
   * - Hardware
     - CPU only
     - CPU, GPU, TPU
   * - Batch solving
     - Sequential (loop)
     - Parallel (``jax.vmap`` / ``jax.pmap``)
   * - Problem scale
     - Large sparse NLPs
     - Small–medium dense NLPs
   * - Parametric solves
     - Re-solve from scratch each time
     - Native — vmapped over parameter batch
   * - JIT compilation
     - No (Python control flow)
     - Yes (``jax.jit``)
   * - Constraint types
     - General (equality, inequality, mixed)
     - General (equality, inequality, mixed)
   * - Global convergence
     - Filter line search (IPOPT)
     - L1 merit + Armijo backtracking
   * - Dependencies
     - ``casadi``, IPOPT (bundled)
     - ``jax`` only
