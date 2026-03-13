SQP algorithm
=============

Problem formulation
-------------------

The JAX SQP solver targets the **parametric NLP**:

.. math::

   \min_{x \in \mathbb{R}^n} \; f(x, p)
   \quad \text{s.t.} \quad
   \ell \le g(x, p) \le r, \quad
   lb \le x \le ub

where :math:`p \in \mathbb{R}^m` is a fixed parameter vector.  When :math:`m = 0`
the problem is non-parametric.  Setting :math:`\ell_i = r_i` defines an equality
constraint; :math:`\ell_i = -\infty` or :math:`r_i = +\infty` define one-sided
inequalities.

Outer SQP iteration
-------------------

At iterate :math:`(x_k, \lambda_k)` the solver performs:

**Step 1 — Evaluate functions and derivatives**

.. math::

   f_k,\; \nabla f_k &= \texttt{jax.value\_and\_grad}(f)(x_k, p) \\
   g_k               &= g(x_k, p) \\
   J_k               &= \texttt{jax.jacfwd}(g)(x_k, p) \in \mathbb{R}^{n_g \times n}

**Step 2 — Solve the QP subproblem** for search direction :math:`d_k`:

.. math::

   \min_{d} \; &\tfrac{1}{2} d^\top H_k d + \nabla f_k^\top d \\
   \text{s.t.} \; &\ell - g_k \le J_k d \le r - g_k \quad \text{(linearised constraints)} \\
                   &lb - x_k \le d \le ub - x_k \quad \text{(box constraints on step)}

This is an inner QP solved via OSQP-style ADMM (see :ref:`qp-subproblem`).

**Step 3 — Penalty update**

.. math::

   \rho_{k+1} = \max\!\left(\|\lambda_k\|_\infty + \varepsilon_\rho,\; \rho_k\right)

**Step 4 — Line search** on the L1 merit function:

.. math::

   \varphi(x;\rho) = f(x,p) + \rho \sum_i
     \max(g_i(x,p) - r_i,\, 0) + \max(\ell_i - g_i(x,p),\, 0)

Armijo backtracking (``lax.while_loop``): find :math:`\alpha_k = \beta^j` such that

.. math::

   \varphi(x_k + \alpha_k d_k) \;\le\; \bar\varphi_k + c\,\alpha_k\, D\varphi_k

where :math:`\bar\varphi_k` is the reference merit (max over last :math:`M`
iterates for the non-monotone variant) and :math:`D\varphi_k` is the directional
derivative.

**Step 5 — Primal and dual update**

.. math::

   x_{k+1} = x_k + \alpha_k d_k, \qquad \lambda_{k+1} = \lambda_\text{QP}

**Step 6 — Hessian update** (damped BFGS, Powell 1978):

.. math::

   s_k &= x_{k+1} - x_k \\
   y_k &= \nabla_x L(x_{k+1}, \lambda_{k+1}, p)
          - \nabla_x L(x_k, \lambda_k, p) \\
   \theta_k &=
     \begin{cases}
       1 & \text{if } s_k^\top y_k \ge 0.2\, s_k^\top H_k s_k \\
       \dfrac{0.8\, s_k^\top H_k s_k}{s_k^\top H_k s_k - s_k^\top y_k}
         & \text{otherwise}
     \end{cases} \\
   r_k &= \theta_k y_k + (1-\theta_k) H_k s_k \\
   H_{k+1} &= H_k
     - \frac{H_k s_k s_k^\top H_k}{s_k^\top H_k s_k}
     + \frac{r_k r_k^\top}{s_k^\top r_k}

**Step 7 — Convergence check**

.. math::

   \|\,\text{proj}_{[lb,ub]}(x - \nabla_x L) - x\,\|_\infty
     &\le \varepsilon_\text{stat} \quad \text{(projected-gradient stationarity)} \\
   \max_i\,\max\bigl(g_i - r_i,\; \ell_i - g_i,\; 0\bigr)
     &\le \varepsilon_\text{feas} \quad \text{(L}\infty\text{ feasibility)}

.. _qp-subproblem:

QP subproblem solver
--------------------

The QP subproblem is cast into OSQP standard form:

.. math::

   \min_{d} \; \tfrac{1}{2} d^\top Q d + c^\top d
   \qquad \text{s.t.} \quad l \le A d \le u

where

.. math::

   Q = H_k + \varepsilon I, \quad c = \nabla f_k, \quad
   A = \begin{bmatrix} J_k \\ I_n \end{bmatrix}, \quad
   l = \begin{bmatrix} \ell - g_k \\ lb - x_k \end{bmatrix}, \quad
   u = \begin{bmatrix} r - g_k \\ ub - x_k \end{bmatrix}

Solved via **three stages**:

1. **Ruiz diagonal equilibration** — scales rows/columns of :math:`(Q, A)` to
   unit :math:`\infty`-norm, removing scale imbalance before ADMM.

2. **ADMM** (``lax.scan``, fixed iterations) — standard OSQP-ADMM with
   adaptive :math:`\rho`, over-relaxation, and per-row equality boosting.

3. **OSQP-style polishing** — classifies the active set from the ADMM solution,
   then solves a reduced KKT system with iterative refinement to recover high
   accuracy.

Robustness features
-------------------

Exact Lagrangian Hessian
~~~~~~~~~~~~~~~~~~~~~~~~

When ``use_exact_hessian=True``, the BFGS approximation is replaced by the
exact second-order information:

.. math::

   \nabla^2_{xx} L(x, \lambda, p)
   = \nabla^2 f(x,p) + \sum_{i=1}^{n_g} \lambda_i\, \nabla^2 g_i(x,p)

computed via ``jax.hessian``.  An inertia correction :math:`\tau I` is applied
to guarantee positive definiteness before passing to the QP solver:

.. math::

   \tau = \max(\delta_\text{min},\; -\lambda_\text{min}(H) + \delta)

Ruiz equilibration
~~~~~~~~~~~~~~~~~~

Finds diagonal matrices :math:`D` (columns) and :math:`E` (rows) such that
after scaling

.. math::

   Q_s = D Q D, \quad A_s = E A D, \quad
   c_s = Dc, \quad l_s = El, \quad u_s = Eu

all row/column :math:`\infty`-norms are approximately 1.  After solving:
:math:`x = D\tilde{x}`, :math:`y = E\tilde{y}`.

OSQP polishing
~~~~~~~~~~~~~~

After ADMM, classifies constraints as lower-active, upper-active, or inactive,
then solves the reduced KKT system (Stellato et al. 2020, Eq. 30):

.. math::

   \begin{bmatrix}
     Q + \sigma I & A^\top \mathrm{diag}(a) \\
     \mathrm{diag}(a) A & \mathrm{diag}((1-a)\varepsilon)
   \end{bmatrix}
   \begin{bmatrix} \hat{x} \\ \hat{y} \end{bmatrix}
   =
   \begin{bmatrix} -c \\ a \odot b_A \end{bmatrix}

where :math:`a_i = 1` for active constraints.  Three acceptance gates prevent
the polished solution from degrading the ADMM result.

Non-monotone line search
~~~~~~~~~~~~~~~~~~~~~~~~

The reference merit is :math:`\bar\varphi_k = \max_{j \in W_k} \varphi_j` over
the last :math:`M` iterates (Grippo et al. 1986), allowing temporary merit
increases to escape saddle points.  Controlled by ``nonmonotone_window`` (set to
1 for standard Armijo).

Stagnation reset
~~~~~~~~~~~~~~~~

After ``stagnation_patience`` consecutive iterations with
:math:`\alpha_k < \alpha_\text{tol}`, the solver resets
:math:`H \leftarrow \sigma_0 I` and :math:`\rho \leftarrow \rho_0`.  This
breaks out of regions where BFGS has accumulated bad curvature.

``lax.scan`` vs ``lax.while_loop``
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - :func:`~septal.jax.sqp.solver.sqp_solve_scan`
     - :func:`~septal.jax.sqp.solver.sqp_solve_single`
   * - Iterations
     - Always ``max_iter``
     - Early exit on convergence
   * - ``jax.vmap``-compatible
     - Yes
     - No (variable iteration count)
   * - ``jax.jit``-compilable
     - Yes
     - Yes
   * - Recommended for
     - Batched / GPU solves
     - Single or sequential multi-start

Converged iterates in ``scan`` mode are frozen via masked tree-map:

.. code-block:: python

   jax.tree.map(
       lambda new, old: jnp.where(state.converged, old, new),
       new_state, state
   )
