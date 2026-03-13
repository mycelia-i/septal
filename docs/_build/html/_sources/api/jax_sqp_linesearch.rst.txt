Line search
===========

.. automodule:: septal.jax.sqp.line_search
   :members:
   :undoc-members:
   :show-inheritance:

L1 merit function
-----------------

The L1 exact penalty merit function is:

.. math::

   \varphi(x;\rho) = f(x,p)
     + \rho \sum_i \max(g_i(x,p) - r_i,\; 0)
     + \rho \sum_i \max(\ell_i - g_i(x,p),\; 0)

It is *exact*: the penalty parameter need not tend to infinity to recover the
true optimal solution.  The directional derivative along the SQP step :math:`d`
is always non-positive when :math:`\rho \ge \|\lambda_k\|_\infty + \varepsilon`,
guaranteeing the Armijo condition can be satisfied.

Non-monotone variant
--------------------

When ``nonmonotone_window > 1``, the Armijo condition uses the maximum merit
over the last :math:`M` iterates as the reference value:

.. math::

   \varphi(x_k + \alpha d_k) \le \max_{j \in W_k} \varphi_j
   + c\,\alpha\,D\varphi_k

This allows temporary merit increases and is effective for escaping saddle
points on non-convex problems (Grippo, Lampariello & Lucidi, 1986).
