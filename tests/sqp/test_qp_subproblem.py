"""Tests for septal.jax.sqp.qp_subproblem."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from septal.jax.sqp.schema import SQPConfig
from septal.jax.sqp.qp_subproblem import form_qp_matrices, solve_qp_subproblem


@pytest.fixture
def cfg():
    return SQPConfig(osqp_tol=1e-8, osqp_max_iter=4000)


class TestFormQPMatrices:
    def test_unconstrained_shape(self, cfg):
        """Box-only QP: G has shape (2n, n) after splitting l<=Cd<=u."""
        n, n_g = 3, 0
        H = jnp.eye(n)
        g = jnp.ones(n)
        jac_g = jnp.zeros((0, n))
        g_val = jnp.zeros(0)
        lb = jnp.full(n, -1.0)
        ub = jnp.full(n, 1.0)
        lhs = jnp.zeros(0)
        rhs = jnp.zeros(0)

        Q, c, G, h = form_qp_matrices(H, g, jac_g, g_val, jnp.zeros(n),
                                       lb, ub, lhs, rhs, n, n_g, cfg.osqp_reg)
        assert Q.shape == (n, n)
        assert c.shape == (n,)
        assert G.shape == (2 * n, n)   # two-sided split → 2n rows
        assert h.shape == (2 * n,)

    def test_constrained_shape(self, cfg):
        """Constrained QP: G has shape (2*(n_g+n), n)."""
        n, n_g = 3, 2
        H = jnp.eye(n)
        g = jnp.ones(n)
        jac_g = jnp.ones((n_g, n))
        g_val = jnp.zeros(n_g)
        lb = jnp.full(n, -1.0)
        ub = jnp.full(n, 1.0)
        lhs = jnp.full(n_g, -jnp.inf)
        rhs = jnp.zeros(n_g)
        x = jnp.zeros(n)

        Q, c, G, h = form_qp_matrices(H, g, jac_g, g_val, x,
                                       lb, ub, lhs, rhs, n, n_g, cfg.osqp_reg)
        assert Q.shape == (n, n)
        assert G.shape == (2 * (n_g + n), n)
        assert h.shape == (2 * (n_g + n),)

    def test_psd_regularisation(self, cfg):
        """Q must be (approx) PSD: min eigenvalue >= 0."""
        n, n_g = 3, 0
        H = jnp.eye(n)
        g = jnp.zeros(n)
        Q, *_ = form_qp_matrices(H, g, jnp.zeros((0, n)), jnp.zeros(0),
                                  jnp.zeros(n), jnp.full(n, -1.0),
                                  jnp.full(n, 1.0), jnp.zeros(0),
                                  jnp.zeros(0), n, n_g, cfg.osqp_reg)
        eigvals = jnp.linalg.eigvalsh(Q)
        assert float(jnp.min(eigvals)) >= -1e-10

    def test_box_bounds_encoded(self, cfg):
        """Upper half of h encodes ub-x; lower half (negated) encodes lb-x."""
        n, n_g = 2, 0
        x = jnp.array([0.5, -0.3])
        lb = jnp.full(n, -1.0)
        ub = jnp.full(n, 1.0)
        H = jnp.eye(n)
        g = jnp.zeros(n)

        _, _, G, h = form_qp_matrices(H, g, jnp.zeros((0, n)), jnp.zeros(0),
                                       x, lb, ub, jnp.zeros(0),
                                       jnp.zeros(0), n, n_g, cfg.osqp_reg)
        # h[:n] = ub - x  (upper bounds for Cd ≤ h_upper)
        assert jnp.allclose(h[:n], ub - x)
        # h[n:] = -(lb - x)  (lower bounds, negated for -Cd ≤ -l)
        assert jnp.allclose(h[n:], -(lb - x))


class TestSolveQPSubproblem:
    def test_unconstrained_box_qp(self, cfg):
        """min ½‖d‖² + d·ones  s.t. -1 ≤ d ≤ 1  →  d* = -1 (clamped at lb)."""
        n = 2
        H = jnp.eye(n)
        grad_f = jnp.ones(n)
        x = jnp.zeros(n)
        lb = jnp.full(n, -1.0)
        ub = jnp.full(n, 1.0)

        d, lam_g = solve_qp_subproblem(
            H, grad_f, jnp.zeros((0, n)), jnp.zeros(0),
            x, lb, ub, jnp.zeros(0), jnp.zeros(0), n, 0, cfg,
        )
        assert d.shape == (n,)
        assert lam_g.shape == (0,)
        # Optimal unconstrained step = -grad_f = -ones, clamped to lb=-1
        assert jnp.allclose(d, jnp.full(n, -1.0), atol=1e-3)

    def test_constrained_qp(self, cfg):
        """min ½‖d‖²  s.t.  d[0] >= 0.5  (linearised as -d[0] <= -0.5).

        QP: min ½‖d‖²  s.t.  -0.5 ≤ J d ≤ ∞  where J = [-1, 0] and g = 0.
        This means  d[0] >= 0.5  →  d* = [0.5, 0].
        """
        n, n_g = 2, 1
        H = jnp.eye(n)
        grad_f = jnp.zeros(n)
        # Constraint  -x[0] <= -0.5  means  lhs=-∞, rhs=-0.5 and g = -x[0] at x=0 → g=0
        # Linearised: J d = [-1, 0] d <= -0.5 - 0 = -0.5
        # But OSQP form is lhs <= Jd <= rhs, so J=[-1,0], lhs=-inf, rhs=-0.5-0=-0.5
        jac_g = jnp.array([[-1.0, 0.0]])
        g_val = jnp.zeros(1)
        x = jnp.zeros(n)
        lb = jnp.full(n, -5.0)
        ub = jnp.full(n, 5.0)
        lhs = jnp.array([-jnp.inf])
        rhs = jnp.array([-0.5])

        d, lam_g = solve_qp_subproblem(
            H, grad_f, jac_g, g_val, x, lb, ub, lhs, rhs, n, n_g, cfg,
        )
        assert d.shape == (n,)
        assert lam_g.shape == (1,)
        # d[0] should satisfy jac_g @ d <= rhs  →  -d[0] <= -0.5  →  d[0] >= 0.5
        assert float(jac_g[0] @ d) <= float(rhs[0]) + 1e-3

    def test_returns_finite(self, cfg):
        """Verify no NaN/Inf in output even for a trivial problem."""
        n = 3
        H = jnp.eye(n)
        grad_f = jnp.zeros(n)
        x = jnp.zeros(n)
        lb = jnp.full(n, -1.0)
        ub = jnp.full(n, 1.0)

        d, lam_g = solve_qp_subproblem(
            H, grad_f, jnp.zeros((0, n)), jnp.zeros(0),
            x, lb, ub, jnp.zeros(0), jnp.zeros(0), n, 0, cfg,
        )
        assert jnp.all(jnp.isfinite(d))

    def test_vmap_compatible(self, cfg):
        """solve_qp_subproblem must be vmappable over batches of H, grad_f."""
        n, batch = 2, 4
        H_batch = jnp.stack([jnp.eye(n)] * batch)
        grad_batch = jax.random.normal(jax.random.PRNGKey(0), (batch, n))
        x_batch = jnp.zeros((batch, n))
        lb = jnp.full(n, -2.0)
        ub = jnp.full(n, 2.0)

        def _solve_one(H, g, x):
            d, _ = solve_qp_subproblem(
                H, g, jnp.zeros((0, n)), jnp.zeros(0),
                x, lb, ub, jnp.zeros(0), jnp.zeros(0), n, 0, cfg,
            )
            return d

        d_batch = jax.vmap(_solve_one)(H_batch, grad_batch, x_batch)
        assert d_batch.shape == (batch, n)
        assert jnp.all(jnp.isfinite(d_batch))
