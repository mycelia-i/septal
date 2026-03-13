# septal

A lightweight Python library for solving **nonlinear programmes (NLPs)** with JAX-native solvers and optional CasADi/IPOPT integration.

```
min  f(x, p)
s.t. lhs ≤ g(x, p) ≤ rhs
     lb  ≤     x   ≤ ub
```

Two solver backends:

| Backend | Entry point | AD mode | Use case |
|---|---|---|---|
| **JAX SQP** | `ParametricSQPFactory` / `sqp_solve_single` | `jax.grad` / `jax.hessian` | Batched parametric NLP — GPU/TPU, `vmap`-compatible |
| **CasADi / IPOPT** | `SolverFactory` | `jvp` / `vjp` callbacks | General constrained NLP on CPU with IPOPT |

---

## Package structure

```
src/septal/
├── casadax/           — CasADi/IPOPT backend, AD callbacks, schema, utilities
│   ├── schema.py          NLPProblem, SolveResult
│   ├── callbacks.py       JAX↔CasADi AD wrappers (casadify_*)
│   ├── solvers.py         CasadiSolver, JaxSolver
│   ├── factory.py         SolverFactory
│   └── utilities.py       generate_initial_guess (Sobol)
└── jax/
    └── sqp/           — Pure-JAX SQP solver
        ├── schema.py      ParametricNLPProblem, SQPConfig, SQPState, SQPResult
        ├── solver.py      sqp_solve_single, sqp_solve_scan, make_solver
        ├── qp_subproblem.py  ADMM + Ruiz equilibration + OSQP polishing
        ├── hessian.py     Damped BFGS + exact Lagrangian Hessian
        ├── line_search.py L1 merit + Armijo backtracking
        ├── convergence.py KKT residual computation
        └── factory.py     ParametricSQPFactory (high-level API)
```

---

## Installation

```bash
pip install -e .
```

**Dependencies**: `jax`, `jaxlib`, `numpy`, `scipy`, `casadi` (for `casadax` backend only).

> **Float64 precision**: importing `septal.jax.sqp` sets `jax_enable_x64 = True` globally. This is required for numerical stability of BFGS and the KKT system.

---

## Quick start

### 1. JAX SQP — single solve

```python
import jax.numpy as jnp
from septal.jax.sqp import ParametricNLPProblem, SQPConfig, ParametricSQPFactory

def objective(x, p):
    return jnp.sum((x - p) ** 2)

def constraints(x, p):
    return jnp.array([jnp.sum(x) - 1.0])   # equality: sum(x) = 1

problem = ParametricNLPProblem(
    objective=objective,
    bounds=[jnp.full(3, -2.0), jnp.full(3, 2.0)],
    n_decision=3,
    n_params=3,
    constraints=constraints,
    constraint_lhs=jnp.zeros(1),
    constraint_rhs=jnp.zeros(1),
)

factory = ParametricSQPFactory(problem, SQPConfig(max_iter=200))

x0 = jnp.zeros(3)
p  = jnp.array([0.5, -0.3, 0.8])
result = factory.solve(x0, p)

print(result.success)              # True
print(result.decision_variables)   # x* ≈ [0.467, -0.133, 0.667]
print(result.kkt_stationarity)     # < 1e-6
```

### 2. JAX SQP — batched parametric solve (GPU)

```python
import jax
import jax.numpy as jnp
from septal.jax.sqp import ParametricNLPProblem, SQPConfig, ParametricSQPFactory

def objective(x, p):
    return 0.5 * jnp.dot(x, x) - jnp.dot(p, x)

problem = ParametricNLPProblem(
    objective=objective,
    bounds=[jnp.full(4, -5.0), jnp.full(4, 5.0)],
    n_decision=4,
    n_params=4,
)

factory = ParametricSQPFactory(problem, SQPConfig())

# Solve 1000 NLPs in parallel
N = 1000
key = jax.random.PRNGKey(0)
params_batch = jax.random.normal(key, (N, 4))
x0_batch     = jnp.zeros((N, 4))

result_batch = factory.solve_batch(x0_batch, params_batch)
print(result_batch.decision_variables.shape)   # (1000, 4)
print(jnp.mean(result_batch.success))          # ≈ 1.0
```

### 3. JAX SQP — lower-level API (used by mycelia)

For live JAX callables (not serialised surrogates), use `make_parametric_nlp` +
`sqp_solve_single` directly:

```python
import jax
import jax.numpy as jnp
from septal.jax.sqp.schema import SQPConfig
from septal.jax.sqp.solver import sqp_solve_single
from septal.casadax.utilities import generate_initial_guess
from mycelia.solvers.constructor import make_parametric_nlp  # or build your own

objective = lambda x: some_classifier(x, param)
bounds    = [lb, ub]
n_d       = lb.shape[0]

problem    = make_parametric_nlp(objective, None, bounds, n_d)
cfg        = SQPConfig(max_iter=500, use_exact_hessian=True)
x0_batch   = generate_initial_guess(n_starts=8, n_d=n_d, bounds=bounds)
p_empty    = jnp.zeros(0)

states = [sqp_solve_single(problem, jnp.asarray(x0), p_empty, cfg)
          for x0 in x0_batch]
# Pick best: prefer converged, minimise f_val
f_vals    = jnp.stack([s.f_val for s in states])
converged = jnp.stack([s.converged for s in states])
best_idx  = jnp.argmin(jnp.where(converged, f_vals, f_vals + 1e10))
best_f    = f_vals[best_idx]
```

### 4. CasADi / IPOPT

```python
import jax.numpy as jnp
from septal.casadax import NLPProblem, SolverFactory, casadify_reverse

def my_objective(x):          # x: (n_d, 1) — CasADi column convention
    return jnp.sum(x ** 2).reshape(1, 1)

n_d = 3
objective_cb = casadify_reverse(my_objective, n_d)

problem = NLPProblem(
    objective=objective_cb,
    bounds=[jnp.zeros(n_d), jnp.ones(n_d)],
)

cfg     = type('C', (), {'n_starts': 5, 'max_solution_time': 60.0})()
factory = SolverFactory.from_problem(cfg, "general_constrained_nlp", problem)
result  = factory.solve(factory.initial_guess())

print(result.success, result.objective, result.decision_variables)
```

---

## API reference

### `ParametricNLPProblem` (`septal.jax.sqp`)

```python
@dataclass
class ParametricNLPProblem:
    objective:       Callable          # f(x, p) -> scalar
    bounds:          List[Any]         # [lb, ub], each (n,)
    n_decision:      int               # n — dim of x
    n_params:        int               # m — dim of p (0 for non-parametric)
    constraints:     Optional[Callable] = None   # g(x, p) -> (n_g,)
    constraint_lhs:  Optional[Any]     = None    # (n_g,) lower bound on g
    constraint_rhs:  Optional[Any]     = None    # (n_g,) upper bound on g
    n_constraints:   Optional[int]     = None    # inferred from g if None
```

Equality constraint: set `constraint_lhs[i] = constraint_rhs[i] = c`.
Inequality `g(x) ≤ 0`: set `constraint_lhs[i] = -inf`, `constraint_rhs[i] = 0`.

### `SQPConfig` (`septal.jax.sqp`)

| Parameter | Default | Description |
|---|---|---|
| `max_iter` | `100` | Outer SQP iterations |
| `tol_stationarity` | `1e-6` | Projected-gradient stationarity tolerance |
| `tol_feasibility` | `1e-6` | L∞ constraint violation tolerance |
| `line_search_beta` | `0.5` | Armijo backtracking step multiplier |
| `line_search_c` | `1e-4` | Armijo sufficient-decrease constant |
| `line_search_alpha0` | `1.0` | Initial step length |
| `max_line_search` | `30` | Maximum backtracking trials |
| `penalty_init` | `1.0` | Initial L1 penalty parameter ρ₀ |
| `penalty_eps` | `1e-3` | Penalty margin ε_ρ |
| `penalty_decrease_factor` | `0.999` | Slow penalty shrink when feasible |
| `bfgs_init_scale` | `1.0` | H₀ = scale · I |
| `bfgs_skip_tol` | `1e-10` | Skip BFGS update when ‖s‖² < tol |
| `bfgs_max_cond` | `1e8` | Reset H to I when cond(H) > threshold |
| `admm_n_iter` | `500` | Fixed ADMM inner iterations |
| `admm_rho` | `1.0` | Initial ADMM penalty ρ₀ |
| `admm_sigma` | `1e-6` | ADMM proximal regularisation σ |
| `admm_alpha` | `1.6` | ADMM over-relaxation factor |
| `admm_adaptive_rho` | `True` | Enable adaptive ρ scaling |
| `admm_rho_eq_scale` | `100.0` | Per-row ρ multiplier for equality constraints |
| `admm_ruiz_iter` | `10` | Ruiz equilibration iterations (0 = disable) |
| `admm_polish` | `True` | OSQP-style active-set polishing |
| `admm_polish_reg` | `1e-8` | KKT system regularisation for polishing |
| `admm_polish_refine` | `3` | Iterative refinement steps in polishing |
| `use_exact_hessian` | `False` | Use `jax.hessian` instead of BFGS |
| `hess_reg_delta` | `1e-4` | Minimum eigenvalue floor for inertia correction |
| `hess_reg_min` | `1e-8` | Minimum regularisation shift |
| `nonmonotone_window` | `5` | Non-monotone line search memory M (1 = standard Armijo) |
| `stagnation_patience` | `30` | Reset after N consecutive near-zero steps |
| `stagnation_alpha_tol` | `1e-6` | Step-size threshold for stagnation detection |
| `stagnation_reset_hessian` | `True` | Reset H to I on stagnation |
| `stagnation_reset_penalty` | `True` | Reset ρ to `penalty_init` on stagnation |

### `SQPState` — JAX pytree fields

All fields are JAX arrays. `SQPState` is a `NamedTuple` and therefore a valid JAX pytree, compatible with `vmap`, `scan`, and `pmap`.

| Field | Shape | Description |
|---|---|---|
| `x` | `(n,)` | Current primal iterate |
| `params_p` | `(m,)` | Parameter vector (fixed throughout solve) |
| `lam` | `(n_g,)` | Dual variables for `g(x, p)` |
| `hessian` | `(n, n)` | BFGS approximation or exact Lagrangian Hessian |
| `grad_lag` | `(n,)` | `∇_x L(x, λ, p)` |
| `f_val` | `()` | `f(x, p)` |
| `penalty` | `()` | L1 penalty parameter ρ |
| `merit` | `()` | L1 merit `φ(x; ρ)` |
| `stationarity` | `()` | Projected-gradient KKT stationarity |
| `feasibility` | `()` | L∞ constraint violation |
| `iteration` | `()` | Iteration counter |
| `converged` | `()` | `True` once KKT tolerances satisfied |
| `merit_window` | `(M,)` | Rolling merit buffer for non-monotone line search |
| `stagnation_count` | `()` | Consecutive near-zero step count |
| `alpha_last` | `()` | Last accepted step length |

### `SQPResult` fields

| Field | Single solve | Batched | Description |
|---|---|---|---|
| `success` | `bool` | `(N,)` | KKT tolerances satisfied |
| `objective` | scalar | `(N,)` | `f(x*, p)` |
| `decision_variables` | `(n,)` | `(N, n)` | `x*` |
| `multipliers` | `(n_g,)` | `(N, n_g)` | `λ*` |
| `constraints` | `(n_g,)` | `(N, n_g)` | `g(x*, p)` |
| `kkt_stationarity` | scalar | `(N,)` | Stationarity residual |
| `kkt_feasibility` | scalar | `(N,)` | Feasibility residual |
| `iterations` | int | `(N,)` | Iterations used |
| `timing` | float | float | Wall-clock seconds |

### Core solver functions (`septal.jax.sqp.solver`)

```python
# Early-exit — JIT-compilable, NOT directly vmappable
sqp_solve_single(problem, x0, p, cfg) -> SQPState

# Fixed iterations — vmappable, use for batched solves
sqp_solve_scan(problem, x0, p, cfg) -> SQPState

# JIT-compiled vmappable solver (used by ParametricSQPFactory)
make_solver(problem, cfg) -> Callable[[x0, p], SQPState]

# Batched solver over devices
make_batch_solver(problem, cfg, mode="pmap") -> Callable[[x0_batch, p_batch], SQPState]

# Convert state to result
state_to_result(state, problem, timing=0.0) -> SQPResult
batch_state_to_result(state, problem, timing=0.0) -> SQPResult
```

### Utilities (`septal.casadax.utilities`)

```python
generate_initial_guess(n_starts, n_d, bounds) -> jnp.ndarray  # (n_starts, n_d)
```

Generates Sobol-sequence quasi-random points within `bounds`, using `scramble=True, seed=42` for reproducibility.

### AD callbacks (`septal.casadax`)

```python
casadify_forward(fn, n_d)         # force JVP (forward) mode
casadify_reverse(fn, n_d)         # force VJP (reverse) mode
casadify(fn, n_d, n_out)          # auto-select: forward if n_out > n_d
```

---

## SQP algorithm

### Core iteration

At iterate `(x_k, λ_k)`:

1. **Evaluate** `f_k, ∇f_k, g_k, J_k` via `jax.value_and_grad` and `jax.jacfwd`
2. **QP subproblem** — solve for step `d_k`:
   ```
   min  ½ dᵀ H_k d + ∇f_kᵀ d
   s.t. lhs - g_k  ≤  J_k d  ≤  rhs - g_k    (linearised constraints)
        lb  - x_k  ≤    d    ≤  ub  - x_k     (box constraints on step)
   ```
   Solved via OSQP-style ADMM with Ruiz equilibration and active-set polishing.
3. **Penalty update** — `ρ_k = max(‖λ_k‖_∞ + ε_ρ, ρ_{k-1})`
4. **Line search** — Armijo backtracking on L1 merit `φ(x; ρ) = f + ρ‖c‖₁`
5. **Primal update** — `x_{k+1} = x_k + α_k d_k`, `λ_{k+1} = λ_QP`
6. **Hessian update** — damped BFGS (Powell 1978) or exact `∇²_xx L` via `jax.hessian`
7. **Convergence** — projected-gradient stationarity ≤ `tol_stationarity` **and** L∞ feasibility ≤ `tol_feasibility`

### Robustness features

**Exact Lagrangian Hessian** (`use_exact_hessian=True`): computes `∇²_xx L = ∇²f + Σ λᵢ ∇²gᵢ` via `jax.hessian`. Combined with inertia correction (`hess_reg_delta`) to ensure the QP Hessian remains PD on curved constraint manifolds.

**Ruiz diagonal equilibration** (`admm_ruiz_iter=10`): scales constraint rows/columns to unit ∞-norm before ADMM. Removes scale imbalance that stalls ADMM convergence on problems with mixed-scale constraints.

**OSQP-style polishing** (`admm_polish=True`): after ADMM, identifies the active set and solves a reduced KKT system with iterative refinement. Three acceptance gates prevent the polished solution from degrading the ADMM result: (1) must be finite, (2) primal residual must not increase, (3) must have active constraints.

**Non-monotone line search** (`nonmonotone_window=5`): reference merit is `max(merit_window)` — allows temporary merit increases to escape saddle points (Grippo et al. 1986).

**Stagnation reset** (`stagnation_patience=30`): after `N` consecutive near-zero steps, resets `H → bfgs_init_scale · I` and `ρ → penalty_init`. Helps escape local traps on non-convex problems.

**Equality-row boosting** (`admm_rho_eq_scale=100.0`): multiplies the ADMM penalty for equality rows by this factor, avoiding the O(1/k) convergence degradation specific to equalities.

### `lax.scan` vs `lax.while_loop`

| | `sqp_solve_scan` | `sqp_solve_single` |
|---|---|---|
| Iterations | Always `max_iter` | Early exit on convergence |
| Vmappable | Yes (`jax.vmap`) | No (variable iteration count) |
| JIT-compilable | Yes | Yes |
| Use for | Batched solves | Single or sequential multi-start |

Converged iterates in `scan` mode are frozen via:
```python
jax.tree.map(lambda new, old: jnp.where(state.converged, old, new), new_state, state)
```

---

## Benchmark results

28 HS/Rosen test problems × 5 Sobol starts = 140 total runs.

| Configuration | Solved / 140 |
|---|---|
| Baseline BFGS | 74 / 140 |
| + exact Lagrangian Hessian | 82 / 140 |
| + Ruiz equilibration | 94 / 140 |
| + OSQP polishing | 95 / 140 |
| + inertia correction + non-monotone LS + stagnation reset | **139 / 140** |

One failure: `hs023` start 1/5 — exterior-of-ellipse feasible set traps the Sobol point. Not a solver defect; multi-start handles this.

Benchmark config (`nlp-bsuite/scripts/benchmark_sqp.py`):
```python
SQPConfig(
    max_iter=1000, admm_n_iter=1000, tol=1e-7,
    use_exact_hessian=True, admm_ruiz_iter=10, admm_polish=True,
    penalty_init=10.0, admm_rho_eq_scale=100.0,
)
```

---

## Schema — `NLPProblem` (`septal.casadax`)

| Field | Type | Description |
|---|---|---|
| `objective` | `Callable` | JAX / CasADi objective |
| `bounds` | `[lb, ub]` | Box bounds |
| `constraints` | `Callable` (optional) | `g(x) -> (n_g,)` |
| `constraint_lhs` / `rhs` | array (optional) | Bounds on `g(x)` |
| `n_starts` | `int` | Multi-start count (default 5) |
