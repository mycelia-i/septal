# septal

A lightweight library that connects **JAX** functions to **CasADi**'s IPOPT solver via automatic-differentiation callbacks, with a clean standard schema for nonlinear programmes (NLPs) and a pure-JAX batched SQP solver.

## Package structure

```
septal/
├── casadax/   — CasADi-based solvers, AD callbacks, schema, utilities
└── jax/
    └── sqp/   — Pure-JAX batched parametric SQP solver
```

## NLP standard form

```
min  f(x)
s.t. lhs ≤ g(x) ≤ rhs
     lb  ≤  x   ≤ ub
```

## Backends

| Backend | Class | AD mode | Use case |
|---|---|---|---|
| CasADi / IPOPT | `CasadiSolver` | forward or reverse `jvp`/`vjp` | General constrained NLP |
| JAXopt L-BFGS-B | `JaxSolver` | JAXopt native | Box-constrained only |
| JAX SQP | `ParametricSQPFactory` | JAX autodiff | Batched parametric NLP, GPU-accelerated |

## Installation

```bash
pip install -e .
```

Dependencies: `casadi`, `jax`, `jaxopt`, `numpy`, `scipy`.

## Quick start

### CasADi / IPOPT solver

```python
import jax.numpy as jnp
from septal.casadax import NLPProblem, SolverFactory, casadify_reverse

# 1. Define objective as a JAX function
def my_objective(x):          # x: (n_d, 1)
    return jnp.sum(x ** 2).reshape(1, 1)

# 2. Wrap for CasADi
n_d = 3
lb = jnp.zeros(n_d)
ub = jnp.ones(n_d)
objective_cb = casadify_reverse(my_objective, n_d)

# 3. Build problem + solve
problem = NLPProblem(objective=objective_cb, bounds=[lb, ub])
factory = SolverFactory.from_problem(cfg, "general_constrained_nlp", problem)
result = factory.solve(factory.initial_guess())

print(result.success, result.objective, result.decision_variables)
```

### JAX SQP solver (batched parametric)

```python
import jax.numpy as jnp
from septal.jax.sqp import ParametricSQPFactory, ParametricNLPProblem, SQPConfig

def objective(x, p):
    return jnp.sum((x - p) ** 2)

problem = ParametricNLPProblem(
    objective=objective,
    bounds=[jnp.full(n_d, -1.0), jnp.ones(n_d)],
    n_decision=n_d,
    n_params=n_p,
)
factory = ParametricSQPFactory(problem, SQPConfig())

# Single solve
result = factory.solve(x0, p)

# Batched solve (vmapped)
result_batch = factory.solve_batch(x0_batch, params_batch)
```

## Schema

### `NLPProblem` (`septal.casadax`)

| Field | Type | Description |
|---|---|---|
| `objective` | `Callable` | JAX / CasADi objective |
| `bounds` | `[lb, ub]` | Box bounds |
| `constraints` | `Callable` (optional) | Constraint function |
| `constraint_lhs` / `rhs` | array (optional) | Bounds on `g(x)` |
| `n_starts` | `int` | Multi-start count |

### `SolveResult` (`septal.casadax`)

| Field | Type | Description |
|---|---|---|
| `success` | `bool` | Convergence flag |
| `objective` | array | `f(x*)` |
| `decision_variables` | array | `x*` |
| `constraints` | array | `g(x*)` |
| `message` | `str` | Solver status |
| `timing` | `float` | Wall-clock seconds |
| `n_solves` | `int` | Successful starts |

### `ParametricNLPProblem` (`septal.jax.sqp`)

| Field | Type | Description |
|---|---|---|
| `objective` | `Callable` | `f(x, p) -> scalar` |
| `bounds` | `[lb, ub]` | Box bounds |
| `n_decision` | `int` | Dimension of `x` |
| `n_params` | `int` | Dimension of `p` |
| `constraints` | `Callable` (optional) | `g(x, p) -> (n_g,)` |
| `constraint_lhs` / `rhs` | array (optional) | Bounds on `g(x, p)` |

## AD callbacks (`septal.casadax`)

```python
from septal.casadax import casadify_forward, casadify_reverse, casadify

cb_fwd = casadify_forward(fn, n_d)   # force forward mode
cb_rev = casadify_reverse(fn, n_d)   # force reverse mode
cb_auto = casadify(fn, n_d, n_out)   # auto-select based on Jacobian shape
```

Forward mode (jvp) is preferred when `n_out > n_d`; reverse mode (vjp) otherwise.
