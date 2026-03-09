# JAX-SQP Solver: Implementation Plan

**Branch**: `feature/jax-sqp`
**Goal**: Implement an SQP (Sequential Quadratic Programming) solver that uses JAX-based QP solvers for subproblems, supports GPU/accelerator execution, and can be batched over parametric NLP data.

---

## 1. Motivation and Design Goals

The existing `casadinlp` library solves NLPs via CasADi/IPOPT or JAXopt L-BFGS-B. Neither backend supports:

1. **Parametric NLPs** — where the problem depends on an external parameter vector `p` and we want to solve the NLP for many different values of `p` simultaneously.
2. **Accelerator execution** — IPOPT runs on CPU only; JAX operations on GPU/TPU require pure-JAX code paths with no Python-level loops.
3. **Batched solves** — solving `N` NLPs in parallel using `jax.vmap` requires the entire solver loop to be a JAX-traceable function (no Python control flow, no CasADi).

The solution is an **SQP solver written entirely in JAX**, using `jax.lax.while_loop` for the outer SQP iteration and `jaxopt.OSQP` (ADMM-based, already a dependency) for the inner QP subproblems. Because everything is pure JAX, the full solve is:
- JIT-compilable (`jax.jit`) for fast repeated solves
- vmappable (`jax.vmap`) for batched parametric solves
- GPU/TPU-ready with no code changes

---

## 2. Parametric NLP Formulation

We target the standard parametric NLP form:

```
min_{x}  f(x, p)
s.t.     lhs ≤ g(x, p) ≤ rhs       (general constraints)
         lb  ≤  x       ≤ ub        (box constraints)
```

where:
- `x ∈ R^n` — decision variables
- `p ∈ R^m` — fixed parameter vector (defines one instance of the NLP)
- `f : R^n × R^m → R` — objective
- `g : R^n × R^m → R^{n_g}` — constraint function
- `lb, ub, lhs, rhs` — bounds (may themselves depend on `p` in extensions)

The batched use case is: given a batch of parameter vectors `P = [p_1, ..., p_N]` (shape `(N, m)`), solve all `N` NLPs simultaneously on GPU using `jax.vmap(solve, in_axes=(None, None, 0))(problem, x0, P)`.

---

## 3. SQP Algorithm Overview

At iterate `x_k` with multipliers `λ_k`:

**Step 1 — Evaluate functions and derivatives**
```
f_k, ∇f_k = value_and_grad(f)(x_k, p)
g_k        = g(x_k, p)
J_k        = jacfwd(g)(x_k, p)           # shape (n_g, n)
```

**Step 2 — Solve the QP subproblem** for search direction `d_k ∈ R^n`:
```
min_{d}   ½ dᵀ H_k d + ∇f_kᵀ d
s.t.      lhs - g_k ≤ J_k d ≤ rhs - g_k    (linearised general constraints)
          lb  - x_k ≤ d     ≤ ub  - x_k     (box constraints on step)
```

Cast into OSQP standard form `min ½ dᵀQd + cᵀd, l ≤ Ad ≤ u`:
```
Q = H_k,   c = ∇f_k
A = [ J_k ]     l = [ lhs - g_k ]     u = [ rhs - g_k ]
    [  I  ]         [ lb  - x_k ]         [ ub  - x_k ]
```

**Step 3 — Penalty parameter update**
```
ρ_k = max(‖λ_k‖_∞ + ε_ρ, ρ_{k-1})
```

**Step 4 — Backtracking line search** on L1 merit function:
```
φ(x; ρ) = f(x, p) + ρ · ‖[max(g(x,p) - rhs, 0); max(lhs - g(x,p), 0)]‖₁
α_k = argmax_{α = β^j, j=0,1,...} s.t. φ(x_k + α d_k) ≤ φ(x_k) + c·α·Dφ_k
```
Implemented as `jax.lax.while_loop`.

**Step 5 — Primal update and dual update**
```
x_{k+1} = x_k + α_k d_k
λ_{k+1} = λ_QP                        (from QP dual solution)
```

**Step 6 — Hessian update** (Damped BFGS):
```
s_k = x_{k+1} - x_k
y_k = ∇_x L(x_{k+1}, λ_{k+1}, p) - ∇_x L(x_k, λ_k, p)
θ_k = damping factor (Powell 1978, ensures yᵀs > 0.2 sᵀHs)
r_k = θ_k y_k + (1-θ_k) H_k s_k
H_{k+1} = H_k - (H_k s_k sᵀ_k H_k)/(sᵀ_k H_k s_k) + r_k rᵀ_k / (sᵀ_k r_k)
```

**Convergence check (KKT residuals)**:
```
‖∇f_k + Jᵀ_k λ_k‖_∞ ≤ ε_stat           (stationarity)
‖max(g_k - rhs, 0) + max(lhs - g_k, 0)‖_∞ ≤ ε_feas   (primal feasibility)
```

All six steps are pure JAX — no Python conditionals, no CasADi. The outer loop is `jax.lax.while_loop((state, k) → not_converged(state) & k < max_iter, sqp_step, init_state)`.

---

## 4. Repository Layout After Implementation

```
casadinlp/
├── src/casadinlp/
│   ├── __init__.py                    ← add SQP exports
│   ├── schema.py                      ← unchanged
│   ├── callbacks.py                   ← unchanged
│   ├── solvers.py                     ← unchanged
│   ├── factory.py                     ← unchanged
│   ├── utilities.py                   ← unchanged
│   └── sqp/                           ← NEW sub-package
│       ├── __init__.py
│       ├── schema.py                  ← ParametricNLPProblem, SQPConfig, SQPState, SQPResult
│       ├── qp_subproblem.py           ← form QP matrices and call jaxopt.OSQP
│       ├── hessian.py                 ← damped BFGS update (pure JAX)
│       ├── line_search.py             ← L1 merit + backtracking (lax.while_loop)
│       ├── convergence.py             ← KKT residual computation
│       ├── solver.py                  ← SQPSolver: single vmappable JAX solve
│       └── factory.py                 ← ParametricSQPFactory: batched API
└── tests/
    ├── ...existing tests...
    └── sqp/                           ← NEW test sub-package
        ├── conftest.py
        ├── test_schema.py
        ├── test_qp_subproblem.py
        ├── test_hessian.py
        ├── test_line_search.py
        ├── test_convergence.py
        ├── test_solver.py
        └── test_factory.py
```

---

## 5. Detailed Component Specifications

### 5.1 `sqp/schema.py` — Data Structures

#### `ParametricNLPProblem`
```python
@dataclass
class ParametricNLPProblem:
    """Parametric NLP: min f(x,p) s.t. lhs ≤ g(x,p) ≤ rhs, lb ≤ x ≤ ub."""
    objective:       Callable          # f(x, p) -> scalar
    bounds:          List[Any]         # [lb, ub], each shape (n,)
    n_decision:      int               # n — dimension of x
    n_params:        int               # m — dimension of p
    constraints:     Optional[Callable] = None   # g(x, p) -> (n_g,)
    constraint_lhs:  Optional[Any] = None
    constraint_rhs:  Optional[Any] = None
    n_constraints:   Optional[int] = None        # inferred if None

    # Convenience properties: lb, ub, has_constraints (same as NLPProblem)
```

#### `SQPConfig`
```python
@dataclass
class SQPConfig:
    max_iter:          int   = 100
    tol_stationarity:  float = 1e-6
    tol_feasibility:   float = 1e-6
    line_search_beta:  float = 0.5     # backtracking factor
    line_search_c:     float = 1e-4    # Armijo constant
    line_search_alpha0: float = 1.0    # initial step
    max_line_search:   int   = 30
    penalty_eps:       float = 1e-3    # ε_ρ for penalty update
    bfgs_init_scale:   float = 1.0     # initial H = scale * I
    osqp_tol_abs:      float = 1e-7
    osqp_tol_rel:      float = 1e-7
    osqp_max_iter:     int   = 4000
```

#### `SQPState`  (JAX pytree — all arrays, no Python objects)
```python
class SQPState(NamedTuple):
    x:          jnp.ndarray   # (n,)  current primal iterate
    lam:        jnp.ndarray   # (n_g + n,) current dual variables
    hessian:    jnp.ndarray   # (n, n) BFGS Hessian approximation
    penalty:    jnp.ndarray   # () scalar penalty parameter ρ
    iteration:  jnp.ndarray   # () integer
    converged:  jnp.ndarray   # () bool
    merit:      jnp.ndarray   # () current merit function value
```

#### `SQPResult`
```python
@dataclass
class SQPResult:
    success:             bool
    objective:           Any
    decision_variables:  Any        # x*
    multipliers:         Any        # λ*
    constraints:         Optional[Any]
    iterations:          int
    kkt_stationarity:    float
    kkt_feasibility:     float
    timing:              float
    message:             str
```

---

### 5.2 `sqp/qp_subproblem.py` — QP Formation and Solution

**Key function**:
```python
def solve_qp_subproblem(
    hessian: jnp.ndarray,      # H_k  shape (n, n)
    grad_f:  jnp.ndarray,      # ∇f_k shape (n,)
    jac_g:   jnp.ndarray,      # J_k  shape (n_g, n)
    g_val:   jnp.ndarray,      # g_k  shape (n_g,)
    x:       jnp.ndarray,      # x_k  shape (n,)
    lb:      jnp.ndarray,      # box lb shape (n,)
    ub:      jnp.ndarray,      # box ub shape (n,)
    lhs:     jnp.ndarray,      # constraint lhs shape (n_g,)
    rhs:     jnp.ndarray,      # constraint rhs shape (n_g,)
    cfg:     SQPConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:   # (d, lambda_QP)
```

Internal steps:
1. Stack `A = jnp.concatenate([jac_g, jnp.eye(n)], axis=0)` — shape `(n_g + n, n)`
2. Compute `l = jnp.concatenate([lhs - g_val, lb - x])`, `u = jnp.concatenate([rhs - g_val, ub - x])`
3. Symmetrize: `Q = 0.5 * (hessian + hessian.T) + ε * I` (ensure PSD for OSQP)
4. Call `jaxopt.OSQP(Q, grad_f, A, l, u, tol_abs=..., tol_rel=..., maxiter=...)` and extract primal `d` and dual `λ_QP`
5. Return `(d, lam_qp)`

Note: `jaxopt.OSQP` is a differentiable ADMM-based QP solver — it is JIT-compilable and vmappable with static matrix sizes (which are always static for a fixed problem dimension).

---

### 5.3 `sqp/hessian.py` — Damped BFGS Update

```python
def bfgs_update(
    H:   jnp.ndarray,    # current Hessian (n, n)
    s:   jnp.ndarray,    # x_{k+1} - x_k  (n,)
    y:   jnp.ndarray,    # ∇L_{k+1} - ∇L_k (n,)
) -> jnp.ndarray:        # updated Hessian (n, n)
```

Algorithm (Powell damped BFGS, guarantees positive definiteness):
```python
sHs = s @ H @ s
sy  = s @ y
# Damping: if sy < 0.2 * sHs, blend y towards H@s
theta = jnp.where(sy >= 0.2 * sHs, 1.0, 0.8 * sHs / (sHs - sy))
r = theta * y + (1.0 - theta) * (H @ s)
sr = s @ r
H_new = H - jnp.outer(H @ s, H @ s) / sHs + jnp.outer(r, r) / sr
return H_new
```

Additional helper:
```python
def lagrangian_grad(
    x: jnp.ndarray, lam: jnp.ndarray, p: jnp.ndarray,
    problem: ParametricNLPProblem,
) -> jnp.ndarray:
    """∇_x L(x, λ, p) = ∇f(x,p) + Jg(x,p)ᵀ λ"""
```

---

### 5.4 `sqp/line_search.py` — L1 Merit + Backtracking

```python
def l1_merit(
    x: jnp.ndarray, p: jnp.ndarray, rho: jnp.ndarray,
    problem: ParametricNLPProblem,
) -> jnp.ndarray:
    """φ(x; ρ) = f(x,p) + ρ · Σ max(g_i - rhs_i, 0) + max(lhs_i - g_i, 0)"""

def backtracking_line_search(
    x: jnp.ndarray, d: jnp.ndarray, p: jnp.ndarray,
    merit_val: jnp.ndarray, directional_deriv: jnp.ndarray,
    rho: jnp.ndarray, problem: ParametricNLPProblem, cfg: SQPConfig,
) -> jnp.ndarray:    # returns step size α
    """Armijo backtracking via lax.while_loop."""
```

The directional derivative of the L1 merit:
```
Dφ(x; d) = ∇f(x,p)ᵀd - ρ·(‖max(g-rhs,0)‖₁ + ‖max(lhs-g,0)‖₁)
```
(This is always negative for a descent direction when ρ is large enough.)

`lax.while_loop` structure:
```python
def cond(state):   α, _ = state; return φ(x + α·d) > φ_0 + c·α·Dφ
def body(state):   α, i = state; return (β·α, i+1)
init = (α0, 0)
α, _ = lax.while_loop(cond, body, init)
```

---

### 5.5 `sqp/convergence.py` — KKT Residuals

```python
def kkt_residuals(
    x: jnp.ndarray, lam: jnp.ndarray, p: jnp.ndarray,
    problem: ParametricNLPProblem,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns (stationarity_residual, feasibility_residual)."""
    grad_f = jax.grad(problem.objective)(x, p)
    g_val  = problem.constraints(x, p)
    jac_g  = jax.jacfwd(problem.constraints)(x, p)
    lam_g  = lam[:problem.n_constraints]

    stationarity = jnp.max(jnp.abs(grad_f + jac_g.T @ lam_g))
    feas_upper   = jnp.max(jnp.maximum(g_val - problem.constraint_rhs, 0.0))
    feas_lower   = jnp.max(jnp.maximum(problem.constraint_lhs - g_val, 0.0))
    feasibility  = jnp.maximum(feas_upper, feas_lower)
    return stationarity, feasibility

def is_converged(
    stationarity: jnp.ndarray, feasibility: jnp.ndarray, cfg: SQPConfig,
) -> jnp.ndarray:    # bool scalar
```

---

### 5.6 `sqp/solver.py` — Core SQP Solver

**Single-problem solve** (vmappable over `p`):

```python
def sqp_solve(
    problem:  ParametricNLPProblem,
    x0:       jnp.ndarray,     # (n,) initial guess
    p:        jnp.ndarray,     # (m,) parameter vector
    cfg:      SQPConfig,
) -> SQPResult:
```

Internal structure:
```python
def init_state(x0, p, cfg) -> SQPState:
    """Build initial SQPState: H = cfg.bfgs_init_scale * I, lam = 0, etc."""

def sqp_step(state: SQPState, _) -> SQPState:
    """One SQP iteration: eval → QP → penalty update → line search → BFGS."""

# Outer loop
final_state, _ = lax.scan(sqp_step, init_state(x0, p, cfg), None, length=cfg.max_iter)

# OR with early-exit (more JIT-efficient for GPU batching, avoids variable iteration count):
# Use lax.while_loop with converged flag inside SQPState
```

**Design note — `lax.scan` vs `lax.while_loop`**:
- `lax.while_loop` allows early exit but cannot be vmapped directly (variable iterations)
- `lax.scan` with a `converged` mask in state runs all `max_iter` steps but is fully vmappable
- **Default**: use `lax.scan` (always `max_iter` steps) for GPU batching; provide a `while_loop` variant for single-problem solves

```python
# vmappable version (scan-based, for batched use)
@partial(jax.jit, static_argnums=(0, 3))
def sqp_solve_batched_inner(problem, x0, p, cfg):
    ...scan-based...

# early-exit version (while_loop, for single or small batches)
@partial(jax.jit, static_argnums=(0, 3))
def sqp_solve_single(problem, x0, p, cfg):
    ...while_loop-based...
```

**Batch solve** (over parameters):
```python
def sqp_solve_batch(
    problem:  ParametricNLPProblem,
    x0:       jnp.ndarray,   # (N, n) or (n,) broadcast
    params:   jnp.ndarray,   # (N, m)
    cfg:      SQPConfig,
) -> SQPResult:
    """Solve N parametric NLPs simultaneously via jax.vmap."""
    _solve = partial(sqp_solve_batched_inner, problem, cfg=cfg)
    # vmap over (x0, p) simultaneously
    batched = jax.vmap(_solve, in_axes=(0, 0))
    return batched(x0, params)
```

---

### 5.7 `sqp/factory.py` — High-Level Factory

```python
class ParametricSQPFactory:
    """High-level API for batched parametric SQP solves.

    Usage
    -----
    factory = ParametricSQPFactory(problem, cfg)

    # Single solve
    result = factory.solve(x0, p)

    # Batched solve (GPU-accelerated)
    results = factory.solve_batch(x0_batch, params_batch)

    # JIT-compiled batch solve (call once to compile, fast thereafter)
    compiled_solve = factory.compile_batch(N, dtype=jnp.float32)
    results = compiled_solve(x0_batch, params_batch)
    """

    def __init__(self, problem: ParametricNLPProblem, cfg: SQPConfig):
        ...

    def solve(self, x0, p) -> SQPResult:
        """Single parametric NLP solve."""

    def solve_batch(self, x0_batch, params_batch) -> SQPResult:
        """Batched solve — N NLPs in parallel."""

    def compile_batch(self, N: int, dtype=jnp.float32):
        """AOT-compile the batched solver for batch size N and return callable."""

    @classmethod
    def from_nlp_problem(cls, problem: NLPProblem, n_params: int, cfg: SQPConfig):
        """Construct from an existing NLPProblem by adding parameter support."""
```

---

## 6. Implementation Phases

### Phase 1 — Schema and Stubs (Day 1)
**Files**: `sqp/__init__.py`, `sqp/schema.py`

- [ ] Define `ParametricNLPProblem` dataclass
- [ ] Define `SQPConfig` dataclass with all defaults
- [ ] Define `SQPState` as a `NamedTuple` (JAX pytree-compatible)
- [ ] Define `SQPResult` dataclass
- [ ] Write `tests/sqp/test_schema.py` — construction, property checks, pytree registration
- [ ] Export new types from `casadinlp/__init__.py`

### Phase 2 — QP Subproblem Solver (Day 1–2)
**Files**: `sqp/qp_subproblem.py`, `tests/sqp/test_qp_subproblem.py`

- [ ] Implement `form_qp_matrices()` — construct `Q, c, A, l, u` from SQP iterate
- [ ] Implement `solve_qp_subproblem()` wrapping `jaxopt.OSQP`
- [ ] Add PSD regularisation: `Q += ε * I` with `ε = 1e-8`
- [ ] Handle unconstrained case (no `g` function): skip general constraint rows
- [ ] Test: simple 2D QP with known solution; verify `d` and `λ_QP` are correct
- [ ] Test: `jax.vmap` compatibility — verify `solve_qp_subproblem` can be vmapped

### Phase 3 — BFGS Hessian Update (Day 2)
**Files**: `sqp/hessian.py`, `tests/sqp/test_hessian.py`

- [ ] Implement `bfgs_update()` — damped BFGS (Powell 1978)
- [ ] Implement `lagrangian_grad()` helper
- [ ] Verify damping maintains positive definiteness via eigenvalue tests
- [ ] Test: known BFGS update sequence from Nocedal & Wright Example 18.1
- [ ] Test: `jax.vmap` and `jax.jit` compatibility

### Phase 4 — Merit Function and Line Search (Day 2–3)
**Files**: `sqp/line_search.py`, `tests/sqp/test_line_search.py`

- [ ] Implement `l1_merit()` — L1 exact penalty merit function
- [ ] Implement `merit_directional_deriv()` — derivative along search direction
- [ ] Implement `update_penalty()` — ensure penalty large enough for descent
- [ ] Implement `backtracking_line_search()` via `jax.lax.while_loop`
- [ ] Test: verify merit decrease on simple quadratic
- [ ] Test: ensure `while_loop` terminates and gives correct step size
- [ ] Test: `jax.vmap` compatibility with fixed-width backtracking

### Phase 5 — Convergence Checker (Day 3)
**Files**: `sqp/convergence.py`, `tests/sqp/test_convergence.py`

- [ ] Implement `kkt_residuals()` — stationarity and feasibility
- [ ] Implement `is_converged()` — combined check
- [ ] Test: KKT residuals are zero at a known optimal point
- [ ] Test: residuals are large at a non-optimal point

### Phase 6 — Core SQP Solver (Day 3–4)
**Files**: `sqp/solver.py`, `tests/sqp/test_solver.py`

- [ ] Implement `init_sqp_state()` — initialise `SQPState` from `x0, p`
- [ ] Implement `sqp_step()` — single SQP iteration (scan-compatible body function)
- [ ] Implement `sqp_solve_batched_inner()` — scan-based full solve (vmappable)
- [ ] Implement `sqp_solve_single()` — while_loop-based single solve
- [ ] Implement `sqp_solve_batch()` — vmap wrapper
- [ ] Test: 2D quadratic with known solution, verify `x* ≈ solution` to tolerance
- [ ] Test: constrained 2D problem (circle intersection), verify feasibility
- [ ] Test: Rosenbrock (classic non-convex), compare with CasADi/IPOPT reference
- [ ] Test: batched solve with `N=10` identical problems, verify results match single solve
- [ ] Test: batched solve with `N=10` different parameter values, verify each result

### Phase 7 — Factory and Public API (Day 4)
**Files**: `sqp/factory.py`, `tests/sqp/test_factory.py`

- [ ] Implement `ParametricSQPFactory` class
- [ ] Implement `solve()`, `solve_batch()`, `compile_batch()`
- [ ] Implement `from_nlp_problem()` class method
- [ ] Update top-level `casadinlp/__init__.py` with SQP exports
- [ ] Test: end-to-end solve via factory matches solver directly
- [ ] Test: batched factory solve on parametric quadratic (varies `p` linearly)
- [ ] Test: `compile_batch` returns a callable that produces correct results

### Phase 8 — Integration Tests and Benchmarks (Day 5)
**Files**: `tests/sqp/test_integration.py`, `benchmarks/sqp_benchmark.py`

- [ ] Integration test: compare SQP results against CasADi/IPOPT on a library of problems
  - Unconstrained quadratic
  - Constrained quadratic
  - Rosenbrock
  - HS71 (Hock-Schittkowski test problem 71)
- [ ] Benchmark: single solve timing (CPU) vs CasADi baseline
- [ ] Benchmark: batched solve timing (CPU/GPU) — N=1, 10, 100, 1000 parameter values
- [ ] Verify GPU execution: `jax.devices()` shows GPU; assert arrays on GPU device

---

## 7. Key Engineering Decisions and Rationale

### 7.1 `lax.scan` for GPU-Batched Solves
`jax.lax.while_loop` supports early exit but cannot be composed with `jax.vmap` when different batch elements converge at different iterations. `lax.scan` with a `converged` flag in `SQPState` runs the full `max_iter` iterations but treats early-converged elements as identity updates (no-ops). This is the standard pattern for batched iterative algorithms on GPU.

```python
def sqp_step(state, _):
    new_state = _do_sqp_step(state)
    # If already converged, don't update
    return jax.tree_map(
        lambda new, old: jnp.where(state.converged, old, new),
        new_state, state
    ), None
```

### 7.2 `jaxopt.OSQP` as QP Backend
`jaxopt.OSQP` (already a project dependency) is:
- Pure JAX (JIT-able, vmappable, GPU-compatible)
- ADMM-based (robust, no factorisation needed for GPU)
- Differentiable through the QP solution (enables second-order SQP in the future)
- Handles inequality constraints natively via the `l ≤ Ax ≤ u` form

Alternative: implement a simple active-set QP in JAX. This is more complex and only justified if OSQP proves too slow for small-scale subproblems (n ≤ 20).

### 7.3 Damped BFGS vs. L-BFGS
Dense BFGS is `O(n²)` in memory and `O(n²)` per update. For small-to-medium `n` (≤ 500), this is fine and gives better Hessian quality than L-BFGS. On GPU, the dense matrix-vector products are also very efficient. L-BFGS will be provided as a future extension.

### 7.4 L1 Exact Penalty Merit Function
The L1 merit `φ(x) = f(x) + ρ‖c(x)‖₁` is exact (penalty parameter need not → ∞ for exact solutions) and non-smooth (no second-order terms needed). It is the standard merit function for SQP. The augmented Lagrangian merit is smoother but requires multiplier estimates; it can be added as an option later.

### 7.5 No Changes to Existing Code
The `sqp/` sub-package is entirely additive. Existing `solvers.py`, `callbacks.py`, `factory.py`, etc. are not modified, only `__init__.py` is updated to re-export new names.

---

## 8. New Dependencies

No new top-level dependencies. The implementation uses only:
- `jax` (≥0.4) — already present
- `jaxopt` (≥0.8) — already present, provides `OSQP`
- `numpy`, `scipy` — already present

**Recommended**: pin `jaxopt>=0.8.3` in `pyproject.toml` to ensure `OSQP` API stability.

---

## 9. Public API Summary

After completion, users can write:

```python
import jax.numpy as jnp
from casadinlp.sqp import ParametricNLPProblem, SQPConfig, ParametricSQPFactory

# Define parametric NLP: min (x - p)^2 s.t. x >= 0
def objective(x, p):
    return jnp.sum((x - p) ** 2)

def constraints(x, p):
    return -x          # g(x,p) = -x ≤ 0  →  lhs=-inf, rhs=0

problem = ParametricNLPProblem(
    objective=objective,
    bounds=[jnp.array([-10.0]), jnp.array([10.0])],
    n_decision=1,
    n_params=1,
    constraints=constraints,
    constraint_lhs=jnp.array([-jnp.inf]),
    constraint_rhs=jnp.array([0.0]),
)
cfg = SQPConfig(max_iter=50, tol_stationarity=1e-6, tol_feasibility=1e-6)

factory = ParametricSQPFactory(problem, cfg)

# Single solve
result = factory.solve(x0=jnp.array([0.5]), p=jnp.array([3.0]))
# result.decision_variables ≈ [3.0], result.success = True

# Batched solve over 1000 parameter values simultaneously (GPU-accelerated)
params = jnp.linspace(0.0, 5.0, 1000).reshape(-1, 1)
x0s    = jnp.zeros((1000, 1))
results = factory.solve_batch(x0s, params)
# results.decision_variables has shape (1000, 1)
# results.success has shape (1000,)
```

---

## 10. Testing Strategy

| Test Type | Location | Key Assertions |
|-----------|----------|----------------|
| Unit — QP solver | `test_qp_subproblem.py` | d matches analytic solution for simple QPs |
| Unit — BFGS | `test_hessian.py` | PSD after update; matches finite-difference Hessian |
| Unit — line search | `test_line_search.py` | Armijo condition satisfied; step ≥ α_min |
| Unit — KKT | `test_convergence.py` | Residuals zero at known optimal |
| Integration — solver | `test_solver.py` | x* matches IPOPT reference to 1e-5 |
| Batched — vmap | `test_solver.py` | N=10 batch matches N sequential solves |
| End-to-end | `test_factory.py` | Factory API produces same results as direct solver |
| Benchmark | `benchmarks/` | Batched GPU speedup ≥ 10× vs sequential CPU for N≥100 |

---

## 11. Future Extensions (Out of Scope for This Plan)

- **L-BFGS Hessian**: for large-scale problems (n > 500)
- **Exact Hessian**: via `jax.hessian` for small n where it's cheap
- **Warm-starting**: pass previous `SQPState` as initial state for re-solve
- **Sensitivity analysis**: exploit OSQP's differentiability for `dx*/dp`
- **Second-order corrections**: Maratos effect mitigation
- **Augmented Lagrangian merit**: smoother alternative to L1 merit
- **SQP with CasADi Hessians**: hybrid mode using exact CasADi Hessians for CPU solves
