# CasADi Optimization & Robotics Tutorial Guide
### A Conceptual Map of the 6-Notebook Series

---

## What Is CasADi and Why Does It Exist?

CasADi is a **symbolic framework for automatic differentiation and numerical optimization**. At its core, it solves one class of problem: given a mathematical expression built from symbolic variables, compute it, differentiate it exactly, and hand it to a QP or NLP solver — all without the user writing derivatives by hand.

The key design philosophy is a clean separation between **symbolic construction** (build the expression graph once) and **numeric evaluation/solving** (call it many times with different numbers). This distinction runs through every notebook and is the single most important pattern to internalize — it is exactly how `Function`, `qpsol`, and `nlpsol` all behave.

The supporting libraries in these tutorials:
- **`casadi`** — symbolic modeling, automatic differentiation, QP/NLP solving
- **`numpy`** — numeric arrays, interoperates directly with CasADi's `DM`
- **`matplotlib`** — visualizing convergence and robot trajectories
- **`pinocchio` (`pin`)** — numeric rigid-body kinematics/Jacobians for the robot
- **`pinocchio.casadi` (`cpin`)** — the *same* kinematics, but built as a differentiable CasADi expression
- **`robot_descriptions`** — loads the UR5 robot model used in the applied notebooks

---

## Notebook 01 — Symbolic Framework

### The Core Idea: `SX`, `MX`, `DM` Are Not Interchangeable

```python
x = ca.SX.sym("x")          # symbolic scalar, expanded elementwise
X = ca.MX.sym("X", 2, 2)    # symbolic, but kept as a single graph node
A = ca.DM([[1, 2], [3, 4]]) # pure numeric matrix
```

- **`SX`** — best for small, explicit expressions (costs, residuals, kinematics kernels).
- **`MX`** — best for composing large graphs or calling `Function`s inside a bigger expression.
- **`DM`** — numeric only; used for actual data going in/out of functions and solvers.

**Never mix `SX` and `MX` directly in one expression** — it raises a `TypeError`. The fix: wrap the `SX` expression in a `Function`, then call that function from the `MX` graph:

```python
sx_fun = ca.Function("sx_fun", [X_sx, y_sx], [f_sx])
composed_mx = sx_fun(ca.DM.eye(2), 1.0) + X_mx   # works
```

### Elementwise `*` vs. Matrix `@`

```python
A_num * B_num   # elementwise multiplication
A_num @ B_num   # matrix multiplication
```
This is the opposite of what many people expect from math notation — get it wrong and every downstream QP/IK expression (e.g. `J @ dq`) silently breaks.

### Sparsity Is Stored Separately From Values

```python
ca.SX.zeros(3, 3)          # dense zero (structural entries present)
ca.SX(3, 3)                 # sparse zero (no structural entries)
ca.SX.sym("L", ca.Sparsity.lower(3))  # symbols only in the lower triangle
```

### Automatic Differentiation

```python
J     = ca.jacobian(r, q)      # d(residual)/d(q)
grad  = ca.gradient(cost, q)   # d(scalar cost)/d(q)
H, _  = ca.hessian(cost, q)    # second derivatives (+ gradient)
```
Derivatives are computed **exactly** by graph transformation, not by finite differences. This is the machinery every solver in later notebooks relies on internally.

---

## Notebook 02 — Function Objects

### The Core Idea: Build Once, Call Many Times

```python
f = ca.Function("f", [x, y], [x, ca.sin(y) * x])
f([0.1, 0.2], 2.0)                     # positional call
```

A `Function` packages symbolic inputs/outputs into one reusable, numerically callable object — accepting Python lists, NumPy arrays, or `DM` interchangeably.

### Named I/O Makes Complex Calls Readable

```python
f_named = ca.Function("f_named", [x, y], [x, ca.sin(y)*x],
                       ["x", "y"], ["copy_of_x", "scaled_x"])
out = f_named(x=[0.1, 0.2], y=2.0)
out["scaled_x"]
```

### One Function, Many Diagnostic Outputs

```python
tools = ca.Function("least_squares_tools", [u],
                     [r, cost, J, grad, H],
                     ["u"], ["r", "cost", "J", "grad", "H"])
```
Bundling residual, cost, Jacobian, gradient, and Hessian into a single `Function` is the reusable-diagnostics pattern used again in the robot-IK notebooks.

**Why this matters:** a `qpsol`/`nlpsol` solver *is itself* a `Function` — built once symbolically, then called repeatedly with new numeric data. Notebook 02 is a rehearsal for that exact behavior.

---

## Notebook 03 — QP Interface (`qpsol`)

### The Core Idea: Two-Stage Construction / Call Contract

```python
qp = {"x": x, "f": f, "g": g, "p": p}          # symbolic, built once
solver = ca.qpsol("name", "qpoases", qp, opts)  # compiled once
sol = solver(x0=..., p=..., lbx=..., ubx=..., lbg=..., ubg=...)  # called many times
```

| Piece | Meaning |
|---|---|
| `x` | decision variables (required) |
| `f` | quadratic objective (required) |
| `g` | optional constraint expression |
| `p` | optional parameters (fixed per solve, changeable between solves) |
| `lbx/ubx`, `lbg/ubg` | variable and constraint bounds |

### The Parametric Least-Squares Pattern (⇒ becomes IK)

```python
objective = 0.5 * ca.sumsqr(J_param @ dq - e_param) + 0.5 * damping * ca.sumsqr(dq)
qp = {"x": dq, "p": ca.vertcat(ca.reshape(J_param, m*n, 1), e_param), "f": objective}
```
$$\min_{\Delta q}\ \tfrac12\|J\Delta q - e\|_2^2 + \tfrac{\lambda}{2}\|\Delta q\|_2^2$$

This single pattern — minimize a weighted, damped residual, subject to bounds — **is** one step of robot inverse kinematics. Notebook 04 reuses it verbatim with a real robot Jacobian.

---

## Notebook 04 — Constrained IK with QP + Pinocchio

### The Core Idea: Nonlinear IK as a Sequence of Linear Steps

Forward kinematics $p(q)$ is nonlinear, so it is **linearized** at the current configuration and solved as a small QP, repeated until convergence:

$$p(q_k + \Delta q) \approx p(q_k) + J_k \Delta q, \qquad e_k = p_{\text{des}} - p(q_k)$$

```python
ik_step_solver = ca.qpsol("ik_step_solver", "qpoases", qp, qp_opts)  # built ONCE, outside the loop

for iteration in range(max_iterations):
    pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
    J6 = pin.computeFrameJacobian(model, data, q, tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    J_position = J6[:3, :]                      # numeric Jacobian from Pinocchio
    error = desired_position - current_position
    sol = ik_step_solver(x0=..., p=ca.vertcat(J_position.flatten(), error), lbx=lbx, ubx=ubx)
    q = pin.integrate(model, q, sol["x"])        # NEVER q = q + dq directly
```

### Joint Limits and Step Size Are Both Just Box Constraints

```python
lbx = np.maximum(q_min - q, -step_limit)
ubx = np.minimum(q_max - q,  step_limit)
```
Remaining joint-limit margin **and** a trust-region-like max step per iteration are combined element-wise into `Δq`'s bounds.

**Key insight:** the QP solver is compiled once, then called ~80 times with different numeric `p` — the direct payoff of the parametric pattern from notebook 03. This is a concrete instance of **Gauss–Newton / sequential-QP** optimization.

---

## Notebook 05 — NLP Interface (`nlpsol`)

### The Core Idea: Same Contract, Now Nonlinear

```python
nlp = {"x": x, "f": f, "g": g, "p": p}
solver = ca.nlpsol("name", "ipopt", nlp, opts)
sol = solver(x0=..., p=..., lbx=..., ubx=..., lbg=..., ubg=...)
```
Structurally identical to `qpsol` (same dictionary keys, same call signature) — the only difference is `f` and `g` may now be genuinely nonlinear, and IPOPT-specific options use an `ipopt.` prefix (`ipopt.print_level`, `ipopt.max_iter`).

### Initial Guess Now Matters

Unlike a convex QP, an NLP can converge to different local optima — or fail — depending on `x0`. Always check convergence:

```python
solver.stats()["return_status"]   # e.g. "Solve_Succeeded"
```

### Nonlinear Constraints Are Just Another `g`

```python
g = ca.vertcat(u[0]**2 + u[1]**2)          # nonlinear inequality, e.g. u0² + u1² ≤ 1
sol = solver(..., lbg=[-ca.inf], ubg=[1.0])
```

---

## Notebook 06 — Constrained IK with NLP + `pinocchio.casadi`

### The Core Idea: Exact Kinematics, One Global Solve

Instead of linearizing and iterating (notebook 04), build the **true nonlinear** forward-kinematics expression symbolically and solve one NLP:

```python
cmodel = cpin.Model(model)              # pinocchio.casadi: symbolic FK, not numeric
cdata = cmodel.createData()
cpin.framesForwardKinematics(cmodel, cdata, q_sym)
ee_position = cdata.oMf[tool_frame_id].translation   # this is now an SX expression!
```

$$\min_{q}\ \tfrac12\|p_{ee}(q)-p_{\text{des}}\|_2^2 + \tfrac{\alpha}{2}\|q-q_{\text{nom}}\|_2^2 \quad \text{s.t.}\quad q_{\min}\le q\le q_{\max},\ \ \|p_{ee}(q)-c_{obs}\|_2^2\ge r_{obs}^2$$

### Obstacle Avoidance Is a Nonlinear Constraint, Not a Bound

```python
g = ca.vertcat(ca.sumsqr(ee_position - obstacle_center_sym) - obstacle_radius_sym**2)
sol = solver(..., lbg=[0.0], ubg=[ca.inf])   # keep-out sphere
```
This kind of constraint is easy in an NLP but awkward to express as a linear QP constraint — the main practical reason to prefer the NLP formulation over notebook 04's approach.

### Redundancy: Multiple Solutions Can Be Equally Valid

```python
for guess in [q0, guess_1, guess_2, guess_3]:
    sol = ik_nlp_solver(x0=guess, p=p_num, lbx=q_min, ubx=q_max, lbg=[0.0], ubg=[ca.inf])
```
The UR5 has 6 joints but only a 3D position task is tracked (3 redundant DOFs). Solving from four different initial guesses yields different but equally feasible joint configurations — a direct, hands-on demonstration of **kinematic redundancy**.

---

## Summary: The Mental Stack

```
┌─────────────────────────────────────────────────────────────┐
│  06. Constrained IK — NLP + pinocchio.casadi                │
│  Symbolic FK, nonlinear obstacle constraint, redundancy      │
│  ↓ needs: nlpsol contract, robot setup, symbolic AD          │
├─────────────────────────────────────────────────────────────┤
│  05. NLP Interface (nlpsol / IPOPT)                          │
│  Same contract as QP, nonlinear f/g, initial-guess matters   │
│  ↓ needs: the x/f/g/p dictionary pattern from 03             │
├─────────────────────────────────────────────────────────────┤
│  04. Constrained IK — Iterative QP + Pinocchio                │
│  Linearize p(q), solve QP for Δq, integrate, repeat          │
│  ↓ needs: qpsol contract, numeric robot Jacobians             │
├─────────────────────────────────────────────────────────────┤
│  03. QP Interface (qpsol / qpOASES)                           │
│  x/f/g/p dictionary, bounds, parametric least-squares         │
│  ↓ needs: Function packaging, symbolic building blocks        │
├─────────────────────────────────────────────────────────────┤
│  02. Function Objects                                         │
│  Build once, call many times; named I/O; SX-in-MX composition │
│  ↓ needs: symbolic variables and AD from 01                   │
├─────────────────────────────────────────────────────────────┤
│  01. Symbolic Framework                                       │
│  SX / MX / DM, sparsity, indexing, jacobian/gradient/hessian  │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick-Reference: Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Mixing `SX` and `MX` in one expression | `TypeError` | Wrap the `SX` side in a `Function`, call it from the `MX` graph |
| Using `*` when a matrix product was meant | Wrong-shaped or wrong-valued result | Use `@` for matrix multiplication; `*` is elementwise |
| Updating configuration with `q = q + dq` | Invalid/drifting configuration for non-Euclidean joints | Always use `pin.integrate(model, q, dq)` |
| Rebuilding the QP/NLP solver every iteration | Slow, defeats the point of the parametric pattern | Build the solver once; pass changing data through `p` only |
| Ignoring `x0` for an NLP | Solver converges to a bad local optimum or fails | Provide a good initial guess; try multiple starts for redundant systems |
| Not checking solver status | Silently using an infeasible/failed solution | Always inspect `solver.stats()["return_status"]` |
| Treating joint limits and step limits as separate constraints | Overly complex bound handling | Combine both into one element-wise `lbx`/`ubx` per QP iteration |
| Expecting a linear QP to handle a nonlinear obstacle constraint | Poor or infeasible approximation | Use an NLP (`nlpsol`) when constraints are genuinely nonlinear |
