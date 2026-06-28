# Pinocchio Robotics Tutorial Guide
### A Conceptual Map of the 5-Notebook Series

---

## What Is Pinocchio and Why Does It Exist?

Pinocchio is a **rigid-body dynamics library** built for robotics. At its core, it solves one class of problem: given a robot described as a chain (or tree) of rigid links connected by joints, compute the positions, velocities, forces, and accelerations of all those links efficiently.

The key design philosophy is a clean separation between **structure** (the `model`) and **computation results** (the `data`). This distinction runs through every notebook and is the single most important pattern to internalize.

The supporting libraries in these tutorials:
- **`pinocchio` (`pin`)** — kinematics, dynamics, rigid-body math
- **`coal`** — collision detection (distances, overlaps, contact data)
- **`trimesh`** — mesh creation for visualization
- **`viser`** — 3D web-based visualizer
- **`robot_descriptions`** — a registry of ready-to-load robot URDFs

---

## Notebook 01 — Import and Visualize Robots

### The Core Idea: Three Models, Not One

When you load a robot in Pinocchio, you get **three separate models**, not one:

```python
robot = load_robot_description("ur5_description")
robot.model           # kinematic / dynamic structure
robot.collision_model # geometry used for collision queries
robot.visual_model    # geometry used for rendering
```

This separation matters because what you render (high-res mesh with textures) is rarely what you want for collision checking (simplified convex hulls), and neither is the same as the abstract joint-and-link graph used for kinematics.

### The Configuration Vector `q`

A robot's state is fully described by its **configuration vector `q`**, a numpy array of shape `(nq,)`. For a standard 6-DOF arm like the UR5, `nq = 6` and each entry is a joint angle. But for more exotic joints (quaternion-parameterized ball joints, free-flyers) `nq > nv`, where `nv` is the number of **velocity** degrees of freedom. This distinction matters for integration:

```python
q = pin.neutral(model)      # safe zero-like configuration (respects joint constraints)
q_rand = pin.randomConfiguration(model)   # uniformly random valid configuration
q_new = pin.integrate(model, q, v * dt)   # correct way to update q, handles quaternions
```

**Never do `q = q + v * dt` directly.** The quaternion component of a floating base doesn't add linearly.

### Frames vs. Joints

Pinocchio distinguishes two layers of geometric reference:
- **Joints** — the movable connections. Accessed by joint ID via `model.getJointId(name)`.
- **Frames** — fixed offsets attached to links, tools, sensors, or fingertips. Accessed by frame ID via `model.getFrameId(name)`.

The UR5's `tool0` is a **frame** (a fixed offset from the last link), not a joint. When you want the gripper's pose in the world, you query a frame, not a joint.

```python
data = model.createData()
pin.framesForwardKinematics(model, data, q)  # compute all frame placements
T_tool = data.oMf[model.getFrameId("tool0")]
```

You can also add **custom frames** when no existing frame captures what you need:

```python
new_frame = pin.Frame("my_sensor", parent_joint_id, parent_frame_id,
                      placement_relative_to_parent, pin.FrameType.OP_FRAME)
model.addFrame(new_frame, False)
```

### Floating-Base Robots

When a robot is not bolted to the world (a drone, a legged robot), you load it with a `FreeFlyer` root joint:

```python
floating_robot = load_robot_description("ur5_description",
                                         root_joint=pin.JointModelFreeFlyer())
```

Now `nq = 7` (position xyz + quaternion xyzw) and `nv = 6` (linear + angular velocity). The base's orientation is stored as a quaternion to avoid gimbal lock, which is why `pin.integrate` is always required.

### Composing Robots with `pin.appendModel`

You can bolt one robot onto a frame of another:

```python
combined_model, combined_visual = pin.appendModel(
    ur5.model, allegro.model,
    ur5.visual_model, allegro.visual_model,
    attach_frame_id,          # frame on UR5 where Allegro connects
    pin.SE3.Identity()        # relative pose at the attachment point
)
```

The result is a single unified Pinocchio model with `nq = nq_ur5 + nq_allegro`.

---

## Notebook 02 — Forward Kinematics

### The `model / data` Separation (Most Important Pattern)

```
model  ─── stores the robot's structure (joints, links, inertias, frame definitions)
           DOES NOT change when you move the robot
           
data   ─── stores the results of computations (current joint poses, velocities, etc.)
           MUST be updated every time q changes
```

This design means algorithms can be called in tight loops without re-allocating memory: `data` is a pre-allocated workspace. But it creates a classic pitfall:

```python
# WRONG — reading stale data
pin.forwardKinematics(model, data, q_old)
q_new = something_different
T_tool = data.oMf[tool_frame_id]   # ← still from q_old!

# CORRECT
q_new = something_different
pin.framesForwardKinematics(model, data, q_new)   # recompute
T_tool = data.oMf[tool_frame_id]   # ← now correct
```

### Two Levels of Forward Kinematics

Level 1: **Joint placements** — where are the joints?
```python
pin.forwardKinematics(model, data, q)
T_joint = data.oMi[model.getJointId("wrist_3_joint")]
```

Level 2: **Frame placements** — where are all frames (tools, sensors, etc.)?
```python
# Option A: convenient bulk call (does both levels at once)
pin.framesForwardKinematics(model, data, q)

# Option B: explicit two-step (use when you want control)
pin.forwardKinematics(model, data, q)       # updates oMi
pin.updateFramePlacements(model, data)      # updates oMf from oMi
```

Use `pin.framesForwardKinematics` as your default. The split version matters when you want to compute FK once and then selectively update only one frame in a tight loop.

### Notation You Will See Everywhere

In Pinocchio, transforms follow the naming convention `AМB`, meaning "the pose of frame B expressed in frame A":
- `oMi[j]` — pose of joint `j` in the world (origin) frame
- `oMf[f]` — pose of frame `f` in the world frame

Relative transforms follow directly:
```python
oMbase = data.oMf[base_frame_id]
oMtool = data.oMf[tool_frame_id]
baseMtool = oMbase.inverse() * oMtool   # pose of tool IN base frame
```

This relative transform is what you actually need for most robot programming tasks (e.g., expressing target positions in a sensor's local frame, computing errors in a controller).

---

## Notebook 03 — Transformations, Interpolation, and Screw Motion

### SE(3): The Space of Rigid Transforms

A **rigid transform** (rotation + translation, no scaling or shearing) lives in SE(3), the Special Euclidean group in 3D. Pinocchio's `pin.SE3` represents it as a 3×3 rotation matrix `R` and a 3×1 translation vector `p`:

```python
T = pin.SE3(R, p)       # construct from R and p
T.rotation              # the 3×3 matrix R
T.translation           # the 3×1 vector p
T.homogeneous           # 4×4 homogeneous matrix [[R p]; [0 1]]
```

**Composition** of transforms uses `*`, not matrix multiplication on your side:
```python
T_A_to_C = T_A_to_B * T_B_to_C   # correct
T.inverse()                        # gives T^{-1}
```

### The Lie Group / Lie Algebra Connection

SE(3) is a **Lie group** — a group that is also a smooth manifold. Rotations can't be added like ordinary numbers; going from one rotation to another requires walking along the manifold. The **Lie algebra** se(3) is the tangent space at the identity, which *is* a vector space and can be added normally.

Pinocchio exposes this via `log` and `exp`:
```python
xi = pin.log(T)         # SE3 → se(3): a 6D "twist" vector [v; omega]
T  = pin.exp(xi)        # se(3) → SE3
```

This is critical for computing errors in controllers. You should **never** subtract rotation matrices. Instead:
```python
T_error = T_current.inverse() * T_target
xi_error = pin.log(T_error)   # 6D error vector, safe to use in PID/MPC
```

### Interpolating Between Two Poses

There are two natural ways to interpolate from pose `T0` to `T1`. They give different paths.

**Method 1: Separate R3 (translation) and SO(3) (rotation) interpolation**
```python
def interpolate_r3_so3(T0, T1, alpha):
    p = (1 - alpha) * T0.translation + alpha * T1.translation  # linear in position
    R = T0.rotation @ pin.exp3(alpha * pin.log3(T0.rotation.T @ T1.rotation))
    return pin.SE3(R, p)
```
The position moves in a straight line. The rotation follows a geodesic on SO(3) (the shortest path on the rotation sphere). These are **independent** — the path of the origin and the rotation trajectory are decoupled.

**Method 2: Direct SE(3) interpolation (Screw motion)**
```python
def interpolate_se3(T0, T1, alpha):
    return T0 * pin.exp(alpha * pin.log(T0.inverse() * T1))
```
This is the **screw motion**: translation and rotation are coupled as they would be for a rigid body rotating around and translating along a fixed axis in space. The origin traces a **helix** (curved path) rather than a straight line.

| Property | R3 × SO(3) interp. | SE(3) interp. (screw) |
|---|---|---|
| Position path | Straight line | Helix (curved) |
| Rotation path | Geodesic on SO(3) | Coupled to translation |
| Usage | Cartesian trajectory planning | Physically motivated motion |

For robot end-effector control, **SE(3) interpolation is usually preferred** because it corresponds to what a rigid body actually does. For controllers that separately track position and orientation, R3 × SO(3) is more common.

---

## Notebook 04 — Dynamics

### The Equation of Motion

For a robot not in contact with anything, Newton-Euler dynamics gives:

$$M(q)\, \ddot{q} = \tau - h(q, \dot{q})$$

- `M(q)` — the **mass matrix** (also called inertia matrix), shape `(nv, nv)`. It maps joint accelerations to joint forces. It is symmetric positive definite.
- `τ` — the **generalized torques** (motor inputs), shape `(nv,)`.
- `h(q, v)` — **nonlinear effects**: gravity + Coriolis + centrifugal forces, shape `(nv,)`.

In Pinocchio:
```python
M = pin.crba(model, data, q)    # Composite Rigid Body Algorithm: O(n²) mass matrix
h = pin.nle(model, data, q, v)  # NonLinear Effects: gravity + Coriolis + centrifugal
```

### Forward Dynamics: Computing Accelerations from Torques

Given `q, v, τ`, what is `a = q̈`? Solve:
```
a = M(q)^{-1} * (τ - h(q, v))
```

You can do this yourself, but **you should use ABA instead**:
```python
a = pin.aba(model, data, q, v, tau)  # Articulated Body Algorithm: O(n) — much faster
```

The ABA (Articulated Body Algorithm) is asymptotically `O(n)` in the number of joints, while explicitly inverting `M` is `O(n³)`. For most robots this doesn't matter, but for 30+ DOF systems it becomes critical. Use ABA by default.

The explicit solve is useful for **verification and understanding**:
```python
def forward_dynamics_explicit(q, v, tau):
    M = pin.crba(model, data, q)
    h = pin.nle(model, data, q, v)
    return np.linalg.solve(M, tau - h)   # equivalent to M^{-1} (τ - h)
```

### Numerical Integration (Euler Method)

Given an acceleration `a` at time `t`, the simple Euler integration step is:
```python
v = v + a * dt           # velocity update (linear, fine for small dt)
q = pin.integrate(model, q, v * dt)   # config update — USE THIS, not q += v*dt
```

`pin.integrate` handles quaternions correctly. A free-flyer's orientation quaternion lives on the unit sphere S³ — adding a 3D angular displacement to it requires an exponential map, not addition.

### Why `pin.crba` and ABA Both Exist

| Method | Algorithm | Cost | When to use |
|---|---|---|---|
| `pin.crba` | CRBA | O(n²) | When you need M explicitly (e.g., for control) |
| `pin.aba` | ABA | O(n) | When you only need the acceleration |
| `np.linalg.solve(M, ...)` | Direct solve | O(n³) | Only for teaching/verification |

**Key insight**: ABA is faster because it exploits the tree structure of the robot to avoid ever forming M explicitly. It's the standard in modern robotics simulators.

---

## Notebook 05 — Collision Detection with COAL

### Three Different Questions About Geometry

The notebook draws a clear distinction between three related but separate queries:

```
Distance query   → "How far apart are the objects?"       (returns a scalar ≥ 0)
Collision query  → "Are they overlapping?"                (returns a boolean)
Contact query    → "How deep is the overlap, and where?"  (returns depth, normal, position)
```

These are not equivalent and use different algorithms internally.

### Setting Up a Query

You always need:
1. **Geometry** — a shape type (`coal.Sphere`, `coal.Box`, `coal.Cylinder`, etc.)
2. **Placement** — where the shape sits in the world, as a `coal.Transform3f`

```python
sphere = coal.Sphere(radius=0.20)
box = coal.Box(0.40, 0.30, 0.30)   # extents in x, y, z
T_sphere = make_transform([0.0, 0.0, 0.0])
T_box = make_transform([0.90, 0.0, 0.0])
```

### Distance Query: Objects Not Touching

```python
dist_request = coal.DistanceRequest()
dist_result = coal.DistanceResult()
distance = coal.distance(sphere, T_sphere, box, T_box, dist_request, dist_result)

# The nearest points ("witness points") on each object:
p_on_sphere = dist_result.getNearestPoint1()
p_on_box    = dist_result.getNearestPoint2()
normal      = dist_result.normal          # direction from sphere toward box
```

The **witness points** are the closest surface points on each shape. The vector connecting them is parallel to `normal`.

### Collision and Contact Query: Objects Overlapping

```python
col_request = coal.CollisionRequest()
col_result  = coal.CollisionResult()
coal.collide(sphere, T_sphere, box, T_box, col_request, col_result)

if col_result.isCollision():
    contact = col_result.getContact(0)
    contact.pos               # contact point position in world frame
    contact.normal            # outward normal (points out of sphere, into box)
    contact.penetration_depth # how deep the overlap is (positive = penetrating)
```

**Sign convention**: `contact.normal` points **out of object 1** (the sphere in this case) and **into object 2** (the box). The penetration depth is the distance you would need to separate them along the normal.

### The Pattern for Physics Simulators

The sweep example in notebook 05 is actually the core loop of a contact physics engine:

```
for each timestep:
    1. Compute distance/collision for all object pairs
    2. If distance > 0: no contact, just dynamics
    3. If distance == 0 (touching): apply contact constraint
    4. If penetrating (depth > 0): resolve by applying impulse or penalty force
```

Understanding what COAL returns (world-frame witness/contact points, the normal, the depth) is exactly what you need as inputs to the constraint solvers (PGS, LCP, NCP, CCP) that are covered later in the course.

### Converting World-Frame Data to Local Frames

COAL returns everything **in the world frame**. For many physics algorithms, you need the contact point in the **local frame of each object**:

```python
# T_sphere is the coal transform, coal_to_se3 converts it to pin.SE3
T_sphere_pin = coal_to_se3(T_sphere)
T_box_pin    = coal_to_se3(T_box)

contact_world = np.array(contact.pos)

# Transform into local object frame using inverse transform
contact_local_sphere = T_sphere_pin.inverse().act(contact_world)
contact_local_box    = T_box_pin.inverse().act(contact_world)
```

---

## Summary: The Mental Stack

Reading these notebooks in order builds a stack of concepts where each layer depends on the one below:

```
┌─────────────────────────────────────────────────────────────┐
│  05. COAL Collisions                                        │
│  Geometry queries, witness points, contact data             │
│  ↓ needs: SE3 to place objects, Pinocchio data structures   │
├─────────────────────────────────────────────────────────────┤
│  04. Dynamics                                               │
│  M(q)a = τ - h(q,v),  ABA, CRBA, NLE, integration          │
│  ↓ needs: FK to know where links are, SE3 algebra           │
├─────────────────────────────────────────────────────────────┤
│  03. Transformations                                        │
│  SE3 compose/invert/interpolate, log/exp, screw motion      │
│  ↓ needs: the model/data concept to have FK results         │
├─────────────────────────────────────────────────────────────┤
│  02. Forward Kinematics                                     │
│  model vs data, oMi vs oMf, FK calls, stale data pitfall    │
│  ↓ needs: a loaded robot with joints and frames             │
├─────────────────────────────────────────────────────────────┤
│  01. Import and Visualize                                   │
│  load_robot_description, three models, q-vector, frames,   │
│  floating base, appendModel                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick-Reference: Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Forgetting to recompute FK after changing `q` | Wrong/stale frame positions | Always call `pin.framesForwardKinematics` or `pin.forwardKinematics` before reading `data.oMf` |
| Using `q += v * dt` for floating-base robots | Quaternion drift, invalid poses | Always use `pin.integrate(model, q, v * dt)` |
| Inverting M explicitly for dynamics | Slow for large systems | Use `pin.aba(model, data, q, v, tau)` |
| Computing rotation errors by subtracting matrices | Nonsensical error vectors | Use `pin.log(T0.inverse() * T1)` |
| Treating contact normal as pointing inward | Wrong impulse direction | `contact.normal` points **out of** object 1 |
| Mixing joint IDs and frame IDs | Wrong index lookups | Joints → `getJointId` → `data.oMi`; Frames → `getFrameId` → `data.oMf` |
