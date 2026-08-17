# MuJoCo Robotics Tutorial Guide
### A Conceptual Map of the Tutorial Series

---

## What Is MuJoCo and Why Is It Used?

**MuJoCo** (**Mu**lti-**Jo**int dynamics with **Co**ntact) is a physics engine, not just a renderer: its job is to take a description of a mechanical system and compute how it moves forward in time, including what happens when parts of it collide. In this tutorial it plays four overlapping roles at once:

- **A modeling framework** — you describe bodies, joints, geometries, and actuators in an XML-based language (**MJCF**).
- **A dynamics engine** — given the current state, it computes the next state by solving the equations of motion (optionally including contact).
- **A simulation environment** — it advances time step by step, exposing forces, positions, and other quantities you can read or modify at every step.
- **A visualization tool** — an interactive viewer lets you *see* the simulated system and perturb it with the mouse.

The tutorial is built around one architectural idea that recurs in every notebook: the strict separation between **static model description** and **dynamic simulation state**. Understanding this separation early makes everything downstream (kinematics, dynamics, contact, control) easier to reason about.

```
robot/model description  →  MjModel   (never changes during simulation)
simulation state          →  MjData    (changes every timestep)
physics/dynamics          →  mj_step(), mj_forward(), mj_kinematics()  (functions that read MjModel + MjData, write into MjData)
control inputs             →  data.ctrl   (you write this)
contacts                   →  discovered automatically from geometry, solved as part of the dynamics
visualization               →  a separate layer (viewer / renderer) that only *reads* MjData to draw a picture
```

The three notebooks build on each other directly: Notebook 01 establishes the model/data/simulation-loop/control vocabulary using toy pendulums; Notebook 02 uses that same vocabulary to study **contact** in isolation with a falling box; Notebook 03 combines everything — a real robot description, a PD controller, and contact — into one applied exercise (a 3-DOF finger robot hitting a cube).

---

## Notebook 01 — Introduction to MuJoCo

### The Core Idea: The Model/Data Split, and How a Simulation Advances in Time

The notebook's central goal is to make you fluent in four things: (1) how a system is *described* (MJCF → `MjModel`), (2) how its *state* is represented and evolves (`MjData`), (3) how to *run* a simulation loop, and (4) how to *control* it with actuators.

### MJCF: Describing a Robot Declaratively

**MJCF** is MuJoCo's XML-based modeling language. A minimal valid model is just `<mujoco/>`. Every physical element lives inside `<worldbody>`, which is the single top-level body and defines the global Cartesian origin.

```xml
<mujoco>
  <worldbody>
    <body name="pendulum" euler="0 0 90">
      <geom name="blue_rod" type="cylinder" pos="0 0 .4" size=".025 .35" rgba="0 0 1 1"/>
    </body>
  </worldbody>
</mujoco>
```

**Key distinction: `body` vs. `geom`.** A `body` is where mass, inertia, and joints are attached — it is the object that *moves*. A `geom` attached to a body defines its *visual appearance and collision shape*. A body can have zero, one, or several geoms. **Do not confuse a body (the movable, inertial entity) with a geom (its visual/collision shape).**

`mujoco.MjModel.from_xml_string(...)` (or `from_xml_path(...)` for a file) invokes MuJoCo's **model compiler**, turning the human-readable XML into a compact binary `MjModel` object that the simulation engine can use efficiently.

### `MjModel`: The Static Description

`MjModel` stores everything that **does not change over the course of a simulation** — the structure of the mechanism, its constants, and its parameters. Examples encountered in the notebook:

| Field | Meaning |
|---|---|
| `ngeom` / `nbody` | number of geoms / bodies in the model |
| `geom_rgba` | colors of the geometries |
| `nq` | number of generalized position coordinates |
| `nv` | number of generalized velocity coordinates (DOFs) |
| `nu` | number of actuators |
| `opt` | physics options (timestep, gravity, integrator, ...) |

**Important — `nq` is not always equal to `nv`.** They coincide for simple hinge/slide joints (1 position value ↔ 1 velocity value each), but a **free joint** (6 spatial DOFs) uses a 3D position + a 4D unit quaternion for orientation, i.e. `nq` contributes 7 while `nv` contributes only 6 for that joint. The notebook doesn't dwell on this explicitly for the free joint (that appears more concretely in Notebook 02), but it does drive home the general principle that **`nq`, `nv`, and `nu` are conceptually distinct quantities and must not be assumed equal.**

### Degrees of Freedom Come From Joints, Not Bodies

A body with geoms but **no joint** is rigidly welded to its parent (here, the world) and has **0 DOF** — it cannot move at all, no matter how many geoms it has. The notebook demonstrates this directly: the first pendulum model (no `<joint>`) has zero degrees of freedom. DOFs are added explicitly by inserting a `<joint>` element into a body, which defines *how that body is allowed to move relative to its parent*:

```xml
<body name="pendulum" euler="0 0 90">
  <joint name="swing" type="hinge" axis="1 0 0" pos="0 0 .5"/>
  ...
</body>
```

**Key distinction: body ≠ joint.** The body is the rigid link; the joint is the kinematic constraint that permits (or removes) relative motion between that link and its parent. A body with no joint is permanently fixed to its parent frame.

The notebook also introduces `<keyframe>`, a way to define a named, reusable initial state (e.g. `<key name="home" qpos="0.3"/>`) that can later be loaded with `mujoco.mj_resetDataKeyframe(model, data, 0)` instead of starting from the (often unstable) default zero configuration.

### Named Access

Rather than remembering integer indices, the Python bindings let you index model/data arrays by name:

```python
model.geom('green_sphere').rgba
```

This is a convenience wrapper around `mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'green_sphere')`, which resolves a name string to the underlying integer ID actually used inside the compiled arrays. Calling the accessor with no argument (e.g. `model.geom()`) conveniently raises a `KeyError` listing all valid names — a handy debugging trick. **Note:** body index `0` is always reserved for the special `world` body and cannot be renamed.

### `MjData`: The Mutable Simulation State

`MjData` holds the **state** (time, generalized position, generalized velocity) and **everything computed from that state** (e.g., Cartesian positions of bodies). It is created from a compiled model: `data = mujoco.MjData(model)`.

| Field | Meaning | Shape |
|---|---|---|
| `qpos` | generalized position | `(nq,)` |
| `qvel` | generalized velocity | `(nv,)` |
| `qacc` | generalized acceleration | `(nv,)` |
| `ctrl` | actuator control input | `(nu,)` |
| `xpos` | Cartesian position of each body frame | `(nbody, 3)` |
| `xipos` | Cartesian position of each body's center of mass | `(nbody, 3)` |
| `xquat` | Cartesian orientation of each body frame (quaternion) | `(nbody, 4)` |
| `xmat` | Cartesian orientation of each body frame (rotation matrix) | `(nbody, 9)` |

**Key distinction: `MjModel` vs. `MjData`.** `MjModel` is the fixed blueprint (geometry, mass, joint types, parameters); `MjData` is the live, per-simulation snapshot of everything that changes as the system evolves. You create one `MjModel` from an XML description and can then instantiate (or reset) many independent `MjData` objects from it.

**Important — derived quantities are not automatically up to date.** Fields like `data.geom_xpos` (world-frame Cartesian positions) are *functions of* `qpos`, but they are not recomputed for free every time `qpos` changes — they must be explicitly propagated. The minimal function for this in the notebook is `mujoco.mj_kinematics(model, data)`, which computes global Cartesian poses for all objects (excluding cameras/lights) from the current `qpos`. `MjData` also supports the same named-access pattern as `MjModel`, e.g. `data.geom('red_box').xpos`.

### The Simulation Loop

Conceptually, each iteration of a MuJoCo simulation does:

```
Current state (qpos, qvel)
        ↓
mujoco.mj_step(model, data)   → integrates the dynamics one timestep forward
        ↓
viewer.sync()                  → pushes the new state to the interactive viewer
        ↓
(optional) real-time pacing    → sleep until the next timestep is "due" on the wall clock
        ↓
New state
        ↺
```

`mj_step(model, data)` advances the system dynamics by one integration timestep. Conceptually it solves
$$\dot{x}_{t+h} = f(x_t)$$
where $x$ is the full state (position + velocity) and $h$ is `model.opt.timestep`: it uses the current `qpos`, `qvel`, `ctrl`, and any active contacts/forces to compute the next state.

`viewer.launch_passive(model, data)` opens a **non-blocking** interactive viewer — your Python script remains in control of timing and must call `mj_step` and `viewer.sync()` itself; mouse-drag perturbations only take effect once you call `viewer.sync()`. This is contrasted with rendering frames offscreen with `mujoco.Renderer` and saving them to video (used when `use_viewer=False` in the notebook's `run_simulation` helper) — useful for headless environments where no interactive display is available.

**Important — do not confuse rendering with simulating.** `viewer.sync()` only *displays* the current state; it does not advance physics. Physics only advances when `mj_step` is called. Skipping `mj_step` and only calling `viewer.sync()` in a loop will show a static, frozen scene.

The interactive viewer also allows applying external perturbation forces by hand (double-click a body, then Alt+Ctrl+drag for a force, or Alt+drag for a torque) — useful for qualitatively probing a system's dynamic response.

### Changing Physics Parameters

Physics options (gravity, integrator, timestep, contact on/off, energy tracking, etc.) live in `model.opt`, and can be set either **in the XML** (`<option gravity="0 0 10" integrator="RK4" timestep="0.001"/>`) or **from Python** after loading the model (`model.opt.gravity = (0, 0, 10)`). The notebook demonstrates flipping gravity and shrinking the timestep this way, and separately shows a more elaborate **chaotic double/triple pendulum** example that disables contact (`<flag contact="disable"/>`) and enables energy tracking, purely to illustrate multi-body dynamics without collision complexity.

**Notebook observation → conceptual interpretation:** the chaotic pendulum experiment is not deeply analyzed quantitatively in the notebook (no plot of divergence is produced), so no numerical conclusion about chaos should be assumed beyond the qualitative fact that this multi-link pendulum exhibits visually irregular, sensitive motion — a classic illustration of a chaotic mechanical system.

### `MjSpec`: Building Models Programmatically

As an alternative to writing/editing raw XML strings, MuJoCo exposes `mujoco.MjSpec()`, letting you construct a model directly in Python (`spec.worldbody.add_body(...)`, `body.add_joint(...)`, `body.add_geom(...)`, `spec.add_key(...)`) and then compile it with `spec.compile()`. The notebook presents this only as a brief preview ("a bit too advanced for this introduction"), showing that it can reproduce the same simple pendulum built earlier from XML. The main takeaway is *that this capability exists* for programmatically editing/generating models, not a deep dive into its API.

### Actuation and Control

So far the pendulum only moved passively under gravity. To *drive* it, the notebook introduces:

**A generic PD controller class**, independent of MuJoCo:
```python
torques = kp * (q - q_des) + kd * (0 - v)
```
This computes a desired joint torque from position error and velocity damping — pure control theory, not yet connected to MuJoCo.

**Connecting the controller to the simulation** happens by writing the computed value into `data.ctrl` before each `mj_step`. Critically, the notebook shows that **doing this alone is not enough**: without an `<actuator>` element defined in the MJCF, `data.ctrl` has no effect on the system at all, because there is nothing in the model that transmits it into a generalized force. Only after adding

```xml
<actuator>
  <motor name="my_motor" joint="swing" gear="1"/>
</actuator>
```

does writing to `data.ctrl` actually produce motion. With a `<motor>` actuator and `gear="1"`, and `model.actuator_gainprm[:, 0] = 1` (unity gain), `data.ctrl[i]` is directly interpretable as an applied joint torque in this specific configuration.

**Important — do not automatically equate `data.ctrl[i]` with joint torque.** This equivalence only holds because the notebook uses a `<motor>` actuator with gear ratio 1 and gain 1. For other actuator types (position servos, velocity servos, muscles) or different gains/gear ratios, `ctrl` represents a different physical quantity (a desired position, a desired velocity, an activation level, etc.), and MuJoCo's actuator dynamics translate it into a force/torque internally. `mjData.act` is mentioned as the vector of internal actuator activation states used by more complex actuators (e.g., muscles).

### Key Takeaways

- `MjModel` stores everything static (structure, parameters); `MjData` stores everything that evolves in time plus quantities derived from the current state.
- A body with no joint has zero DOF, no matter how much geometry it has — joints are what create motion, not bodies or geoms.
- `nq`, `nv`, and `nu` are conceptually different counters (positions, velocities/DOFs, actuators) and must not be assumed equal — free joints are the clearest example of `nq ≠ nv`.
- Derived quantities like Cartesian body poses are not automatically fresh; they require an explicit call such as `mj_kinematics` (or `mj_forward`/`mj_step`, seen implicitly through the loop).
- Advancing physics (`mj_step`) and displaying the current state (`viewer.sync()`) are two separate operations — rendering never substitutes for simulating.
- Writing to `data.ctrl` only has an effect if a matching `<actuator>` exists in the model; even then, `ctrl` is not automatically "torque" — its meaning depends entirely on the actuator's type and gain configuration.
- Physics parameters (gravity, timestep, integrator, contact enable/disable) can be changed either declaratively in the MJCF or imperatively via `model.opt` in Python.

---

## Notebook 02 — Contact Analysis

### The Core Idea: How MuJoCo Detects, Visualizes, and Quantifies Contact

Building directly on Notebook 01's model/data/loop vocabulary, this notebook isolates **contact** as a topic: how it is generated, how to visualize it, and how to read out numerical contact quantities (forces, penetration) at each timestep. The notebook notes that MuJoCo's name reflects this focus and that its contact handling is based on an **optimization-based formulation** of contact physics (referencing the original MuJoCo publication).

### Free Bodies and the Free Joint

To let an object fall and tumble freely under gravity, the notebook uses a **free joint**:

```xml
<body name="box" pos="0 0 0">
  <freejoint/>
  <geom name="red_box" type="box" size=".1 .1 .1" .../>
</body>
```

A `<freejoint/>` gives a body all 6 spatial degrees of freedom (3 translation + 3 rotation) relative to the world — this is the standard way to represent an unconstrained rigid body (e.g., an object that can be picked up, thrown, or dropped) as opposed to an articulated link connected via a hinge or slide joint. Its position is stored as `qpos[0:3]` (Cartesian position) plus `qpos[3:7]` (a unit quaternion for orientation) — this is exactly the case (flagged in Notebook 01) where a single joint contributes 7 position values but only 6 velocity values.

### Visualizing Contact

The MuJoCo viewer can render contact geometry directly, controlled through visualization flags:

```python
viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
```

along with scale parameters (`model.vis.scale.contactwidth`, `contactheight`, `forcewidth`, `model.vis.map.force`) that control how large the contact markers and force arrows appear. This is purely a visualization aid — it does not change the underlying physics, only how it is drawn.

### The Contact Pipeline

Conceptually, MuJoCo's per-step contact handling follows this pipeline (contact is one part of the larger `mj_step` computation):

```
Geometry positions (from qpos, via kinematics)
        ↓
Collision detection  (which geom pairs are close/overlapping?)
        ↓
Contact generation   (contact points, normal directions, friction cone)
        ↓
Constraint/contact solver  (optimization-based; enforces non-penetration + friction)
        ↓
Contact forces
        ↓
Generalized forces (mapped through Jacobians onto qpos/qvel space)
        ↓
Accelerations (qacc), integrated into the next qpos, qvel
```

### Reading Out Contact Data

Active contacts at the current timestep are stored in `data.contact`, an array you can iterate over. Each entry carries information such as:

| Field | Meaning |
|---|---|
| `geom1`, `geom2` | the two geoms involved in this contact |
| `friction` | friction coefficients for this contact (5 values) |
| `dist` | penetration/separation distance |

To get the actual **contact force** (not just detection), the notebook uses `mujoco.mj_contactForce(model, data, contact_index, forcetorque)`, which fills a 6-element array: the first 3 components are the **contact-frame linear force** (index 0 = normal direction, indices 1–2 = the two tangential/friction directions), and the last 3 are torque components. **Important — this force is expressed in the local contact frame, not the world frame** (normal along the contact normal, not necessarily along a world axis); this is why the notebook's plots label the components "normal z / friction x / friction y" rather than world X/Y/Z.

`data.ncon` gives the current number of active contacts — useful for tracking how many contact points exist at any instant (e.g., a box resting flat may have 4 corner contacts; a box balanced on an edge might have 2).

### The Experiment: Random Orientation and Penetration/Force Analysis

The notebook drops a box with a randomized initial orientation (`data.qpos[3:7]` set to a random unit quaternion) and plots, over time: the contact force components, the number of contacts, the normal force on a log scale (compared against the box's theoretical weight $mg$, computed from `model.body("box").mass[0]` and `model.opt.gravity`), and the penetration depth.

**What changed:** orientation of the falling box is randomized on each reset. **What is observed:** contact forces spike briefly on impact, and at rest the normal contact force should balance the box's weight ($F_z \approx mg$) — this is the physical justification for plotting the `mg` reference line for comparison. **Why:** at equilibrium, Newton's second law requires the net force on the box to vanish, so the (static) contact normal force must support gravity. **Robotics principle demonstrated:** this is a simple, concrete instance of how a contact constraint force resolves an external load — the same principle that underlies robot foot/ground contact or gripper/object contact in more complex systems.

**Notebook observation → conceptual interpretation on penetration:** the notebook plots "penetration depth (mm)" over time. MuJoCo's contact model is *soft* by default (governed by `solref`/`solimp` parameters set in the `<geom>` definitions, e.g. `solimp=".99 .99 .01" solref=".001 1"`), meaning bodies are permitted a small amount of interpenetration that is resisted like a stiff spring-damper rather than being enforced as a perfectly rigid, zero-penetration constraint. The notebook does not derive the numeric relationship between `solref`/`solimp` and the resulting penetration depth — it only visualizes the resulting penetration trace, so no quantitative claim about their relationship should be assumed beyond "some small, non-zero penetration is expected and is a normal feature of this compliant contact formulation," which the plot is used to confirm qualitatively.

### The Friction Experiment: Sliding vs. Tipping

The notebook sets up a box with a nonzero initial horizontal velocity (`data.qvel[1] = 3`) sliding onto the ground, and changes the geom's `friction` attribute.

**What changed:** the friction coefficient parameter on the box geom (`FRICTION = 2` vs. a low value like `0.1`).
**What was observed (per the notebook's own description):** with high friction (`FRICTION = 2`), the box tends to **tip** rather than slide; with low friction (`FRICTION = 0.1`), the box **slides** across the floor instead.
**Why this happens:** friction at the contact point resists horizontal sliding. If friction is high enough to arrest the contact point's horizontal motion quickly, the box's momentum above the contact point creates a **torque about the contact edge**, causing it to rotate (tip) rather than translate. If friction is low, the contact point can slip freely, and momentum is dissipated as translational sliding instead.
**Robotics principle demonstrated:** this is a hands-on illustration of the coupling between **friction coefficient and rotational vs. translational response** at a contact — directly relevant to any legged or manipulation task where whether an object slides or tips/rolls under a pushing force depends critically on the friction available at the contact.

### Key Takeaways

- A `<freejoint/>` is the standard way to give a rigid body full 6-DOF unconstrained motion; its `qpos` block is 7-dimensional (3 position + 4-quaternion orientation) while its `qvel` block is 6-dimensional (3 linear + 3 angular velocity) — a concrete instance of `nq ≠ nv`.
- Contact visualization (`mjVIS_CONTACTPOINT`, `mjVIS_CONTACTFORCE`) is purely a rendering aid and does not alter physics.
- Contact happens in a pipeline: collision detection → contact generation → constraint/contact solver → forces → generalized forces → accelerations; `data.contact` exposes the intermediate per-contact data, and `mj_contactForce` extracts the actual solved force for a given contact.
- Contact force components from `mj_contactForce` are expressed in the **local contact frame** (normal + two friction directions), not the world frame.
- MuJoCo's default contact model is **compliant/soft** (governed by `solref`/`solimp`), permitting small interpenetration rather than enforcing perfectly rigid non-penetration — this is why a nonzero penetration depth is normal and expected, not a bug.
- **Do not confuse contact detection with contact force computation** — detecting that two geoms are touching (`data.contact`, `data.ncon`) is a separate step from computing the actual resolved force at that contact (`mj_contactForce`).
- Friction magnitude qualitatively determines whether a sliding object tips (high friction) or continues sliding (low friction) — a direct, tunable consequence of the friction coefficient parameter in the geom definition.

---

## Notebook 03 — 3D Finger EDU

### The Core Idea: Applying Model/Control/Contact Concepts to a Real Robot

This notebook is explicitly framed as an integration exercise: "use what you learned in the previous two notebooks" to (1) load a real robot description, (2) drive it with a PD controller to a target pose, (3) analyze its transient behavior via plots, and (4) make it strike an object and analyze the resulting contact — directly reusing the `PDController` pattern from Notebook 01 and the `mj_contactForce`/plotting pattern from Notebook 02, now applied to a multi-DOF articulated arm instead of a single pendulum or a single free box.

### Loading a Real Robot Description

The robot is the **3D Finger EDU**, a 3-DOF robot arm from the Open Dynamic Robot Initiative hardware family. Its original description is a **URDF** file, which is not natively compatible with MuJoCo's MJCF format — the notebook notes that the description has already been converted for the tutorial (with a documentation link on how such URDF→MJCF conversions are generally done).

```python
model = mujoco.MjModel.from_xml_path(MODEL_FILE)  # MODEL_FILE = ".../finger_edu_scene.xml"
```

The scene file adds a floor to the robot by using `<include file="finger_edu.xml"/>`, which imports the bare robot model (no floor) without duplicating its contents — a modular pattern for composing scenes out of reusable model fragments, useful whenever a robot description needs to be reused across multiple different scenes (e.g., with vs. without a manipulated object).

### Generalizing the PD Controller to Multiple DOFs

Notebook 01's `PDController` is extended (`self.n_act`) to handle a **vector** of desired joint positions instead of a scalar, so the same class can drive all three joints of the finger simultaneously:

```python
torques = kp * (q_des - q[:n_act]) + kd * (0 - v)[:n_act]
```

The simulation loop pattern is identical in structure to Notebook 01/02: read `q, v` from `data.qpos, data.qvel`, compute `torque` from the controller, write it into `data.ctrl` (truncated to `model.nu` since there may be more DOFs in the model than actuated ones), then call `mj_step` and `viewer.sync()`.

**Important — a multi-joint robot's `nv` is not automatically equal to `nu`.** The loop explicitly slices `torque[:n_act]` before assigning to `data.ctrl`, which only makes sense if the number of actuators (`nu`) can differ from the number of DOFs (`nv`) being commanded by the raw PD law — reinforcing the Notebook 01 warning against assuming these counts coincide.

### Logging and Plotting Trajectories

To diagnose controller behavior beyond what's visible in the 3D viewer, the notebook collects `qpos`, `qvel`, `ctrl`, and `time` into a `sim_data` dictionary at every simulation step, then plots joint positions, velocities, and control inputs over time in three stacked subplots. This is a general and important pattern: **the interactive viewer is good for qualitative intuition, but quantitative analysis of transient behavior (overshoot, oscillation, settling time) requires logging state over time and plotting it**, exactly as the contact-force plots did in Notebook 02.

### Experiment: Gain Tuning to Reduce Oscillation

After first driving the arm to a target pose with one set of gains and observing (per the notebook's own description) "quite some oscillations in the motion," the notebook asks the reader to **retune the derivative gain `KD`** and re-plot the trajectories to check whether oscillations are reduced.

**What changed:** the derivative gain `KD` of the PD controller. **What the notebook expects you to observe:** by increasing damping (`KD`) relative to the proportional gain (`KP`), the oscillatory response should be reduced. **Why:** in a PD control law, the proportional term (`KP`) drives the system toward the target but, on its own, tends to produce oscillatory (underdamped) convergence in a system with inertia; the derivative term (`KD`) opposes velocity and dissipates energy, moving the response toward critical damping. **Robotics principle demonstrated:** this is the same underdamped/overdamped/critically-damped trade-off familiar from general PD/PID control theory, now made tangible on a physical robot arm in simulation.

**Notebook observation → conceptual interpretation:** because `KD` (and the target `Q_DES`) are left as exercises for the reader to fill in and tune, the notebook does not itself provide a specific numeric gain value or a specific resulting plot to report — only the qualitative goal ("reduce oscillations") and the mechanism (increase `KD`). No specific tuned value should be treated as "the" answer.

### Adding a Manipulated Object and Filtering Contacts

To study contact between the arm and an external object, the notebook has the reader **extend the scene file** with a new free body (a small cube, mass `0.01`, with a `<freejoint/>`) placed near the robot's workspace, saved as a separate scene XML (`finger_edu_scene_cube.xml`) that includes the floor and robot as before.

Because the cube can be in contact with **both the floor and the robot end-effector simultaneously**, a naive sum over `data.contact` would conflate "cube resting on floor" contacts with "robot pushing cube" contacts. The notebook's solution is to:

1. Resolve the floor geom's integer ID once: `floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_geom_name)` (following the exact pattern introduced in Notebook 01's discussion of `mj_name2id`).
2. Extend `plot_contacts` with a `discard_contacts_id` list, and for each entry in `data.contact`, **skip** (via `continue`) any contact where either `geom1` or `geom2` is in that discard list.

This is a directly reusable pattern: **whenever more than two objects can touch each other in a scene, contacts must be explicitly filtered by geom identity to isolate the interaction of interest**, since `data.contact`/`data.ncon` report *all* active contacts in the scene indiscriminately, not just the ones you care about.

### Analyzing the Robot–Cube Contact

With the floor contacts filtered out, the same force/penetration/`ncon` plotting machinery from Notebook 02 is reused to visualize only the robot-arm-to-cube interaction as the PD-controlled arm swings into the cube. The notebook's final question — to read off the maximum penetration distance and maximum force from the resulting plot — is left as a reader exercise; **the notebook provides the tooling to measure these quantities but does not itself report specific numeric results**, so no particular peak-force or peak-penetration value should be assumed here.

### Key Takeaways

- Real robot descriptions are often authored as URDF and must be converted to MJCF (MuJoCo's native format) before they can be simulated; scene composition via `<include file="...">` lets a bare robot model be reused across multiple scene variants (e.g., with or without additional objects).
- The `PDController`/simulation-loop pattern from Notebook 01 generalizes directly from a single scalar joint to a vector of joints — the control law and loop structure do not fundamentally change, only the dimensionality of `q`, `v`, and `ctrl`.
- The interactive viewer gives qualitative intuition; **logging state (`qpos`, `qvel`, `ctrl`, `time`) and plotting it** is necessary to diagnose quantitative transient behavior like oscillation.
- Increasing a PD controller's derivative gain (`KD`) generally increases damping and reduces oscillatory overshoot — a direct, tunable trade-off between responsiveness and smoothness.
- When more than two bodies can be in contact in a scene, `data.contact` reports *all* pairwise contacts indiscriminately — isolating a specific interaction (e.g., robot-vs-object, ignoring object-vs-floor) requires explicitly filtering by `geom` ID.
- This notebook is fundamentally an *integration* exercise: it does not introduce major new MuJoCo concepts beyond scene composition and contact filtering, but demonstrates that the model/data/control/contact vocabulary from Notebooks 01–02 scales directly to a real, multi-DOF robot.

---

## Summary: The Mental Stack

```
┌───────────────────────────────────────────────────────────┐
│ Notebook 03 — 3D Finger EDU                                │
│ Real robot (URDF→MJCF), multi-DOF PD control, trajectory   │
│ logging/plotting, contact filtering by geom ID              │
│ ↓ depends on                                                │
├───────────────────────────────────────────────────────────┤
│ Notebook 02 — Contact Analysis                              │
│ Free joints, contact visualization, data.contact,           │
│ mj_contactForce, compliant contact (solref/solimp),         │
│ friction → sliding vs. tipping                               │
│ ↓ depends on                                                │
├───────────────────────────────────────────────────────────┤
│ Notebook 01 — Introduction to MuJoCo                        │
│ MJCF, MjModel vs. MjData, nq/nv/nu, named access,            │
│ mj_step simulation loop, actuators vs. joints, ctrl,         │
│ model.opt physics parameters, MjSpec                        │
└───────────────────────────────────────────────────────────┘
```

Every later notebook's code is literally built by copy-and-extending the simulation-loop and controller functions defined earlier: Notebook 02's `sim_viewer`/`plot_contacts` are direct descendants of Notebook 01's `run_simulation`; Notebook 03 explicitly comments "Import and same functions as in the last notebooks" before redefining its own `sim_viewer`/`PDController`. The conceptual stack mirrors this: **model description → state representation → the step/loop mechanism → control → contact**, each layer assuming fluency in the one below it.

---

## Quick-Reference: Common Pitfalls

| Pitfall | Why It Is Wrong | Correct Approach |
|---|---|---|
| Assuming a body has DOF just because it has geoms | DOF comes only from joints; a body with no `<joint>` is rigidly fixed to its parent | Explicitly add a `<joint>` (hinge, slide, free, etc.) to any body that should move |
| Assuming `nq == nv == nu` | A free joint alone makes `nq` (7, incl. quaternion) ≠ `nv` (6); actuator count `nu` is independent of both | Always check `model.nq`, `model.nv`, `model.nu` separately for a given model |
| Writing to `data.ctrl` and expecting motion | `ctrl` only has an effect if a matching `<actuator>` element exists in the model | Add an `<actuator>` (e.g. `<motor>`) bound to the relevant joint before relying on `ctrl` |
| Treating `data.ctrl[i]` as always being a torque | Its physical meaning depends on the actuator type/gain (position, velocity, torque, muscle activation, ...) | Check the actuator definition and gain parameters before interpreting `ctrl` values |
| Calling only `viewer.sync()` in a loop without `mj_step` | `viewer.sync()` renders the current state; it never advances the physics | Always call `mujoco.mj_step(model, data)` to progress simulation time |
| Reading `data.geom_xpos`/`data.xpos` right after changing `qpos` without recomputing | Derived (Cartesian) quantities are not refreshed automatically | Call `mj_kinematics` (or `mj_forward`/`mj_step`) to propagate `qpos` into derived quantities |
| Expecting zero penetration between contacting bodies | MuJoCo's default contact model is compliant/soft (`solref`/`solimp`), not perfectly rigid | Treat small penetration as expected; adjust `solref`/`solimp` only if a different softness is needed |
| Summing all of `data.contact` when multiple object pairs can touch | This conflates unrelated contacts (e.g., object-vs-floor with robot-vs-object) | Filter contacts by checking `geom1`/`geom2` against the specific geom IDs of interest |
| Interpreting `mj_contactForce` output as a world-frame force vector | The force is expressed in the local **contact frame** (normal + 2 friction directions) | Remember index 0 = normal, indices 1–2 = tangential/friction components, in the contact's local frame |

---

## Final Cheat Sheet

### Core Objects
- **`MjModel`** — the compiled, static description of the mechanical system (structure, geometry, mass, joint types, actuator setup, physics options). Created via `mujoco.MjModel.from_xml_string(...)` / `from_xml_path(...)`.
- **`MjData`** — the mutable simulation state and everything computed from it (time, positions, velocities, forces, contacts). Created via `mujoco.MjData(model)`.
- **`MjSpec`** — a programmatic (Python-side) way to build/edit a model before compiling it with `spec.compile()`.

### Model Structure
- **Body** — a rigid, potentially movable object with mass/inertia; the entity joints attach to and connect.
- **Joint** — defines the allowed relative motion (DOF) between a body and its parent; without one, the body is rigidly fixed.
- **Geom** — the visual and collision shape attached to a body (a body can carry multiple geoms).
- **`<freejoint/>`** — gives a body full unconstrained 6-DOF motion (3 translation + 3 rotation); position stored as `qpos[0:3]` + quaternion `qpos[3:7]`, velocity as `qvel[0:6]`.
- **`<actuator>`** (e.g. `<motor>`) — connects a control input (`data.ctrl`) to a joint; without it, `ctrl` has no physical effect.
- **`<keyframe>`** — a named, reusable initial state, loaded via `mj_resetDataKeyframe`.

### State
- `data.qpos` — generalized position, shape `(nq,)`.
- `data.qvel` — generalized velocity, shape `(nv,)`.
- `data.qacc` — generalized acceleration, shape `(nv,)`.
- `data.ctrl` — actuator control input, shape `(nu,)`.
- `data.time` — current simulation time.
- `model.nq`, `model.nv`, `model.nu` — counts of position vars, velocity DOFs, and actuators (generally not equal).

### Kinematics
- `data.xpos`, `data.xipos` — Cartesian position of each body frame / center of mass, shape `(nbody, 3)`.
- `data.xquat`, `data.xmat` — Cartesian orientation of each body frame, as quaternion `(nbody, 4)` or rotation matrix `(nbody, 9)`.
- `data.geom_xpos` — world-frame position of each geom.
- `mujoco.mj_kinematics(model, data)` — propagates `qpos` into all Cartesian pose quantities.
- `mujoco.mj_name2id(model, objtype, name)` — resolves a name string to its integer ID; underlies all named accessors like `model.geom('name')`.

### Dynamics
- `mujoco.mj_step(model, data)` — advances the full dynamics (including contact) by one `model.opt.timestep`.
- `model.opt` — physics options: `timestep`, `gravity`, `integrator`, contact/energy flags, etc. — settable in XML or from Python.

### Contacts
- `data.contact` — array of currently active contacts (`geom1`, `geom2`, `friction`, `dist`, ...).
- `data.ncon` — number of currently active contacts.
- `mujoco.mj_contactForce(model, data, i, forcetorque)` — fills a 6-vector with the resolved local-frame contact force (0: normal, 1–2: friction) and torque for contact `i`.
- `solref` / `solimp` (geom attributes) — govern the softness/compliance of the contact constraint (how much penetration is tolerated and how it's resisted).
- `mjVIS_CONTACTPOINT`, `mjVIS_CONTACTFORCE` — viewer flags to visualize contact points and force vectors (visualization only, no physical effect).

### Control
- `PDController` pattern: `torque = kp * (q_des - q) + kd * (0 - v)` — a minimal proportional-derivative law used throughout the tutorial.
- `data.act` — internal activation state vector for actuators with their own dynamics (e.g., muscles); not used directly by the simple `<motor>` actuators in this tutorial.

### Simulation
- `viewer.launch_passive(model, data)` — opens a non-blocking interactive viewer; your script drives timing and must call `mj_step`/`viewer.sync()` itself.
- `viewer.sync()` — pushes the current `MjData` state to the viewer for display; does **not** advance physics.
- `mujoco.Renderer(model, H, W)` + `renderer.update_scene(data)` / `renderer.render()` — offscreen rendering to frames (e.g., for saving video), used as an alternative to the interactive viewer.

### Useful Functions
- `mujoco.MjModel.from_xml_string(xml_str)` / `from_xml_path(path)` — compile a model from MJCF.
- `mujoco.mj_resetData(model, data)` / `mujoco.mj_resetDataKeyframe(model, data, key_id)` — reset simulation state to zero or to a named keyframe.
- `mujoco.mj_name2id(model, objtype, name)` — name → integer ID lookup.
