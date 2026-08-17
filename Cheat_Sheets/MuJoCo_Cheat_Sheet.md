# MuJoCo Cheat Sheet
### Practical reference for future projects

---

## Setup / Imports

```python
import mujoco
import mujoco.viewer   # separate import, not bundled in `mujoco`
import time
import numpy as np
```

| Gotcha | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` (via `utils.py`) | `uv sync` inside the project, and make sure the notebook kernel points at `.venv` (check with `import sys; print(sys.executable)`) |
| Any import fails silently breaking a cell you don't even use | Unused imports at the top of a file still execute — comment out imports you don't need if debugging |

---

## Loading a Model & Creating Data

```python
model = mujoco.MjModel.from_xml_string(xml_str)   # from an XML string
model = mujoco.MjModel.from_xml_path("robot.xml")  # from a file
data  = mujoco.MjData(model)
```

- `MjModel` = static (structure, params). `MjData` = dynamic (state), always built **from** a model.
- Re-run `MjData(model)` any time you want to reset from scratch after editing `model`.

---

## Model Inspection (static — `MjModel`)

```python
model.nq, model.nv, model.nu     # positions / velocity DOFs / actuators — check separately, never assume equal
model.nbody, model.ngeom         # counts
model.opt.timestep               # integration timestep (default 0.002s)
model.opt.gravity                # gravity vector, settable: model.opt.gravity = (0,0,10)
model.body("box").mass[0]        # named access to model fields
model.geom("blue_rod").rgba      # rgba array [r,g,b,alpha] — alpha = transparency (0=invisible,1=opaque)
model.geom_rgba                  # raw array (ngeom, 4), if indexing by int ID
```

```python
for i in range(model.ngeom):
    print(i, model.geom(i).name)
```

---

## Data / State Access (dynamic — `MjData`)

```python
data.time                 # current sim time
data.qpos                 # (nq,)  generalized position — angle directly for a hinge joint
data.qvel                 # (nv,)  generalized velocity
data.qacc                 # (nv,)  generalized acceleration
data.ctrl                 # (nu,)  actuator input — needs a matching <actuator> in the model or it's a no-op
data.xpos, data.xipos      # (nbody,3) Cartesian body-frame / COM position
data.xquat, data.xmat      # (nbody,4)/(nbody,9) body orientation
data.geom("red_box").xpos # named access, Cartesian
data.joint("swing").qpos  # named access, safer than raw index if joints could be reordered
```

> **`IndexError: index 0 is out of bounds`** on `qpos[0]` → `model.nq == 0`, i.e. the loaded model has no joints. Print `model.nq` to check before indexing.

**Propagating state → Cartesian quantities (not automatic!):**
```python
mujoco.mj_kinematics(model, data)   # minimal: updates xpos/xmat/etc. from qpos
mujoco.mj_forward(model, data)      # fuller: kinematics + dynamics quantities, no integration
```

---

## Named Access / ID Lookup

```python
mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "green_sphere")
mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
```
- `model.geom("name")` / `data.geom("name")` are convenience wrappers around this.
- Calling `model.geom()` with no name → raises `KeyError` listing all valid names (handy for debugging typos).
- Body `0` is always `world`.

---

## Simulation Loop (canonical pattern)

```python
while viewer.is_running() and data.time < sim_time:
    step_start = time.time()

    mujoco.mj_step(model, data)     # advances physics by one model.opt.timestep
    viewer.sync()                    # DISPLAY ONLY — never advances physics on its own

    dt = model.opt.timestep - (time.time() - step_start)
    if dt > 0:
        time.sleep(dt)
```
- Always pair `mj_step` with `viewer.sync()` — a loop with only `sync()` will look frozen.
- `mj_resetData(model, data)` / `mujoco.mj_resetDataKeyframe(model, data, key_id)` — reset state (to zero, or to a named `<keyframe>`).

---

## Control / Actuators

```python
# Minimal PD law
torque = kp * (q_des - q) + kd * (0 - v)
data.ctrl[:] = torque   # or data.ctrl = torque[:model.nu] if vector length differs
```

| Rule | Detail |
|---|---|
| `data.ctrl` requires an `<actuator>` | e.g. `<motor name="m" joint="swing" gear="1"/>` — without it, `ctrl` does nothing |
| `ctrl[i]` ≠ torque in general | Only ≈ torque with a `<motor>`, `gear=1`, and `model.actuator_gainprm[:,0]=1` |
| `nu` ≠ `nv` in general | Slice explicitly: `data.ctrl = torque[:model.nu]` |
| Log, don't just watch | Collect `qpos`/`qvel`/`ctrl`/`time` into arrays each step and plot with `matplotlib` for real tuning (viewer alone is qualitative only) |

---

## Contact Analysis

```python
data.contact[i].geom1, data.contact[i].geom2   # which geoms touch
data.contact[i].dist                            # penetration/separation distance
data.ncon                                       # number of active contacts

forcetorque = np.zeros(6)
mujoco.mj_contactForce(model, data, i, forcetorque)
# forcetorque[0]   = normal force
# forcetorque[1:3] = friction (tangential), LOCAL contact frame, not world XYZ
# forcetorque[3:6] = torque
```

**Filtering contacts when >2 objects can touch** (e.g. object touches both floor and robot):
```python
floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
for c in data.contact:
    if c.geom1 == floor_id or c.geom2 == floor_id:
        continue   # skip floor contacts, keep only the pair you care about
```

**Contact softness (XML geom attributes):** `solref`, `solimp` — governs how much penetration is tolerated (default is compliant, not perfectly rigid — small penetration is normal).

---

## Viewer & Visualization Flags

```python
viewer = mujoco.viewer.launch_passive(model, data)   # non-blocking — you drive mj_step/sync yourself
viewer = mujoco.viewer.launch_passive(model, data, key_callback=my_callback)  # for keyboard interaction

viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True   # force ALL geoms translucent (viz aid, not model.rgba)

model.vis.scale.contactwidth = 0.1
model.vis.scale.contactheight = 0.03
model.vis.scale.forcewidth = 0.05
model.vis.map.force = 0.3

viewer.close()
```

**Offscreen rendering (headless / video, no display needed):**
```python
renderer = mujoco.Renderer(model, height, width)
renderer.update_scene(data)
pixels = renderer.render()   # RGB array — convert to BGR before cv2.VideoWriter
```

---

## MJCF Quick Reference

```xml
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
  <worldbody>
    <body name="box" pos="0 0 0">
      <freejoint/>                                  <!-- 6-DOF: qpos 7-dim (incl. quat), qvel 6-dim -->
      <joint name="swing" type="hinge" axis="1 0 0" pos="0 0 .5"/>  <!-- 1-DOF: qpos/qvel both 1-dim -->
      <geom name="g" type="box" size=".1 .1 .1" rgba="1 0 0 1" friction="1" solref=".001 1" solimp=".99 .99 .01"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="m" joint="swing" gear="1"/>
  </actuator>
  <keyframe>
    <key name="home" qpos="0.3"/>
  </keyframe>
</mujoco>
```

| Element | Meaning |
|---|---|
| `<body>` | movable, inertial entity — **no joint = 0 DOF, welded to parent** |
| `<geom>` | visual + collision shape attached to a body (≠ body) |
| `<joint>` | adds DOF; `hinge`/`slide` = 1 DOF, `<freejoint/>` = 6 DOF |
| `<actuator>` | required for `data.ctrl` to do anything |
| `<keyframe>` | named reusable initial state, load via `mj_resetDataKeyframe` |
| `<include file="...">` | reuse a model fragment (e.g. robot) across multiple scene files |

---

## Debugging Checklist

1. `print(model.nq, model.nv, model.nu)` — sanity-check DOF/actuator counts before indexing anything.
2. Robot not moving under `ctrl`? → confirm an `<actuator>` exists and is bound to the right joint.
3. Cartesian pose (`xpos`, `geom_xpos`) looks stale? → call `mj_kinematics`/`mj_forward`/`mj_step` after changing `qpos`.
4. Viewer frozen? → confirm `mj_step` is actually being called in the loop, not just `viewer.sync()`.
5. Wrong/duplicate contacts in a multi-object scene? → filter `data.contact` by `geom1`/`geom2` ID.
6. `IndexError` on `qpos`/`qvel`? → the loaded model has fewer DOFs than expected — re-check which XML was compiled.
