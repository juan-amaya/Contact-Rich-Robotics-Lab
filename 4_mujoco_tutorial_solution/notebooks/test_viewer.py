import mujoco
import mujoco.viewer

import time

sphere = """
<mujoco>

  <asset>
    <texture name="grid" type="2d" builtin="checker"
             rgb1="0.2 0.3 0.4"
             rgb2="0.1 0.1 0.1"
             width="512" height="512"/>
    <material name="floor_mat"
              texture="grid"
              texrepeat="10 10"
              texuniform="true"/>
  </asset>

  <visual>
    <headlight ambient="0.4 0.4 0.4"
               diffuse="0.8 0.8 0.8"
               specular="0.2 0.2 0.2"/>
  </visual>

  <worldbody>

    <!-- Floor -->
    <geom name="floor"
          type="plane"
          pos="0 0 0"
          size="5 5 0.1"
          material="floor_mat"/>

    <body name="sphere">
      <freejoint/>
      <geom name="green_sphere"
            type="sphere"
            pos="0 0 0.5"
            size="0.1"
            rgba="0 1 0 1"/>

      <!-- Camera attached to sphere -->
      <camera name="track"
              pos="-2 0 1"
              xyaxes="0 1 0 -0.4 0 1"/>
    </body>

  </worldbody>

</mujoco>
"""

model = mujoco.MjModel.from_xml_string(sphere)
data = mujoco.MjData(model)

SIM_TIME = 5

viewer = mujoco.viewer.launch_passive(model, data)

while (
    viewer.is_running() and
    data.time < SIM_TIME
):
    step_start = time.time()

    # Render new state in the viewer
    viewer.sync() # TODO

    # Compute forwards dynamics
    mujoco.mj_step(model, data) # TODO

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
        time.sleep(time_until_next_step)
	
viewer.close()