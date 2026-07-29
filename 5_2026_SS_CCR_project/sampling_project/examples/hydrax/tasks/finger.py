import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class Finger(Task):
    """A 3DoF finger box push task"""

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/finger/finger_edu_scene_cube.xml"
        )
        super().__init__(mj_model)

        # Get sensor ids
        self.block_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "position"
        )
        self.block_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation"
        )

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        #TODO: Write your own running cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ℓ_T(x_T)."""
        #TODO: Write your own terminal cost
