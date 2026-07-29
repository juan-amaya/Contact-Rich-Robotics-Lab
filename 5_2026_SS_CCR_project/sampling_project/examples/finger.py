import argparse

import mujoco
import numpy as np

from hydrax.algs import MPPI, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.finger import Finger

"""
Run an interactive simulation of the finger push task.
"""

# TODO: Define the cost function inside the class
task = Finger()

#TODO: Set up the controller
crtl = None

# Define the model used for simulation
mj_model = task.mj_model
mj_model.opt.timestep = 0.001
mj_model.opt.iterations = 100
mj_model.opt.ls_iterations = 50

mj_data = mujoco.MjData(mj_model)
mj_data.qpos = [0, 0, -0.15, 0.1, -0.05, 0.05, 1, 0, 0, 0]

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=100,
    show_traces=False,
)