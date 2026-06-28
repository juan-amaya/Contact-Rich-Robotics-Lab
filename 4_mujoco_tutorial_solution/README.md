# Robotic course, XXXX, 20YY

[MuJoCo](https://mujoco.readthedocs.io/en/stable/overview.html) Python tutorial for the contact rich robotics class 2026.
The tutorial is organized in different notebooks that each introduce different concepts of MuJoCo.

## Getting started

### Setup Instructions

- Install uv, a fast Python package and environment manager. Installation instructions can be found at: https://docs.astral.sh/uv/getting-started/installation/

- Create a virtual environment and install the project dependencies:

```
uv venv
source .venv/bin/activate
uv sync
```

- Run a test for the Mujoco viewer, **this test needs to work to do the tutorial**. 

```
uv run notebooks/test_viewer.py
```

- If the viewer doesn't show up, try installing ffmpeg for rendering, try looking up online for solutions.
```
sudo apt install -y ffmpeg
```

### Run a notebook

Run jupyter-lab (in the virtual environment) and enter the jupyter server adress (with token).

```bash
jupyter-lab
```

Or use the Jupyter VS-Code extension and select the right python environment.
