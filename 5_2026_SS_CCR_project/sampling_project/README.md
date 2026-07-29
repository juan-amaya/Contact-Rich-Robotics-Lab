# Contact-Rich Robotics Course, SoSe, 2025

This directory provides the environment and scripts that will allow you to complete the sampling part of the final project for your Contact-Rich Robotics course. The aim of the project is to push a box to a desired location using a fixed base 3-DOF finger robot.

The part of your project was created by Ilyass Taouil, PhD student at Prof. Majid's chair. In case you run into any issues, feel free to contact him at [ilyass.taouil@tum.de](mailto:ilyass.taouil@tum.de).

## Getting started

### Setup Instructions
- Install Miniconda (if already not installed), a minimal conda installer, allowing us to create virtual environments easily. The installation instructions can be found [here](https://docs.anaconda.com/free/miniconda/).
- Create a new virtual environment with the necessary packages already defined in `environment.yml` and activate it using the following commands:

  ```bash
  conda env create -f environment.yml
  conda activate sampling_mujoco_playground
  ```

- Install the dependencies
  ```bash
  pip install -e .
  ```

## Project Instructions
To complete this project, you will mostly only need to complete the following files (starting from the folder root)

1. ./examples/finger.py (implement the controller -> MPPI/CEM/PS)
2. ./hydrax/tasks/finger.py (implement your cost function & sensors)

### Playing the finger robot env
To test your implementation you can run the sampling environment with the following command line from the folder root:

```bash
python examples/finger.py
```