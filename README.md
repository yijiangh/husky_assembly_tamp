# TAPOM: Task-Space Topology-Guided Motion Planning for Manipulating Elongated Objects in Cluttered Environments

## Overview
TAPOM (Task-Space Topology-Guided Motion Planning) is a hierarchical motion planning framework designed for robotic manipulation of elongated objects in cluttered and constrained environments. It leverages task-space topology analysis to navigate narrow passages efficiently, addressing challenges where traditional planners struggle due to sampling inefficiencies or local minima.

![TAPOM Framework](images/depict.png)  
![results](images/results.png)  
![results main](images/results_main.png)

## Key Features
- **Topology-Aware High-Level Planning**: Analyzes task-space topology to model free space connectivity and identify critical regions, enabling effective path planning through narrow passages.
- **Keyframe-Guided Low-Level Planning**: Combines topological insights with a sampling-based planner to generate collision-free trajectories in high-dimensional configuration spaces.
- **Experimental Validation**: Outperforms state-of-the-art baselines in success rate and planning time across simulated and real-world tasks involving elongated objects.

## Repository Structure
- `scripts/`: Source code for the TAPOM planner implementation.
- `ext/`: Dependencies of TAPOM.
- `data/`: Environment models used in experiments.

<!-- ## Installation (Unfinished)

* create virtual env

```bash
conda create -n husky_assembly python==3.9.19
conda activate husky_assembly
```

* install python dependencies

```bash
conda install pinocchio -c conda-forge
pip install numpy pybullet_planning torch matplotlib casadi compas_fab
```

* install curobo https://curobo.org/get_started/1_install_instructions.html

* install OMPL with python bindings https://ompl.kavrakilab.org/core/installation.html -->

## Installation

### Requirements
- **Operating System**: Linux (tested on Ubuntu 20.04)
- **Software**:
  - Python 3.9.19
  - Pinocchio (via conda-forge)
  - NumPy, PyBullet Planning, Torch, Matplotlib, CasADi, COMPAS FAB
  - cuRobo (follow installation instructions at [cuRobo Documentation](https://curobo.org/get_started/1_install_instructions.html))
  - OMPL with Python bindings (version 1.5.2 or later, see [OMPL Installation](https://ompl.kavrakilab.org/core/installation.html))
- **Player Requirements**: No special player is required as the code runs in a Python environment.

### Installation Steps

1. **Create a virtual environment**:
   ```bash
   conda create -n husky_assembly python==3.9.19
   conda activate husky_assembly
   ```

2. **Install Python dependencies**:
   ```bash
   conda install pinocchio -c conda-forge
   pip install numpy pybullet_planning torch matplotlib casadi compas_fab
   ```

3. **Install cuRobo**:
   - Follow the instructions at [cuRobo Documentation](https://curobo.org/get_started/1_install_instructions.html).

4. **Install OMPL with Python bindings**:
   - Follow the instructions at [OMPL Installation](https://ompl.kavrakilab.org/core/installation.html).

## How to Run a Basic Example

To run a basic example of TAPOM, follow these steps:

1. **Navigate to the TAPOM directory**

2. **Run the example script**:
   ```bash
   bash generate_data.sh
   ```
   - This command runs TAPOM on the Rebar Assembly (RA) scenario using the ABB IRB 4600 robot.

3. **Output**:
   - The planner will output a collision-free trajectory if successful.
   - Logs and visualizations will be saved in the `scripts/logs/` directory.

## OMPL and cuRobo Comparison

For reviewers and users interested in the hyperparameters and comparison with OMPL and cuRobo baselines, please refer to the following scripts:

- **OMPL Comparison**:
  - Script and Hyperparameters: `scripts/motion_planner/trajectory_ompl_solver.py`

- **cuRobo Comparison**:
  - Script and Hyperparameters: `scripts/motion_planner/trajectory_curobo_solver.py`

The script `scripts/test/corner_case_for_transfer.py` calls these classes and uses them.

These scripts allow replication of the experiments presented in the paper, including the specific parameter settings used for each planner.

<!-- ## Contact Information
For questions or technical support regarding the multimedia material, please contact:

- **Email**: [your.email@example.com](mailto:your.email@example.com)
- **GitHub Issues**: Open an issue on this repository for code-related queries.

**Note**: IEEE does not provide technical support for this material. -->