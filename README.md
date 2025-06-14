# TAPOM: Task-Space Topology-Guided Motion Planning for Manipulating Elongated Objects in Cluttered Environments

## Overview
TAPOM (Task-Space Topology-Guided Motion Planning) is a hierarchical motion planning framework designed for robotic manipulation of elongated objects in cluttered and constrained environments. It leverages task-space topology analysis to navigate narrow passages efficiently, addressing challenges where traditional planners struggle due to sampling inefficiencies or local minima.

![TAPOM Framework](images/depict.png)
![results](images/results.png)
![results main](images/results_main.png)

<iframe width="560" height="315" src="https://www.youtube.com/embed/G7N8VYgaefw?si=7fkkx5nE2R8jrAa_" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Key Features
- **Topology-Aware High-Level Planning**: Analyzes task-space topology to model free space connectivity and identify critical regions, enabling effective path planning through narrow passages.
- **Keyframe-Guided Low-Level Planning**: Combines topological insights with a sampling-based planner to generate collision-free trajectories in high-dimensional configuration spaces.
- **Experimental Validation**: Outperforms state-of-the-art baselines in success rate and planning time across simulated and real-world tasks involving elongated objects.

## Repository Structure
- `scripts/`: Source code for the TAPOM planner implementation.
- `ext/`: Dependence of TAPOM.
- `data/`: Environment models used in experiments.

## Installation (Unfinished)

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

* install OMPL with python bindings https://ompl.kavrakilab.org/core/installation.html

## How to Run?

Coming soon...
