# Husky Assembly Project

## Installation

* create virtual env

```bash
conda create -n husky_assembly python==3.9.19
conda activate husky_assembly
```

```bash
pip install matplotlib torch rich pybullet_planning pyyaml casadi compas_fab pysdtw pynvml
conda install pinocchio -c conda-forge
```

* install python dependencies

```bash
conda install pinocchio -c conda-forge
pip install numpy pybullet_planning torch matplotlib casadi compas_fab
```

* install curobo https://curobo.org/get_started/1_install_instructions.html

* install OMPL with python bindings https://ompl.kavrakilab.org/core/installation.html

## How to Run?

* corner case

```bash
cd scripts
python test/corner_case_for_transfer [--birrt] [--curobo] [--tampor] [--ompl RRTConnect PRM ...] [--save] [--random] [--visualize] [--manual] [--repeat {int}] [--scene {scene_name}] [--task {task_name}] [--max_time {float}] [--time_stamp {time_stamp}]
```

## TODO List

* cuboid_1
  - [ ] BiRRT
  - [ ] RRTConnect
  - [ ] PRM
  - [ ] LazyRRT
  - [ ] EST
  - [ ] STRIDE 
  - [ ] BIT*
  - [ ] EIT*
  - [ ] BFMT
  - [ ] cuRobo
  - [ ] TAPOM

* shelf_1
  - [ ] BiRRT
  - [ ] RRTConnect
  - [ ] PRM
  - [ ] LazyRRT
  - [ ] EST
  - [ ] STRIDE 
  - [ ] BIT*
  - [ ] EIT*
  - [ ] BFMT
  - [ ] cuRobo
  - [ ] TAPOM