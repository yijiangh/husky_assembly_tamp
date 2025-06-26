# Dual-Arm Robot Testing and Planning Documentation

This document provides comprehensive documentation for two key Python scripts used in dual-arm robot manipulation and constrained planning: `dual_test.py` and `dual_constrain_test.py`.

## Table of Contents
1. [Overview](#overview)
2. [dual_test.py - Interactive Dual-Arm Testing](#dual_testpy---interactive-dual-arm-testing)
3. [dual_constrain_test.py - Constrained Motion Planning](#dual_constrain_testpy---constrained-motion-planning)
4. [Running Instructions](#running-instructions)

## Overview

These scripts are designed for testing and planning dual-arm robotic systems using the Husky dual-arm robot with UR5e manipulators. The scripts utilize PyBullet for physics simulation, OMPL for motion planning, and provide interactive interfaces for configuration and analysis.

---

## dual_test.py - Interactive Dual-Arm Testing

### Purpose
`dual_test.py` is an interactive testing environment for dual-arm robot configurations. It provides real-time inverse kinematics solutions and allows users to record configurations for later analysis.

### Key Features

#### 1. **Interactive Configuration Interface**
- **Element Pose Control**: 6-DOF sliders (x, y, z, roll, pitch, yaw) for positioning a grasped object
- **Box Offset Control**: 4-DOF sliders for adjusting left and right arm target poses relative to the grasped object
- **Real-time Updates**: Immediate visual feedback of robot configurations

#### 2. **Dual-Arm Coordination**
- **Synchronized Movement**: Both arms maintain relative positioning to a shared grasped object
- **Inverse Kinematics**: Real-time IK solutions using TracIK solver
- **Collision Visualization**: Visual representation of grippers and grasped objects

#### 3. **Data Recording System**
- **Record Button**: Start recording current configurations
- **Stop Button**: Stop recording and save data to timestamped file
- **Print Button**: Output current joint positions to console
- **Automatic Saving**: Data saved as NPZ files with timestamps

### Code Structure

#### Initialization Phase
```python
# Robot setup
robot = pp.load_pybullet(robot_urdf, fixed_base=False, cylinder=False)
left_ee = pp.create_obj(gripper_obj, scale=1)
right_ee = pp.create_obj(gripper_obj, scale=1)

# IK solvers
left_solver = TracIKSolver(robot_urdf, "base_link", "left_ur_arm_tool0")
right_solver = TracIKSolver(robot_urdf, "base_link", "right_ur_arm_tool0")
```

#### Main Control Loop
1. **User Input Processing**: Read slider values and button states
2. **Pose Calculation**: Compute target poses based on slider inputs
3. **Inverse Kinematics**: Solve for joint configurations
4. **Robot Update**: Apply joint positions to robot model
5. **Data Recording**: Store configurations if recording is active

#### Data Format
Recorded data includes:
- `box_sliders`: Offset parameters for both arms
- `element_sliders`: 6-DOF pose of grasped object
- `left_sol`: Left arm joint configuration
- `right_sol`: Right arm joint configuration

### Usage Scenarios
- **Configuration Testing**: Test feasibility of dual-arm grasping configurations
- **Data Collection**: Record sequences of valid configurations for training
- **Visualization**: Visualize dual-arm coordination and collision scenarios
- **Calibration**: Fine-tune relative positioning parameters

---

## dual_constrain_test.py - Constrained Motion Planning

### Purpose
`dual_constrain_test.py` implements constrained motion planning for dual-arm robots where both arms must maintain a fixed relative relationship (e.g., grasping a rigid object).

### Key Features

#### 1. **Relative Position Constraint**
```python
class RelativeEndEffectorConstraint(ob.Constraint):
    """Constraint that keeps the relative position of the left end-effector 
    in the right end-effector's coordinate frame fixed."""
```

The constraint enforces:
- **Fixed Relative Position**: Left EE position in right EE coordinate frame
- **Bidirectional Enforcement**: Both left-in-right and right-in-left relationships
- **6-DOF Constraint**: 3 positional constraints for each relative relationship

#### 2. **Advanced Motion Planning**
- **OMPL Integration**: Uses Open Motion Planning Library
- **Multiple Planners**: Support for RRTConnect, EST, KPIECE1, etc.
- **Constrained Spaces**: Planning in projection-based constraint manifolds
- **State Validation**: Collision checking and joint limit validation

#### 3. **Constraint Violation Analysis**
- **Pre-planning Validation**: Check start/goal constraint violations before planning
- **Trajectory Analysis**: Compute violations along entire planned trajectory
- **Statistical Reporting**: Max, mean, and final violation statistics
- **Visualization**: Plot constraint violations with dual plots (linear and log scale)

#### 4. **Interactive Trajectory Playback**
- **Slider Interface**: Scrub through trajectory waypoints
- **Real-time Visualization**: See robot configuration at each waypoint
- **Trajectory Visualization**: Optional polyline rendering of end-effector paths

### Code Structure

#### Constraint Implementation
```python
def function(self, x, out):
    """Compute constraint residuals given joint configuration."""
    current_right_from_left, current_left_from_right = self._relative_position(x)
    diff_right = current_right_from_left - self.desired_right_from_left
    diff_left = current_left_from_right - self.desired_left_from_right
    out[0:3] = diff_right
    out[3:6] = diff_left
```

#### Planning Pipeline
1. **Constraint Creation**: Initialize relative position constraint
2. **Pre-validation**: Check start/goal constraint violations
3. **Space Setup**: Create constrained configuration space
4. **Planning**: Execute motion planning with specified algorithm
5. **Post-processing**: Interpolate and analyze resulting trajectory
6. **Visualization**: Interactive playback and constraint analysis

#### Constraint Violation Metrics
- **Position Error**: Euclidean distance between current and desired relative positions
- **Combined Metric**: Average of left-in-right and right-in-left violations
- **Threshold Checking**: Warning system for violations exceeding acceptable limits

### Mathematical Foundation

#### Relative Position Calculation
For two end-effectors with poses `P_left` and `P_right`:
```
T_right_from_left = T_right^(-1) * T_left
relative_position = translation_component(T_right_from_left)
```

#### Constraint Function
```
f(q) = [current_relative_pos - desired_relative_pos]
```
Where `q` is the joint configuration vector.

---

## Running Instructions

### Running dual_test.py

#### Basic Execution
```bash
cd eth_ws/src/husky_assembly/scripts/test
python dual_test.py
```

#### Interactive Usage
1. **Launch**: Execute the script to open PyBullet GUI
2. **Configure Object Pose**: Use 6-DOF sliders (x, y, z, roll, pitch, yaw)
3. **Adjust Arm Offsets**: Use box offset sliders for fine-tuning
4. **Monitor Solutions**: Watch real-time IK solutions and robot updates
5. **Record Data**: Click "Record" to start, "Stop" to save configurations
6. **Print States**: Click "Print" to output current joint positions **(This is used to generate and copy joint confs manually, especially for dual_constrain_test.py)**

#### Slider Controls
- **Element Sliders**: 
  - `x, y, z`: Object position (meters)
  - `roll, pitch, yaw`: Object orientation (radians)
- **Box Sliders**:
  - `left_y, left_pitch`: Left arm offset parameters
  - `right_y, right_pitch`: Right arm offset parameters

#### Data Output
- **Location**: `eth_ws/src/husky_assembly/scripts/recordings/`
- **Format**: `recorded_data_YYYYMMDD_HHMMSS.npz`
- **Content**: Dictionary with slider values and joint solutions

### Running dual_constrain_test.py

#### Basic Execution
```bash
cd eth_ws/src/husky_assembly/scripts/test
python3 dual_constrain_test.py [OPTIONS]
```

#### Command Line Options
```bash
# Basic planning with RRTConnect
python3 dual_constrain_test.py --planner RRTConnect --space PJ --time 600 --interpolate-points 300
```

#### Available Parameters
- **`--planner`**: Planning algorithm (RRTConnect, EST, KPIECE1, BiEST, BiKPIECE1)
- **`--space`**: Constraint space type (PJ, TB, AT)
- **`--time`**: Planning time limit in seconds
- **`--tolerance`**: Constraint tolerance
- **`--interpolate-points`**: Number of trajectory interpolation points
- **`--output`**: Save trajectory to file
- **`--bench`**: Run benchmark comparison
- **`--plot-violations`**: Generate constraint violation plots

#### Execution Flow
1. **Initialization**: Load robot model and setup PyBullet environment
2. **Configuration Display**: Visualize start and goal configurations
3. **Pre-validation**: Check constraint violations for start/goal states
4. **Planning**: Execute constrained motion planning
5. **Analysis**: Compute trajectory constraint violations
6. **Visualization**: Interactive trajectory playback with slider control

#### Output Files
- **Trajectory**: `dual_constraint_trajectory.txt` (if --output specified)
- **Violation Plot**: `dual_constraint_violations_YYYYMMDD_HHMMSS.png`
- **Violation Data**: `dual_constraint_violations_data_YYYYMMDD_HHMMSS.txt`

#### Interactive Controls
- **Trajectory Slider**: Scrub through planned trajectory waypoints
- **Keyboard**: Ctrl+C to exit
- **Real-time**: Robot configuration updates as slider moves

---

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# ModuleNotFoundError: No module named 'ompl'
# need to install OMPL in README.md

# pybullet_planning import issues
pip install pybullet-planning
```

#### 2. URDF Loading Failures
- Verify `DATA_DIR` environment variable
- Check file permissions for URDF/SRDF files
- Ensure mesh files are accessible

#### 3. IK Solution Failures
- Adjust initial joint configurations
- Verify target poses are within workspace
- Check for collision conflicts

#### 4. Planning Failures
- Increase planning time limit
- Try different constraint spaces (PJ, TB, AT)
- Verify start/goal configurations satisfy constraints
- Adjust constraint tolerance
- **Believe the god (This planner is kind of unstable)**

---

## Technical Notes

### Coordinate Frames
- **Base Frame**: Robot base coordinate system
- **Tool Frames**: `left_ur_arm_tool0`, `right_ur_arm_tool0`
- **World Frame**: PyBullet world coordinate system

### Constraint Mathematics
The relative position constraint maintains:
```
P_left_in_right = T_right^(-1) * T_left = constant
P_right_in_left = T_left^(-1) * T_right = constant
```

### Planning Spaces
- **PJ (Projection)**: Direct projection onto constraint manifold
- **TB (Tangent Bundle)**: Tangent space approximation
- **AT (Atlas)**: Multiple chart representation

### Joint Naming Convention
```python
LEFT_JOINT_NAMES = [
    "left_ur_arm_shoulder_pan_joint",
    "left_ur_arm_shoulder_lift_joint", 
    "left_ur_arm_elbow_joint",
    "left_ur_arm_wrist_1_joint",
    "left_ur_arm_wrist_2_joint", 
    "left_ur_arm_wrist_3_joint"
]
```

This documentation provides a comprehensive guide for using both dual-arm testing scripts. For additional support or advanced configurations, refer to the inline code documentation and OMPL/PyBullet official documentation.
