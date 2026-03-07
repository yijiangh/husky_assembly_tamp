"""
Analytical inverse kinematics wrapper for UR5e, designed to work with the
constrained bimanual planner (adapted from the IIWA 7-DOF version).

Uses the ur-analytic-ik library (https://github.com/Victorlouisdg/ur-analytic-ik)
which provides closed-form IK solutions (~18us per call) based on Hawkins (2013).

Key differences from IIWA 7-DOF:
- UR5e is 6-DOF: no continuous self-motion parameter (psi)
- Up to 8 discrete IK solutions per target pose
- Planning uses 6D (left arm joints) + discrete branch selection

Install:
    pip install ur-analytic-ik
"""

import numpy as np

try:
    from ur_analytic_ik import ur5e
except ImportError:
    raise ImportError(
        "ur-analytic-ik not installed. Run: pip install ur-analytic-ik"
    )

# UR5e joint limits from our calibrated URDF
# shoulder_pan, shoulder_lift: +/- 270 deg (4.71238898038 rad)
# elbow: +/- 180 deg (3.14159265359 rad)
# wrist_1, wrist_2, wrist_3: +/- 240 deg (4.18879020479 rad)
UR5E_LIMITS_LOWER = np.array([
    -4.71238898038,  # shoulder_pan
    -4.71238898038,  # shoulder_lift
    -3.14159265359,  # elbow
    -4.18879020479,  # wrist_1
    -4.18879020479,  # wrist_2
    -4.18879020479,  # wrist_3
])

UR5E_LIMITS_UPPER = np.array([
    4.71238898038,
    4.71238898038,
    3.14159265359,
    4.18879020479,
    4.18879020479,
    4.18879020479,
])

# Number of DOF
N_JOINTS = 6
# Maximum number of IK solution branches
MAX_BRANCHES = 8


class AnalyticIK_UR5e:
    """
    Analytical IK solver for UR5e, matching the interface pattern of
    Analytic_IK_7DoF used by the IIWA constrained bimanual planner.

    Usage:
        ik = AnalyticIK_UR5e()
        pose = ik.FK(q6)              # Forward kinematics -> 4x4 matrix
        sols = ik.IK_all(pose)        # All valid IK solutions (up to 8)
        q = ik.IK(pose, branch=0)     # Single solution by branch index
    """

    def __init__(self):
        self.limits_lower = UR5E_LIMITS_LOWER.copy()
        self.limits_upper = UR5E_LIMITS_UPPER.copy()
        self.n_joints = N_JOINTS

    def FK(self, q6):
        """
        Forward kinematics for UR5e.

        Args:
            q6: 6D joint configuration (numpy array or list)

        Returns:
            4x4 homogeneous transformation matrix (numpy array)
        """
        q = np.asarray(q6, dtype=float)
        return ur5e.forward_kinematics(*q)

    def IK_all(self, target_pose, filter_limits=True):
        """
        Compute all analytical IK solutions for a target pose.

        Args:
            target_pose: 4x4 homogeneous transformation matrix
            filter_limits: if True, filter out solutions outside joint limits

        Returns:
            List of 6D joint configurations (numpy arrays).
            May return 0 to 8 solutions.
        """
        target = np.asarray(target_pose, dtype=float)
        solutions = ur5e.inverse_kinematics(target)

        if len(solutions) == 0:
            return []

        valid = []
        for sol in solutions:
            q = np.asarray(sol)
            # Filter NaN solutions
            if np.any(np.isnan(q)):
                continue
            if filter_limits:
                if np.any(q < self.limits_lower) or np.any(q > self.limits_upper):
                    continue
            valid.append(q)

        return valid

    def IK(self, target_pose, branch=0):
        """
        Get a specific IK solution by branch index.

        Args:
            target_pose: 4x4 homogeneous transformation matrix
            branch: index into the sorted list of valid solutions (0 = first)

        Returns:
            6D joint configuration (numpy array)

        Raises:
            ValueError: if no valid solution exists for the given branch
        """
        solutions = self.IK_all(target_pose)
        if len(solutions) == 0:
            raise ValueError("No valid IK solution found for target pose")
        if branch >= len(solutions):
            raise ValueError(
                f"Branch {branch} requested but only {len(solutions)} solutions available"
            )
        return solutions[branch]

    def IK_closest(self, target_pose, q_current):
        """
        Get the IK solution closest to a reference configuration.

        Args:
            target_pose: 4x4 homogeneous transformation matrix
            q_current: 6D reference joint configuration

        Returns:
            Tuple of (6D joint config, branch_index) or (None, -1) if no solution
        """
        solutions = self.IK_all(target_pose)
        if len(solutions) == 0:
            return None, -1

        q_ref = np.asarray(q_current)
        distances = [np.linalg.norm(sol - q_ref) for sol in solutions]
        best_idx = int(np.argmin(distances))
        return solutions[best_idx], best_idx

    def IK_all_with_branches(self, target_pose):
        """
        Return all valid solutions with their original branch indices.

        Returns:
            List of (solution, original_branch_index) tuples
        """
        target = np.asarray(target_pose, dtype=float)
        solutions = ur5e.inverse_kinematics(target)

        valid = []
        for i, sol in enumerate(solutions):
            q = np.asarray(sol)
            if np.any(np.isnan(q)):
                continue
            if np.any(q < self.limits_lower) or np.any(q > self.limits_upper):
                continue
            valid.append((q, i))

        return valid
