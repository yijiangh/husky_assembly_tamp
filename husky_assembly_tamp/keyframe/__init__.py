"""Keyframe solving package: dual-arm IK + base search, Rhino-free.

This package is the offline half of the bar-assembly keyframe workflow. It holds
everything needed to solve "where does the robot base stand, and what are the
per-keyframe arm configurations" for one BarAssemblyAction, using only the JSON
files exported from the design front-end (RobotCell.json, BarActions/<bar>.json,
WalkableGround.json):

- ``config``          solver tuning constants + ssik path/backend resolution
- ``ssik_inprocess``  native in-process ssik solver (offline Py3.10+ venv)
- ``ssik_client``     ssik entry point: in-process when ssik imports, else the sidecar
- ``dual_arm_ik``     the dual-arm IK solvers (ssik + gradient backends)
- ``ik_keyframe``     the chained M1 -> M2 -> M3 keyframe solve
- ``walkable_ground`` ground meshes, seed-base derivation, expanding base search
- ``ssik_sidecar/``   sidecar server the ssik backend spawns only for Rhino's 3.9

! Keep this __init__ import-free: the Rhino front-end imports
! ``husky_assembly_tamp.keyframe.config`` on every script run, and any re-export
! here would drag the heavy compas / pybullet stack into that cheap import.
"""
