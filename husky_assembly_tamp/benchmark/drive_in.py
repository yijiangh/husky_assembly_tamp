"""Whether the robot can drive to a bar's base pose with its arms frozen in a loading posture, the bar held.

The robot takes its loading posture at the operator's station, gets the bar there, and keeps the posture while it
drives to where the carry starts. `DriveIn` asks the monitor's base planner, copied without its follower checks
(`husky_assembly_tamp.base_planner`), for a path of that drive in the scene of the bar's carry.

! Importing this module needs the shared core library (`scaffolding_assembly_core`).
"""
from __future__ import annotations

import math
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scaffolding_assembly_core.geometry import shape_mesh
from scaffolding_assembly_core.process.types import GROUND_PREFIX

from husky_assembly_tamp.base_planner.collision import held_bodies
from husky_assembly_tamp.base_planner.lattice import Settings, drive_margins, plan_drive
from husky_assembly_tamp.base_planner.path import densify
from husky_assembly_tamp.base_planner.support import BASE_PLAN_DISTANCE, POSITION_STEP, yaw_step
from husky_assembly_tamp.base_planner.world import CLEARANCE_LIMIT, PlanningWorld
from husky_assembly_tamp.benchmark.process_export import carry_scene, refresh_world_poses

#: Standing bodies whose lowest point is more than this above the robot's top are left out, metres. No check
#: measures that far up (the planning world's CLEARANCE_LIMIT), and the search's floor map keeps only the topmost
#: surface of each cell, so a beam overhead would hide what stands under it.
OVERHEAD = CLEARANCE_LIMIT
#: Joints closer than this, radians, count as the same posture.
SAME_POSTURE = 1e-4
#: How the search words a direction that ran out of time (`base_planner.lattice._search`).
TIMED_OUT = "no path found in"


class DriveIn:
    """Can the robot drive from the station to a bar's base pose with its arms frozen in a loading posture, bar held?

    ! It has its own PyBullet world, so it can be asked while a carry is being planned in another.
    ! "No path" from the search is a time-out, not a proof: see `timed_out` in `refused`.

    Attributes:
        station: (x, y, yaw) the drive starts at, metres and radians, in the process's world frame.
        goal: (x, y, yaw) of the bar's base pose, where the drive ends.
        asks: How often it was asked; `reused` of those were answered by a path found for an earlier posture.
        seconds: Time spent answering so far; the first answer builds the world.
        refused: Per refused posture, in the order asked: `reason`, `timed_out` and `seconds`.
        left_out: Other robots left out of the scene because the process does not say where they are, and why.
    """

    def __init__(self, process, bar_id: str, joint_names_12: Sequence[str], station: Sequence[float], base=None,
                 margin: float = BASE_PLAN_DISTANCE, time_limit: float = 8.0,
                 log: Optional[Callable[[str], None]] = None) -> None:
        """Build the scene of the bar's carry as the drive sees it; the world opens with the first question.

        Args:
            process: The process.
            bar_id: The bar, e.g. "B21".
            joint_names_12: The twelve arm joints, in the order the postures are given in.
            station: (x, y, yaw) of the robot's base at the station, metres and radians, world frame.
            base: The acting robot's base at the carry, if not the planned one.
            margin: Keep this far from everything where the drive can, metres (`base_planner.lattice.plan_drive`).
            time_limit: The search's time limit, seconds; a refusal for lack of time takes up to about twice it.
            log: Gets a line per answer.
        """
        self.names, self.log = list(joint_names_12), log
        self.station = tuple(float(value) for value in station)
        self.margin, self.time_limit = float(margin), float(time_limit)
        self.scene, action, _carry = carry_scene(process, bar_id, base=base)
        self.robot = action.robot
        scene = self.scene
        held = set(held_bodies(scene, self.robot))
        tools = {tool.id for tool in scene.robots[self.robot].model.tools.values()}
        for body_id, body in scene.bodies.items():
            # * The ground is a thin sheet only the wheels may touch: the chassis is always within the margin of it.
            if body_id.startswith(GROUND_PREFIX):
                body.enabled = False
            # * Strict, as the carry is planned: nothing the arms carry or wear may touch a standing body.
            if body_id in held:
                body.touches = tuple(other for other in body.touches if other not in scene.bodies or other in held)
            else:
                body.touches = tuple(other for other in body.touches if other not in held and other not in tools)
        # * As the arm planner does: another robot the process does not place stands nowhere, not at a guess.
        self.left_out: List[str] = []
        for robot_id, other in scene.robots.items():
            if robot_id != self.robot and other.enabled and other.acting_problems():
                other.enabled = False
                self.left_out.append(f"{robot_id}: {'; '.join(other.acting_problems())}")
        # The standing bodies that can be in the way, with the height of their lowest point.
        self._lowest = {body_id: _lowest_point(scene, body_id) for body_id, body in scene.bodies.items()
                        if body.enabled and body_id not in held and body_id in scene.world_poses
                        and body.geometry.collision}
        base_pose = scene.robots[self.robot].base
        qx, qy, qz, qw = base_pose.orientation
        self.goal = (float(base_pose.position[0]), float(base_pose.position[1]),
                     math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
        self.world = PlanningWorld()
        self.asks, self.reused, self.seconds = 0, 0, 0.0
        self.refused: List[Dict[str, Any]] = []
        # Per posture asked (`_key`): the answer. Per accepted posture: its joints and which of `_paths` it drives.
        self._verdicts: Dict[Tuple[int, ...], str] = {}
        self._accepted: List[Tuple[np.ndarray, int]] = []
        # The drive paths found so far: (N, 3) waypoints of (x, y, yaw), and whether the straight move was free.
        self._paths: List[Tuple[np.ndarray, bool]] = []
        if self.log is not None:
            for line in self.left_out:
                self.log(f"  [drive-in] left out {line}")

    def __call__(self, conf12: Sequence[float]) -> str:
        """Whether a drive path exists with the arms at these joints.

        Args:
            conf12: The loading posture, in the order of `joint_names_12`.

        Returns:
            str: "" when a path exists (`summary` gives it), else why not, in plain words.
        """
        started = perf_counter()
        self.asks += 1
        key = _key(conf12)
        if key not in self._verdicts:
            reason, timed_out = self._check(conf12)
            self._verdicts[key] = reason
            if reason:
                self.refused.append({"reason": reason, "timed_out": timed_out,
                                     "seconds": round(perf_counter() - started, 3)})
        self.seconds += perf_counter() - started
        if self.log is not None:
            self.log(f"  [drive-in] {self._verdicts[key] or 'a drive path exists'}")
        return self._verdicts[key]

    def _check(self, conf12: Sequence[float]) -> Tuple[str, bool]:
        """A new posture's answer, cheapest check first; see `__call__`.

        Returns:
            Tuple[str, bool]: The reason ("" when a path exists), and whether a refusal is for lack of time.
        """
        self._sync(conf12)
        world, robot = self.world, self.robot
        narrow, close = drive_margins(world, robot, self.station, self.goal, self.margin)
        with world.keeping(robot, narrow, close):
            for name, pose in (("station", self.station), ("goal", self.goal)):
                hit = world._hit(robot, pose, narrow)
                if hit is not None:
                    part, other = (self.scene.label(object_id) for object_id in hit)
                    return f"at the {name}, {part} is in collision with {other}", False
            # * A path found for another posture, walked with these arms as the search checks its own.
            turn_step = yaw_step(world, robot, (Settings().x_icr,))
            for index, (points, _direct) in enumerate(self._paths):
                poses = [tuple(points[0])] + densify(points, POSITION_STEP, turn_step)
                if all(world._hit(robot, pose, narrow) is None for pose in poses):
                    self.reused += 1
                    self._accepted.append((np.asarray(conf12, dtype=float), index))
                    return "", False
        # ? No "boxed in at the goal" shortcut: the search may reach the goal by a turn-drive-turn from anywhere
        #   near it, so no fixed set of last moves has to be free.
        result = plan_drive(world, robot, self.station, self.goal, self.margin, self.time_limit)
        if result.path is None:
            return f"no drive path from the station: {result.reason}", TIMED_OUT in result.reason
        self._paths.append((result.path, result.direct))
        self._accepted.append((np.asarray(conf12, dtype=float), len(self._paths) - 1))
        return "", False

    def _sync(self, conf12: Sequence[float]) -> None:
        """Put the arms at these joints in the world, and leave out what stands wholly above the robot (OVERHEAD)."""
        scene = self.scene
        robot = scene.robots[self.robot]
        robot.joints = {**robot.joints, **dict(zip(self.names, (float(value) for value in conf12)))}
        refresh_world_poses(scene)
        for body_id in self._lowest:
            scene.bodies[body_id].enabled = True
        self.world.sync(scene)
        # ! The robot's top depends on the posture, so this is decided per question.
        ceiling = self.world._park(self.robot) + OVERHEAD
        overhead = [body_id for body_id, lowest in self._lowest.items() if lowest > ceiling]
        if overhead:
            for body_id in overhead:
                scene.bodies[body_id].enabled = False
            self.world.sync(scene)

    def summary(self, conf12: Optional[Sequence[float]] = None) -> Dict[str, Any]:
        """What was asked and answered so far, as plain data for the plan's record.

        Args:
            conf12: An accepted posture (the start of the returned plan), to add its drive path.

        Returns:
            Dict[str, Any]: `station`, `goal`, `margin`, `time_limit`, `asks`, `accepted` (postures that can drive
            in), `reused`, `seconds`, `refused`; and for an accepted `conf12` its path as `points`
            ([[x, y, yaw], ...]), `length_m` and `direct`.
        """
        out = {"station": list(self.station), "goal": list(self.goal), "margin": self.margin,
               "time_limit": self.time_limit, "asks": self.asks, "accepted": len(self._accepted), "reused": self.reused,
               "seconds": round(self.seconds, 3), "refused": [dict(entry) for entry in self.refused]}
        # * By distance, not by `_key`: a planned path's first waypoint is the accepted posture up to rounding.
        index = None if conf12 is None else next(
            (index for posture, index in self._accepted
             if np.abs(posture - np.asarray(conf12, dtype=float)).max() <= SAME_POSTURE), None)
        if index is not None:
            points, direct = self._paths[index]
            out.update(points=points.tolist(), direct=direct,
                       length_m=float(np.hypot(*np.diff(points[:, :2], axis=0).T).sum()))
        return out

    def close(self) -> None:
        """Close the world."""
        self.world.close()


def _key(conf12: Sequence[float]) -> Tuple[int, ...]:
    """A posture as the key its answer is kept under: its joints in steps of SAME_POSTURE."""
    return tuple(int(round(float(value) / SAME_POSTURE)) for value in conf12)


def _lowest_point(scene, body_id: str) -> float:
    """The height of the lowest point of a body's collision shapes, metres, in the world."""
    matrix = scene.world_poses[body_id].matrix()
    return float(matrix[2, 3]) + min(float((shape_mesh(shape).vertices @ matrix[2, :3]).min())
                                     for shape in scene.bodies[body_id].geometry.collision)
