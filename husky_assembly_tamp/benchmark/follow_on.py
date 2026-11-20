"""Whether the movements after a bar's carry can follow from the posture the carry ends in.

A carry planned to the goal flange poses may end in another arm posture than the authored one (another IK branch,
or a joint a whole turn away). The insert and the retreat after it then need joints of their own: `FollowOn` plans
both as straight moves from that posture, in their own scenes, and keeps the joints they end in.
"""
from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from husky_assembly_tamp.benchmark.process_export import (
    RELEASE_CODE,
    action_id,
    attach_to_left,
    carry_movement,
    goal_link_poses,
    jointing_action,
    refresh_world_poses,
    require_core,
)
from husky_assembly_tamp.motion_planner import api


class FollowOn:
    """Plans a bar's insert, then its retreat, from a given end posture of the carry.

    ! It has its own PyBullet world, so it can be asked while a carry is being planned in another.

    Attributes:
        keyframes: per accepted posture, in the order asked: its twelve joints, and the joints the insert and the
            retreat end in, all in the planner's joint order.
        seconds: time spent in the checks so far.
    """

    def __init__(self, process, bar_id: str, joint_names_12: Sequence[str], base=None,
                 log: Optional[Callable[[str], None]] = None) -> None:
        """Find the two movements and open the world.

        Args:
            process: The process.
            bar_id: The bar, e.g. "B21".
            joint_names_12: The twelve arm joints, left six then right six.
            base: The acting robot's base, if not the planned one.
            log: Gets a line per check.

        Raises:
            ValueError: The process has no straight, coupled insert after the carry, or no straight retreat in the
                bar's release action.
        """
        require_core()
        from scaffolding_assembly_core.mirrors.compas_fab import CompasFabMirror

        self.process, self.names, self.base, self.log = process, list(joint_names_12), base, log
        self.action = jointing_action(process, bar_id)
        self.robot = self.action.robot
        after = self.action.movements[self.action.movements.index(carry_movement(self.action)) + 1:]
        inserts = [m for m in after if m.arms and m.path == "linear" and m.coupled]
        release = process.actions.get(action_id(bar_id, RELEASE_CODE))
        retreats = [m for m in (release.movements if release is not None else ()) if m.arms and m.path == "linear"]
        if not inserts or not retreats:
            raise ValueError(f"{self.action.id}: no straight insert after the carry, or no straight retreat in "
                             f"{action_id(bar_id, RELEASE_CODE)}")
        self.insert, self.retreat = inserts[0], retreats[0]
        self.mirror = CompasFabMirror(self.robot)
        self.keyframes: List[Dict[str, List[float]]] = []
        self.seconds = 0.0

    def _state(self, movement, joints: Sequence[float], held: bool):
        """Sync the world to a movement's start with the robot at these joints, and give its state."""
        scene = self.process.scene_at(movement)
        robot = scene.robots[self.robot]
        robot.joints = {**robot.joints, **dict(zip(self.names, (float(v) for v in joints)))}
        robot.unmeasured = frozenset()
        robot.enabled = True
        if self.base is not None:
            robot.base, robot.base_tracked = self.base, True
        refresh_world_poses(scene)
        if held:
            attach_to_left(scene, self.process, self.action, movement)
        self.mirror.sync(scene)
        return self.mirror.state

    def _goal(self, movement) -> Dict[str, Any]:
        """A movement's two goal flange poses as compas frames."""
        from scaffolding_assembly_core.mirrors.compas import frame_from_pose

        return {side: frame_from_pose(pose) for side, pose in goal_link_poses(movement, self.robot).items()}

    @staticmethod
    def _why(report: Dict[str, Any]) -> str:
        """A failed straight move's cause, from the planner's report."""
        rejected = sorted(set(report.get("rejected_by") or ()))
        cause = report.get("reason", "no path") + (f" ({', '.join(rejected)})" if rejected else "")
        return f"{cause} at waypoint {report.get('waypoint')} of {report.get('n_waypoints')}"

    def __call__(self, conf12: Sequence[float]) -> str:
        """Whether the insert and the retreat can be planned from these joints.

        Args:
            conf12: The carry's end posture, twelve joints.

        Returns:
            str: "" when both are planned (their end joints are then in `keyframes`), else which one failed and why.
        """
        started = perf_counter()
        try:
            verdict = self._check(np.asarray(conf12, dtype=float))
        finally:
            self.seconds += perf_counter() - started
        if self.log is not None:
            self.log(f"  [follow-on] {'insert and retreat follow' if not verdict else verdict}")
        return verdict

    def _check(self, conf12: np.ndarray) -> str:
        """The two plans; see `__call__`."""
        report: Dict[str, Any] = {}
        state = self._state(self.insert, conf12, held=True)
        with self.mirror.lend() as planner:
            inserted = api.plan_constrained_dual_arm_linear(
                planner, state, active_bar_id=self.action.bar, goal_ee_frames=self._goal(self.insert),
                skip_env_collisions=False, report=report)
        if inserted is None:
            return f"insert: {self._why(report)}"
        insert_end = [float(v) for v in inserted.points[-1].joint_values]
        report = {}
        state = self._state(self.retreat, insert_end, held=False)
        with self.mirror.lend() as planner:
            retreated = api.plan_dual_arm_linear_independent(
                planner, state, goal_ee_frames=self._goal(self.retreat), skip_env_collisions=False, report=report)
        if retreated is None:
            return f"retreat: {self._why(report)}"
        self.keyframes.append({"end": conf12.tolist(), "insert": insert_end,
                               "retreat": [float(v) for v in retreated.points[-1].joint_values]})
        return ""

    def keyframes_at(self, conf12: Sequence[float], within: float = 1e-4) -> Optional[Dict[str, List[float]]]:
        """The joints the insert and the retreat end in from an accepted posture.

        Args:
            conf12: The posture, twelve joints.
            within: How far (rad, largest joint) an accepted posture may be from it and still count as the same.

        Returns:
            Optional[Dict[str, List[float]]]: `end`, `insert` and `retreat`, or None when no such posture was accepted.
        """
        asked = np.asarray(conf12, dtype=float)
        for found in reversed(self.keyframes):
            if np.abs(np.asarray(found["end"]) - asked).max() <= within:
                return found
        return None

    def close(self) -> None:
        """Close the world."""
        self.mirror.close()
