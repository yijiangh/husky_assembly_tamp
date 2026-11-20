# * Copy of husky-assembly-teleop `husky_assembly_teleop/world/base_collision.py` at monitor commit b75b9975.
# * Kept close to the original so it can be re-synced: what is here is verbatim, the rest is cut.
# * Cut: `collision_hit`, `_scene_at`, `_contact`, `_names`, `path_hit`, `self_hit` and its helpers
# *      (`_self_pair_indices`, `_may_hold_touch`, `_holder_link`, `_self_pairs`), `srdf_allowed_pairs`, and the
# *      `collision_hit` import; the module docstring's sentence on `self_hit`.
"""
BaseCollisionWorld: what a robot base at a floor pose comes near, in a private PyBullet mirror of the scene.

* What a robot holds (`held_bodies`) moves with it and is checked with it; contact between the two is left out, as
  driving never changes it.
! Worker thread only: construct it anywhere, but sync, check and close on one worker. A mirror belongs to one thread.
! Use raw `p` calls with `physicsClientId`: `pp.CLIENT` is one global shared by the whole process.
"""

from __future__ import annotations

import math

import numpy as np
import pybullet as p

from scaffolding_assembly_core.geometry import Pose, compose, invert
from scaffolding_assembly_core.mirrors.pybullet import PyBulletMirror
from scaffolding_assembly_core.scene import Attachment, Scene


def held_bodies(snapshot: Scene, robot: str) -> list[str]:
    """The bodies a robot holds: attached to one of its links or its base, or to a body it holds.

    Args:
        snapshot: The world.
        robot: The robot, by id.
    """
    def holder(body_id: str, seen: frozenset) -> str | None:
        placement = snapshot.bodies[body_id].placement
        if not isinstance(placement, Attachment):
            return None
        if placement.parent in snapshot.robots:
            return placement.parent
        if placement.parent in snapshot.bodies and placement.parent not in seen:
            return holder(placement.parent, seen | {body_id})
        return None

    return [body_id for body_id in snapshot.bodies if holder(body_id, frozenset()) == robot]


class BaseCollisionWorld:
    """A private PyBullet mirror of the scene that says what a robot base at a floor pose comes near. Worker only."""

    def __init__(self):
        """Start empty; the first `sync` loads the robots and bodies (slow, on the worker)."""
        self.mirror: PyBulletMirror | None = None
        #: The snapshot synced last: display text for our ids, and the scene of a `CollisionHit`.
        self.snapshot = Scene()
        # ? Cheap pre-check for `hit_by`: floor footprints (min x, min y, max x, max y) by id, and robot reach.
        self._footprints: dict[str, tuple[float, float, float, float]] = {}
        self._reach: dict[str, float] = {}
        # What each robot holds, by robot id: body id -> (pose in the robot's base frame, pose as synced).
        self._held: dict[str, dict[str, tuple[Pose, Pose]]] = {}
        # Robots whose held bodies a check moved: the mirror never moves a body back, so `sync` does.
        self._moved: set[str] = set()

    def sync(self, snapshot: Scene) -> None:
        """Make the world match a snapshot, and refresh footprints, reach and what each robot holds.

        Args:
            snapshot: The tick's copy of the world, taken on the main thread.
        """
        if self.mirror is None:
            self.mirror = PyBulletMirror()  # here, on the worker: a mirror lives on one thread
        for robot in self._moved:
            for body_id, (_, synced) in self._held[robot].items():
                self._put(body_id, synced)
        self._moved = set()
        self.mirror.sync(snapshot)
        self.snapshot = snapshot
        # Footprint, reach and what is held depend on arm pose, so refresh them every sync.
        ids = list(self.mirror.robots) + self.mirror.obstacle_ids()
        self._footprints = {object_id: self._footprint(self.mirror.body_ids(object_id)) for object_id in ids}
        self._held, self._reach = {}, {}
        for object_id, body in self.mirror.robots.items():
            base = invert(snapshot.robots[object_id].base)
            self._held[object_id] = {body_id: (compose(base, snapshot.world_poses[body_id]),
                                               snapshot.world_poses[body_id])
                                     for body_id in held_bodies(snapshot, object_id) if body_id in self._footprints}
            footprint = self._footprint([body for part in self._parts(object_id)
                                         for body in self.mirror.body_ids(part)])
            position = p.getBasePositionAndOrientation(body, physicsClientId=self.mirror.client_id)[0]
            corners = np.array([(footprint[i], footprint[j]) for i in (0, 2) for j in (1, 3)])
            self._reach[object_id] = float(np.max(np.linalg.norm(corners - np.array(position[:2]), axis=1)))

    def _footprint(self, bodies: list[int]) -> tuple[float, float, float, float]:
        """The floor footprint of some PyBullet bodies: their bounding box over every link, in x and y.

        Args:
            bodies: Body ids in the mirror's world.

        Returns:
            tuple[float, float, float, float]: (min x, min y, max x, max y), metres.
        """
        client = self.mirror.client_id
        boxes = [p.getAABB(body, link, physicsClientId=client)
                 for body in bodies for link in range(-1, p.getNumJoints(body, physicsClientId=client))]
        low = np.min([lo for lo, _ in boxes], axis=0)
        high = np.max([hi for _, hi in boxes], axis=0)
        return float(low[0]), float(low[1]), float(high[0]), float(high[1])

    def hit_by(self, own: str, pose, margin: float) -> str | None:
        """Place a robot at a floor pose and say what it hits, if anything.

        Args:
            own: The robot being planned for, by id.
            pose: (x, y, yaw).
            margin: Keep at least this far from everything else, metres.

        Returns:
            str | None: Label of the first robot or body within `margin`; None if clear.
        """
        hit = self._hit(own, pose, margin)
        return self.snapshot.label(hit[1]) if hit is not None else None

    def reach(self, own: str) -> float:
        """How far robot `own` and what it holds reach across the floor from base_footprint, as last synced, metres."""
        return self._reach[own]

    def _hit(self, own: str, pose, margin: float) -> tuple[str, str] | None:
        """Place a robot at a floor pose with what it holds, and say which of them comes within `margin` of what.

        Returns:
            tuple[str, str] | None: The robot's or held body's id, and the id of the first thing it hits; None if clear.
        """
        self._place(own, pose)
        candidates = self._candidates(own, pose, margin)
        for part in self._parts(own):
            # * One by one, nearest first, up to the first hit: most checks near things hit, and a check costs
            #   per object.
            for other in candidates:
                if self.mirror.collisions(part, margin, [other]):
                    return part, other
        return None

    def _candidates(self, own: str, pose, margin: float) -> list[str]:
        """The other robots and bodies whose footprint lies within reach of robot `own` at `pose`, plus `margin`.

        Nearest footprint first.
        """
        x, y, _ = pose
        reach = self._reach.get(own, math.inf) + margin
        parts = self._parts(own)
        candidates = []
        for object_id, footprint in self._footprints.items():
            # Distance from the robot's base origin to the footprint rectangle.
            dx = max(footprint[0] - x, 0.0, x - footprint[2])
            dy = max(footprint[1] - y, 0.0, y - footprint[3])
            if object_id not in parts and math.hypot(dx, dy) <= reach:
                candidates.append((math.hypot(dx, dy), object_id))
        return [object_id for _, object_id in sorted(candidates)]

    def _parts(self, own: str) -> list[str]:
        """A robot and what it holds, robot first."""
        return [own, *self._held.get(own, {})]

    def _place(self, own: str, pose) -> None:
        """Move a robot to a floor pose (x, y, yaw), at its height, with what it holds."""
        body = self.mirror.robot(own)
        client = self.mirror.client_id
        z = p.getBasePositionAndOrientation(body, physicsClientId=client)[0][2]
        x, y, yaw = pose
        base = Pose((x, y, z), (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))
        # ? Moving the robot here is fine: the mirror re-poses robots on every sync.
        p.resetBasePositionAndOrientation(body, base.position, base.orientation, physicsClientId=client)
        held = self._held.get(own)
        if held:
            self._moved.add(own)
            for body_id, (offset, _) in held.items():
                self._put(body_id, Pose(*p.multiplyTransforms(base.position, base.orientation, offset.position,
                                                              offset.orientation)))

    def _put(self, body_id: str, pose: Pose) -> None:
        """Move every PyBullet body of a body to a world pose."""
        for body in self.mirror.body_ids(body_id):
            p.resetBasePositionAndOrientation(body, pose.position, pose.orientation,
                                              physicsClientId=self.mirror.client_id)

    def close(self) -> None:
        """Disconnect the world. On the worker, once no search is running."""
        if self.mirror is not None:
            self.mirror.close()
            self.mirror = None
