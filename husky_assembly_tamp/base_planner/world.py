# * Copy of husky-assembly-teleop `husky_assembly_teleop/plugins/base_planner/world.py` at monitor commit b75b9975.
# * Kept close to the original so it can be re-synced: what is here is verbatim, the rest is cut.
# * Cut: `stowed`, `PlanningWorld.touch`, `PlanningWorld.set_gui`, and the imports only they need (`RobotConfig`,
# *      the monitor's kinematics, `replace`, the core's `link_pose`, `PyBulletMirror`, `Scene`, `world_poses`).
# * Changed: the import of `BaseCollisionWorld` (from `.collision`), and the module docstring's sentence on `stowed`.
"""
The planning world: the shared base collision world (`BaseCollisionWorld`) plus the base planner's own checks.

The planner plans the robot as it is, arms and what it holds included.

! Worker thread only: construct `PlanningWorld` anywhere, but sync, check and close on the worker.
! Use raw `p` calls with `physicsClientId`: `pp.CLIENT` is one global shared by the whole process.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator, Mapping

import numpy as np
import pybullet as p

from .collision import BaseCollisionWorld

#: The algorithms' own margin, metres, when called on their own (tests, offline experiments). The planner plugin plans
#: with the request's margin, else BASE_PLAN_DISTANCE (`world/clearance.py`), as the panel's Margin slider starts.
COLLISION_MARGIN = 0.02
#: `clearance` measures up to CLEARANCE_LIMIT, to CLEARANCE_STEP, metres.
CLEARANCE_LIMIT = 0.30
CLEARANCE_STEP = 0.005
#: Things lower than this, metres, are floor: the floor maps leave them out.
FLOOR_BOTTOM = 0.05
#: Rays per floor cell along x and y in `floor_heights`, spread evenly over it: FLOOR_SAMPLES rays across a 5 cm cell
#: leave gaps narrower than the 2 cm bars, so no bar falls between them.
FLOOR_SAMPLES = 3
#: Rays per PyBullet call, below its batch limit.
RAY_BATCH = 8192
#: Where the planned robot is moved out of the way, metres along x and y.
PARK = -1000.0


class PlanningWorld(BaseCollisionWorld):
    """The base collision world, plus what the base planner measures: clearance, contact points and floor maps.

    Inside `keeping`, the objects a start or target is already close to keep their own, smaller margin.
    """

    #: (robot, its usual margin, object id -> its own margin) while `keeping`, else None.
    _keeping: tuple[str, float, dict[str, float]] | None = None

    def close_objects(self, own: str, poses, margin: float) -> dict[str, float]:
        """Every object robot `own` (or what it holds) is closer than `margin` to at any of `poses`, and how close.

        Returns:
            dict[str, float]: Object id -> the least distance, metres, as PyBullet measures it; 0 where it touches.
        """
        found: dict[str, float] = {}
        client = self.mirror.client_id
        for pose in poses:
            self._place(own, pose)
            candidates = self._candidates(own, pose, margin)
            for part in self._parts(own):
                for other in self.mirror.collisions(part, margin, candidates):
                    points = [point for own_body in self.mirror.body_ids(part) for body in self.mirror.body_ids(other)
                              for point in p.getClosestPoints(own_body, body, margin, physicsClientId=client)]
                    nearest = max(0.0, min((point[8] for point in points), default=0.0))
                    found[other] = min(found.get(other, margin), nearest)
        return found

    @contextmanager
    def keeping(self, own: str, margin: float, close: Mapping[str, float]) -> Iterator[None]:
        """Within the block, robot `own` keeps from each object in `close` its own margin instead of `margin`.

        A check for `margin` plus some extra (room for the follower's drift, a portal's room) keeps that object's
        margin plus the same extra. Measurements (`clearance`) are unchanged.
        """
        before, self._keeping = self._keeping, (own, float(margin), dict(close))
        try:
            yield
        finally:
            self._keeping = before

    @contextmanager
    def measured_from(self, margin: float) -> Iterator[None]:
        """Within the block, the objects kept (`keeping`) count their margins from `margin`.

        A check for `margin` plus some extra keeps each one its own margin plus that extra. Nothing changes outside
        `keeping`.
        """
        before = self._keeping
        if before is not None:
            self._keeping = (before[0], float(margin), before[2])
        try:
            yield
        finally:
            self._keeping = before

    def _hit(self, own: str, pose, margin: float) -> tuple[str, str] | None:
        """As `BaseCollisionWorld._hit`, each object in `keeping` checked at its own margin (plus the same extra)."""
        keeping = self._keeping
        if keeping is None or keeping[0] != own or not keeping[2]:
            return super()._hit(own, pose, margin)
        _own, usual, close = keeping
        self._place(own, pose)
        candidates = self._candidates(own, pose, margin)
        # * Nearest first, up to the first hit, as `BaseCollisionWorld._hit`.
        margins = [max(0.0, close[object_id] + margin - usual) if object_id in close else margin
                   for object_id in candidates]
        for part in self._parts(own):
            for object_id, own_margin in zip(candidates, margins):
                if self.mirror.collisions(part, own_margin, [object_id]):
                    return part, object_id
        return None

    def hit_by(self, own: str, pose, margin: float = COLLISION_MARGIN) -> str | None:
        """Like `BaseCollisionWorld.hit_by`, keeping the planner's own COLLISION_MARGIN by default."""
        return super().hit_by(own, pose, margin)

    def clearance(self, own: str, poses, limit: float = CLEARANCE_LIMIT) -> tuple[float, tuple | None]:
        """The least distance from a robot to everything else over some floor poses, to CLEARANCE_STEP.

        Args:
            own: The robot, by id.
            poses: (x, y, yaw) poses, close enough together that checking each checks the way between.
            limit: Measure up to this, metres.

        Returns:
            tuple[float, tuple | None]: The clearance, metres (0 where it touches, `limit` if never nearer), and the
            pose where it is least (None if never nearer than `limit`).
        """
        keeping, self._keeping = self._keeping, None  # * true distances, whatever margins are kept
        try:
            return self._clearance(own, poses, limit)
        finally:
            self._keeping = keeping

    def nearest(self, own: str, pose, limit: float = CLEARANCE_LIMIT) -> tuple[float, str | None]:
        """The true distance from a robot at one floor pose to the nearest thing, to CLEARANCE_STEP, and its label.

        Returns:
            tuple[float, str | None]: The clearance, metres (`limit` if nothing is nearer), and the label of what is
            that close (None if nothing is nearer than `limit`).
        """
        keeping, self._keeping = self._keeping, None  # * true distances, whatever margins are kept
        try:
            clear, where = self._clearance(own, [pose], limit)
            return clear, None if where is None else self.hit_by(own, pose, clear + CLEARANCE_STEP)
        finally:
            self._keeping = keeping

    def _clearance(self, own: str, poses, limit: float) -> tuple[float, tuple | None]:
        """`clearance`, measured with the usual checks."""
        lowest, where = limit, None
        for pose in poses:
            # * Only a pose nearer than the lowest so far needs measuring.
            if self.hit_by(own, pose, lowest) is None:
                continue
            clear, blocked = 0.0, lowest
            if self.hit_by(own, pose, 0.0) is not None:
                blocked = 0.0
            while blocked - clear > CLEARANCE_STEP:
                middle = (clear + blocked) / 2.0
                if self.hit_by(own, pose, middle) is None:
                    clear = middle
                else:
                    blocked = middle
            lowest, where = clear, tuple(float(value) for value in pose)
        return lowest, where

    def floor_heights(self, own: str, low, cell: float, shape: tuple[int, int], bottom: float) -> np.ndarray:
        """How high the other robots and bodies reach above each floor cell, from `bottom` up.

        FLOOR_SAMPLES by FLOOR_SAMPLES rays per cell, from above everything down, and the cell takes the highest hit:
        anything over part of a cell counts for all of it. Only the topmost surface counts: something above the robot
        hides what is under it. Bodies `own` may touch or holds are left out, and `own` is moved out of the way.

        Args:
            own: The robot being planned for, by id.
            low: (x, y) of the centre of cell (0, 0), metres.
            cell: Cell size, metres.
            shape: Cells along x and along y.
            bottom: Lowest height that counts, metres.

        Returns:
            np.ndarray: Heights, metres, of `shape`; -inf where nothing is.
        """
        self._park(own)
        parts = self._parts(own)
        others = {other for object_id in self._footprints if object_id not in parts
                  and not self.mirror.allowed(own, object_id) for other in self.mirror.body_ids(object_id)}
        # * Start above every body: a ray that starts inside one does not see it.
        client = self.mirror.client_id
        top = max((p.getAABB(body, link, physicsClientId=client)[1][2] for body in others
                   for link in range(-1, p.getNumJoints(body, physicsClientId=client))), default=bottom) + 0.1
        x, y = np.meshgrid(low[0] + cell * np.arange(shape[0]), low[1] + cell * np.arange(shape[1]), indexing="ij")
        offsets = cell * ((np.arange(FLOOR_SAMPLES) + 0.5) / FLOOR_SAMPLES - 0.5)
        heights = np.full(shape, -np.inf)
        for dx in offsets:
            for dy in offsets:
                hit = self._rays(np.column_stack([x.ravel() + dx, y.ravel() + dy]), top, bottom, others)
                heights = np.fmax(heights, np.where(np.isnan(hit), -np.inf, hit).reshape(shape))
        return heights

    def robot_heights(self, own: str, cell: float, bottom: float) -> tuple[np.ndarray, np.ndarray]:
        """How high and how low robot `own` and what it holds reach above each cell around its base, at yaw 0.

        Args:
            own: The robot, by id, its arms as they stand; moved out of the way of everything else.
            cell: Cell size, metres.
            bottom: Lowest height that counts, metres.

        Returns:
            tuple[np.ndarray, np.ndarray]: Its highest and lowest surface per cell, metres (-inf and inf where it is
            not), on a square grid of odd size whose centre cell is base_footprint, x along the first axis.
        """
        top = self._park(own)
        body = self.mirror.robot(own)
        x, y = p.getBasePositionAndOrientation(body, physicsClientId=self.mirror.client_id)[0][:2]
        n = int(math.ceil(self._reach.get(own, 0.0) / cell)) + 1
        dx, dy = np.meshgrid(cell * np.arange(-n, n + 1), cell * np.arange(-n, n + 1), indexing="ij")
        xy = np.column_stack([x + dx.ravel(), y + dy.ravel()])
        bodies = {body for part in self._parts(own) for body in self.mirror.body_ids(part)}
        highest, lowest = self._rays(xy, top, bottom, bodies), self._rays(xy, bottom, top, bodies)
        return (np.where(np.isnan(highest), -np.inf, highest).reshape(dx.shape),
                np.where(np.isnan(lowest), np.inf, lowest).reshape(dx.shape))

    def _park(self, own: str) -> float:
        """Move robot `own` and what it holds far from everything, at yaw 0; the height of their top, metres."""
        client = self.mirror.client_id
        self._place(own, (PARK, PARK, 0.0))
        return max(p.getAABB(body, link, physicsClientId=client)[1][2]
                   for part in self._parts(own) for body in self.mirror.body_ids(part)
                   for link in range(-1, p.getNumJoints(body, physicsClientId=client)))

    def _rays(self, xy: np.ndarray, start: float, end: float, bodies: set[int]) -> np.ndarray:
        """Cast a vertical ray at each (x, y) from height `start` to `end`, and return the heights where they hit.

        A ray's height is nan where it hits nothing, or something not in `bodies` first.
        """
        heights = []
        for i in range(0, len(xy), RAY_BATCH):
            chunk = xy[i:i + RAY_BATCH]
            hits = p.rayTestBatch(np.column_stack([chunk, np.full(len(chunk), start)]).tolist(),
                                  np.column_stack([chunk, np.full(len(chunk), end)]).tolist(),
                                  physicsClientId=self.mirror.client_id)
            heights += [hit[3][2] if hit[0] in bodies else math.nan for hit in hits]
        return np.array(heights, dtype=float)
