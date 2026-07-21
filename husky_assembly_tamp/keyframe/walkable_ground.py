"""Rhino-free WalkableGround geometry + headless base sampling.

This module is the plain-python (numpy only) half of the WalkableGround feature.
It has NO Rhino imports so the headless keyframe solver can use it. It holds:

- the two small base-frame helpers shared with the Rhino front-end
  (``sample_base_offsets`` + ``frame_from_origin_normal_heading``); the design
  repo's ``rs_ik_keyframe`` imports them from here so there is one definition;
- loading the exported ``WalkableGround.json`` into fast numpy "triangle soups";
- the headless twin of the Rhino ``_snap_to_brep`` -- closest point + normal on a
  mesh (``closest_point_on_meshes``);
- deriving an automatic seed base on the ground (there is no human pick offline);
- ``solve_chain_with_base_search`` -- the expanding-radius sampling loop that
  mirrors the Rhino ``_solve_chain_with_sampling`` but snaps to meshes and uses an
  auto-derived seed.

The actual IK chain solve is delegated to ``keyframe.ik_keyframe.solve_keyframe_chain``
(the one shared solver), so the accept rule here is identical to the Rhino side.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from compas.data import json_load

from husky_assembly_tamp.keyframe import config

# NOTE: ``solve_keyframe_chain`` is imported lazily inside
# ``solve_chain_with_base_search`` -- it pulls in the pybullet/compas_fab robot
# stack, which the mesh-loading + closest-point helpers here do NOT need. Keeping
# the import local lets those helpers be used (and unit-tested) on their own.


# A "triangle soup" is a bare mesh reduced to what closest-point needs: an
# (V, 3) vertex array (mm) and an (T, 3) int array of triangle vertex indices.
TriangleSoup = Tuple[np.ndarray, np.ndarray]


# ---------------------------------------------------------------------------
# Small vector helpers (shared with the Rhino front-end -- one copy, here)
# ---------------------------------------------------------------------------


def _unit(vector: np.ndarray) -> np.ndarray:
    """Return the unit-length version of *vector*.

    Args:
        vector (np.ndarray): any non-zero 3-vector.

    Returns:
        np.ndarray: the same direction scaled to length 1.

    Raises:
        ValueError: if *vector* is (numerically) zero-length.
    """
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError("Cannot unitize a zero-length vector.")
    return np.asarray(vector, dtype=float) / norm


def sample_base_offsets(count: int, radius_mm: float):
    """Yield random 2D offset vectors inside a disk in the base tangent plane.

    Each offset is a point picked uniformly-ish in a circle of ``radius_mm`` and
    returned as a 3-vector with z = 0 (the offset lives in the ground's tangent
    plane; the caller adds it to the seed origin and re-snaps to the surface).

    Args:
        count (int): how many offsets to yield.
        radius_mm (float): disk radius in millimeters.

    Yields:
        np.ndarray: a ``[dx, dy, 0]`` offset in millimeters.
    """
    rng = random.Random()
    for _ in range(count):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        r = rng.uniform(0.0, radius_mm)
        yield np.array([r * math.cos(angle), r * math.sin(angle), 0.0], dtype=float)


def frame_from_origin_normal_heading(origin_mm, normal, heading_mm) -> np.ndarray:
    """Build a 4x4 base frame: Z = ground normal, X = heading projected flat.

    The base sits at ``origin_mm`` with its up-axis (Z) along the ground
    ``normal`` and its forward-axis (X) pointing toward ``heading_mm``, but with
    that heading projected onto the plane perpendicular to Z so the frame stays
    orthonormal.

    Args:
        origin_mm (np.ndarray): base origin in millimeters (3-vector).
        normal (np.ndarray): ground surface normal at the origin (3-vector).
        heading_mm (np.ndarray): a world point (mm) the base +X should face.

    Returns:
        np.ndarray: a 4x4 homogeneous transform (mm translation).

    Raises:
        RuntimeError: if the heading direction is collinear with the normal, so
            no in-plane X-axis can be defined.
    """
    z = _unit(normal)
    heading_vec = np.asarray(heading_mm, dtype=float) - np.asarray(origin_mm, dtype=float)
    x_raw = heading_vec - np.dot(heading_vec, z) * z
    if float(np.linalg.norm(x_raw)) < 1e-6:
        raise RuntimeError("Heading point is collinear with base normal; pick a different point.")
    x = _unit(x_raw)
    y = np.cross(z, x)
    frame = np.eye(4, dtype=float)
    frame[:3, 0] = x
    frame[:3, 1] = y
    frame[:3, 2] = z
    frame[:3, 3] = np.asarray(origin_mm, dtype=float)
    return frame


# ---------------------------------------------------------------------------
# WalkableGround.json loading -> triangle soups
# ---------------------------------------------------------------------------


def _mesh_to_soup(mesh) -> TriangleSoup:
    """Convert one compas ``Mesh`` into a numpy ``(vertices, triangles)`` soup.

    Quad (or n-gon) faces are fan-triangulated so the closest-point code only
    ever deals with triangles.

    Args:
        mesh (compas.datastructures.Mesh): a loaded ground mesh (vertices in mm).

    Returns:
        TriangleSoup: ``(vertices (V,3) float, triangles (T,3) int)``.
    """
    # Stable vertex ordering + a lookup so face vertex keys map to row indices.
    vertex_keys = list(mesh.vertices())
    key_to_row = {key: row for row, key in enumerate(vertex_keys)}
    vertices = np.array(
        [mesh.vertex_coordinates(key) for key in vertex_keys], dtype=float
    )

    triangles: List[List[int]] = []
    for face in mesh.faces():
        rows = [key_to_row[key] for key in mesh.face_vertices(face)]
        # Fan-triangulate: (0,1,2), (0,2,3), ... for any face with >= 3 vertices.
        for i in range(1, len(rows) - 1):
            triangles.append([rows[0], rows[i], rows[i + 1]])
    return vertices, np.array(triangles, dtype=int)


def load_walkable_grounds(path: str) -> Dict[str, TriangleSoup]:
    """Load ``WalkableGround.json`` and convert every ground to a triangle soup.

    Args:
        path (str): path to the exported ``WalkableGround.json``
            (``{"grounds": {ground_id: <compas Mesh>}}``).

    Returns:
        dict[str, TriangleSoup]: ``{ground_id: (vertices, triangles)}`` in mm.
    """
    data = json_load(path)
    grounds = data.get("grounds", {}) if isinstance(data, dict) else {}
    return {gid: _mesh_to_soup(mesh) for gid, mesh in grounds.items()}


# ---------------------------------------------------------------------------
# Closest point on a mesh (headless twin of the Rhino _snap_to_brep)
# ---------------------------------------------------------------------------


def _closest_point_on_triangle(p, a, b, c) -> np.ndarray:
    """Return the closest point to ``p`` on triangle ``(a, b, c)``.

    Standard Voronoi-region test (Ericson, Real-Time Collision Detection): the
    closest point is a vertex, an edge point, or an interior (barycentric) point.

    Args:
        p (np.ndarray): the query point (3-vector).
        a, b, c (np.ndarray): the triangle corners (3-vectors).

    Returns:
        np.ndarray: the closest point on the triangle to ``p``.
    """
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a  # closest to vertex a

    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b  # closest to vertex b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab  # closest to edge ab

    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c  # closest to vertex c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac  # closest to edge ac

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)  # closest to edge bc

    # Interior: convert the barycentric weights into a point.
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w


def _closest_point_on_soup(soup: TriangleSoup, point_mm) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Closest point + up-oriented normal on one triangle soup.

    Args:
        soup (TriangleSoup): ``(vertices, triangles)`` in mm.
        point_mm (np.ndarray): the query point in mm.

    Returns:
        tuple: ``(closest_point_mm, normal)`` or ``(None, None)`` if the soup has
        no triangles. The normal is the triangle's geometric normal flipped so it
        points up (positive world Z) -- WalkableGround surfaces are horizontal-ish
        and the robot stands on top, so "up" is the meaningful base Z regardless
        of mesh winding.
    """
    vertices, triangles = soup
    if triangles.size == 0:
        return None, None

    p = np.asarray(point_mm, dtype=float)
    best_pt = None
    best_dist = None
    best_tri = None
    for tri in triangles:
        a, b, c = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        close = _closest_point_on_triangle(p, a, b, c)
        dist = float(np.linalg.norm(close - p))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_pt = close
            best_tri = (a, b, c)

    a, b, c = best_tri
    normal = np.cross(b - a, c - a)
    if float(np.linalg.norm(normal)) < 1e-9:
        normal = np.array([0.0, 0.0, 1.0])  # degenerate triangle -> assume flat up
    else:
        normal = _unit(normal)
    if normal[2] < 0.0:
        normal = -normal  # keep the base standing up, not hanging under the ground
    return best_pt, normal


def closest_point_on_meshes(soups: Sequence[TriangleSoup], point_mm) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Closest point + normal across several ground soups (nearest wins).

    Args:
        soups (Sequence[TriangleSoup]): the associated ground soups to snap to.
        point_mm (np.ndarray): the query point in mm.

    Returns:
        tuple: ``(closest_point_mm, normal)`` for the nearest surface, or
        ``(None, None)`` if no soup yields a point.
    """
    best_pt = None
    best_normal = None
    best_dist = None
    p = np.asarray(point_mm, dtype=float)
    for soup in soups:
        pt, normal = _closest_point_on_soup(soup, p)
        if pt is None:
            continue
        dist = float(np.linalg.norm(pt - p))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_pt = pt
            best_normal = normal
    return best_pt, best_normal


# ---------------------------------------------------------------------------
# Seed derivation + expanding-radius base search
# ---------------------------------------------------------------------------


def _ground_heading_direction(heading_dir_mm, normal) -> np.ndarray:
    """Unit heading direction: *heading_dir_mm* flattened onto the ground plane.

    ``heading_dir_mm`` is the average male-joint insertion direction (a world
    vector). We drop its component along the ground normal so it lies in the
    ground plane, then normalize. Falls back to a world horizontal when the
    insertion direction is missing or (near-)vertical, so a base frame can always
    be built.

    Args:
        heading_dir_mm (np.ndarray | None): the world heading direction (the
            average male-joint insertion direction).
        normal (np.ndarray): the ground normal (base Z) at the base origin.

    Returns:
        np.ndarray: a unit direction lying in the ground plane.
    """
    z = _unit(normal)
    if heading_dir_mm is not None:
        horizontal = np.asarray(heading_dir_mm, dtype=float)
        horizontal = horizontal - np.dot(horizontal, z) * z
        if float(np.linalg.norm(horizontal)) > 1e-6:
            return _unit(horizontal)
    # Insertion direction vertical / unknown -> any horizontal on the ground.
    for world in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
        horizontal = world - np.dot(world, z) * z
        if float(np.linalg.norm(horizontal)) > 1e-6:
            return _unit(horizontal)
    return np.array([1.0, 0.0, 0.0])


def derive_seed_base(soups: Sequence[TriangleSoup], bar_midpoint_mm,
                     heading_dir_mm=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick an automatic seed base: stand behind the bar, face the insertion axis.

    Offline there is no human pick. The base FACES along ``heading_dir_mm`` -- the
    average male-joint insertion direction, i.e. the way the bar is pushed to mate
    -- and stands the standoff distance (``config.IK_BASE_STANDOFF_MM``) BEHIND the
    bar's ground projection (opposite the heading), so moving forward (+X) carries
    the bar into the assembly. Using the insertion direction (rather than "a side
    of the bar") removes the left/right ambiguity of which side to stand on.

    Args:
        soups (Sequence[TriangleSoup]): the bar's associated ground soups.
        bar_midpoint_mm (np.ndarray): the bar center in mm; the base stands behind,
            and faces, this point's ground projection.
        heading_dir_mm (np.ndarray | None): the average male-joint insertion
            direction (world vector). ``None`` -> a world-horizontal fallback.

    Returns:
        tuple: ``(seed_origin_mm, seed_normal, heading_dir)`` -- ``heading_dir`` is
        the unit ground-plane direction the base +X should face. Build the frame
        with ``frame_from_origin_normal_heading(origin, normal, origin + heading_dir * d)``.

    Raises:
        RuntimeError: if no ground point can be found (empty / missing soups).
    """
    bar_ground_point, ground_normal = closest_point_on_meshes(soups, bar_midpoint_mm)
    if bar_ground_point is None:
        raise RuntimeError("No walkable-ground surface to seed a base on.")

    # Stand the standoff BEHIND the bar (opposite the insertion direction), then
    # re-snap that point back onto the ground (it may sit on a different triangle).
    heading_dir = _ground_heading_direction(heading_dir_mm, ground_normal)
    behind_point = np.asarray(bar_ground_point, dtype=float) - heading_dir * float(config.IK_BASE_STANDOFF_MM)
    origin, normal = closest_point_on_meshes(soups, behind_point)
    if origin is None:
        # Offset landed off the ground -> fall back to standing under the bar.
        origin, normal = bar_ground_point, ground_normal

    # Re-flatten the heading onto the (snapped) base's own ground plane.
    heading_dir = _ground_heading_direction(heading_dir_mm, normal)
    return np.asarray(origin, dtype=float), normal, heading_dir


def solve_chain_with_base_search(
    planner,
    movements: dict,
    soups: Sequence[TriangleSoup],
    bar_midpoint_mm,
    *,
    heading_dir_mm=None,
    radius_mm: float = None,
    max_iter: int = None,
    rings: int = 3,
    check_collision: bool = True,
    verbose: bool = True,
    groups=None,
    home_conf_12=None,
) -> Tuple[Optional[dict], Optional[np.ndarray]]:
    """Headless base search: auto-seed, then grow the radius until the chain solves.

    Mirrors the Rhino ``_solve_chain_with_sampling`` but with no Rhino: the seed
    comes from ``derive_seed_base`` and each sampled origin is snapped to the
    ground meshes (``closest_point_on_meshes``) instead of a Rhino brep. A base is
    accepted only when the whole M1->M2->M3 chain solves, via the shared
    ``keyframe.ik_keyframe.solve_keyframe_chain``.

    Args:
        planner (PyBulletPlanner): active dual-arm planner.
        movements (dict): ``{"M1": .., "M2": .., "M3": ..}`` from the BarAction.
        soups (Sequence[TriangleSoup]): the bar's associated ground soups.
        bar_midpoint_mm (np.ndarray): the bar center in mm; every base stands
            behind and faces this point's ground projection.
        heading_dir_mm (np.ndarray | None): the average male-joint insertion
            direction (world vector); every base faces along it.
        radius_mm (float): base of the sampling radius; the ``i``-th ring uses
            ``radius_mm * i``. Defaults to ``config.IK_BASE_SAMPLE_RADIUS``.
        max_iter (int): samples per ring. Defaults to
            ``config.IK_BASE_SAMPLE_MAX_ITER``.
        rings (int): how many growing rings to try after the seed.
        check_collision (bool): pass-through to the chain solver.
        verbose (bool): print per-attempt progress.
        groups (tuple[str, str] | None): pass-through planning-group names.
        home_conf_12 (Sequence[float] | None): pass-through home configuration
            (needed by the ssik backend on the cold first solve).

    Returns:
        tuple: ``(solved_states, used_base_frame_mm)`` on success, else
        ``(None, None)``.
    """
    # Local import: only the base search needs the robot stack (see module note).
    from husky_assembly_tamp.keyframe.ik_keyframe import solve_keyframe_chain

    if radius_mm is None:
        radius_mm = float(config.IK_BASE_SAMPLE_RADIUS)
    if max_iter is None:
        max_iter = int(config.IK_BASE_SAMPLE_MAX_ITER)

    ordered = [("M1", movements["M1"]), ("M2", movements["M2"]), ("M3", movements["M3"])]

    seed_origin, seed_normal, seed_heading_dir = derive_seed_base(
        soups, bar_midpoint_mm, heading_dir_mm=heading_dir_mm,
    )
    seed_frame = frame_from_origin_normal_heading(
        seed_origin, seed_normal, seed_origin + seed_heading_dir * 1000.0
    )

    def _try(base_frame_mm, label: str):
        if verbose:
            o = base_frame_mm[:3, 3]
            print(
                f"keyframe.walkable_ground.solve_chain_with_base_search: trying base "
                f"({label}) at ({o[0]:.1f}, {o[1]:.1f}, {o[2]:.1f}) mm ..."
            )
        return solve_keyframe_chain(
            planner, ordered, base_frame_mm,
            check_collision=check_collision, verbose_pairs=False,
            groups=groups, home_conf_12=home_conf_12,
        )

    # Attempt 0: the seed itself.
    solved = _try(seed_frame, "seed")
    if solved is not None:
        return solved, seed_frame

    # Rings of growing radius; each ring re-snaps its random offsets to the ground.
    for ring in range(1, rings + 1):
        ring_radius = radius_mm * ring
        for idx, offset in enumerate(sample_base_offsets(max_iter, ring_radius), start=1):
            sample_origin_mm = seed_origin + offset
            snapped_origin, normal = closest_point_on_meshes(soups, sample_origin_mm)
            if snapped_origin is None:
                continue
            # Every base faces the SAME insertion direction (re-flattened onto its
            # own ground patch), not a fixed heading point.
            sample_heading_dir = _ground_heading_direction(heading_dir_mm, normal)
            try:
                sample_frame = frame_from_origin_normal_heading(
                    snapped_origin, normal, snapped_origin + sample_heading_dir * 1000.0
                )
            except RuntimeError:
                continue
            solved = _try(sample_frame, f"ring {ring}/{rings} sample {idx}/{max_iter}")
            if solved is not None:
                return solved, sample_frame

    if verbose:
        print(
            "keyframe.walkable_ground.solve_chain_with_base_search: [X] no base solved the "
            f"chain across {rings} ring(s) x {max_iter} sample(s)."
        )
    return None, None
