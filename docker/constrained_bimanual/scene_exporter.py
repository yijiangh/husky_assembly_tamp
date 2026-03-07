"""
Export collision scene from PyBullet to a format consumable by the Drake
constrained bimanual planner.

This module queries PyBullet for static collision objects (walls, planes,
boxes, cylinders, spheres) and generates a JSON description that can be
passed to the Drake planner via the request file.

Usage from your pybullet code:
    from scene_exporter import export_collision_scene
    collision_objects = export_collision_scene(monitor.static_obstacles)
    request["collision_objects"] = collision_objects

The Drake planner reads this and creates matching collision geometry.
"""

import json
import numpy as np

try:
    import pybullet as p
    import pybullet_planning as pp
except ImportError:
    p = None
    pp = None


# PyBullet geometry type constants
GEOM_SPHERE = 2
GEOM_BOX = 3
GEOM_CYLINDER = 4
GEOM_MESH = 5
GEOM_PLANE = 6
GEOM_CAPSULE = 7


def _pose_to_matrix(position, orientation):
    """Convert pybullet position + quaternion to 4x4 matrix."""
    from scipy.spatial.transform import Rotation
    mat = np.eye(4)
    mat[:3, :3] = Rotation.from_quat(orientation).as_matrix()
    mat[:3, 3] = position
    return mat


def export_body_collision(body_id, name="unnamed", client_id=0):
    """
    Export a single PyBullet body's collision geometry.

    Args:
        body_id: PyBullet body ID
        name: human-readable name for the object
        client_id: PyBullet physics client ID

    Returns:
        Dict describing the collision object, or None if unsupported geometry.
        Format:
        {
            "name": str,
            "type": "box" | "cylinder" | "sphere" | "plane",
            "dims": [...],  # type-dependent dimensions
            "pose": [[4x4 matrix as nested list]]
        }
    """
    # Get body pose
    pos, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=client_id)
    body_pose = _pose_to_matrix(pos, orn)

    # Get collision shape data
    shape_data = p.getCollisionShapeData(body_id, -1, physicsClientId=client_id)

    if len(shape_data) == 0:
        return None

    # Use first collision shape (most bodies have one)
    shape = shape_data[0]
    geom_type = shape[2]
    dims = shape[3]  # geometry-dependent dimensions
    local_pos = shape[5]
    local_orn = shape[6]

    # Combine body pose with local collision frame offset
    local_pose = _pose_to_matrix(local_pos, local_orn)
    world_pose = body_pose @ local_pose

    if geom_type == GEOM_BOX:
        # dims = half extents (x, y, z)
        return {
            "name": name,
            "type": "box",
            "dims": [dims[0] * 2, dims[1] * 2, dims[2] * 2],  # full extents
            "pose": world_pose.tolist(),
        }
    elif geom_type == GEOM_CYLINDER:
        # dims = (radius, height, ?)
        # PyBullet cylinder: dims[1] is half-height for some APIs, check
        radius = dims[1] if dims[1] < dims[0] else dims[0]
        height = dims[0] if dims[1] < dims[0] else dims[1]
        return {
            "name": name,
            "type": "cylinder",
            "dims": [radius, height],
            "pose": world_pose.tolist(),
        }
    elif geom_type == GEOM_SPHERE:
        return {
            "name": name,
            "type": "sphere",
            "dims": [dims[0]],  # radius
            "pose": world_pose.tolist(),
        }
    elif geom_type == GEOM_PLANE:
        # Represent as a large thin box (Drake doesn't have infinite planes)
        # Normal is along Z by default in PyBullet
        return {
            "name": name,
            "type": "box",
            "dims": [20.0, 20.0, 0.01],  # large thin box
            "pose": world_pose.tolist(),
        }
    elif geom_type == GEOM_MESH:
        # Mesh export is complex; for now, log a warning
        mesh_file = shape[4].decode("utf-8") if isinstance(shape[4], bytes) else shape[4]
        print(f"Warning: Mesh collision shape for '{name}' (file={mesh_file}) "
              f"not yet supported in scene export. Skipping.")
        return None
    else:
        print(f"Warning: Unknown geometry type {geom_type} for '{name}'. Skipping.")
        return None


def export_collision_scene(static_obstacles, client_id=0):
    """
    Export all static obstacles from a PyBullet scene.

    Args:
        static_obstacles: dict mapping name -> pybullet body ID
            (as stored in monitor.static_obstacles)
        client_id: PyBullet physics client ID (default: pp.CLIENT if available)

    Returns:
        List of collision object dicts suitable for JSON serialization.
    """
    if pp is not None:
        client_id = pp.CLIENT

    objects = []
    for name, body_id in static_obstacles.items():
        obj = export_body_collision(body_id, name=name, client_id=client_id)
        if obj is not None:
            objects.append(obj)
            print(f"  Exported: {name} ({obj['type']}, dims={obj['dims']})")
        else:
            print(f"  Skipped: {name} (unsupported geometry)")

    return objects


def save_collision_scene(static_obstacles, filepath, client_id=0):
    """
    Export collision scene and save to a JSON file.

    Args:
        static_obstacles: dict mapping name -> pybullet body ID
        filepath: output JSON file path
        client_id: PyBullet physics client ID
    """
    objects = export_collision_scene(static_obstacles, client_id)

    with open(filepath, "w") as f:
        json.dump({"collision_objects": objects}, f, indent=2)

    print(f"Collision scene saved to {filepath} ({len(objects)} objects)")
    return objects


def load_collision_scene(filepath):
    """
    Load collision scene from a JSON file.

    Returns:
        List of collision object dicts.
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    return data.get("collision_objects", [])
