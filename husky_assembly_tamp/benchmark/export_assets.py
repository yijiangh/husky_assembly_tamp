"""Export the robot so a browser can draw it in any configuration.

Two files come out of this, and they are meant to be read together.

``scene.glb`` holds the geometry, arranged as the robot's own joint tree: each
link is a node positioned relative to its parent by that joint's fixed offset,
with its meshes hanging underneath. Posing the robot is then a matter of
setting the rotation of twelve of those nodes.

``kinematics.json`` says which twelve, in which order, and about which axis --
the information the browser needs to turn a list of joint angles into a pose.

The alternative, sending the browser a matrix per link per frame, was
considered and rejected: a single planned motion would be megabytes, and the
dashboard shows thousands of them.

Both files depend only on the robot, not on any benchmark run, so they are
written once per robot and shared by every run.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from husky_assembly_tamp.benchmark.glb import (
    GlbBuilder, frame_to_translation_rotation, mesh_arrays,
)
from husky_assembly_tamp.benchmark.scene import load_trimmed_problem


# * Colours, chosen so the two arms can be told apart at a glance and the bar
# * stands out against all of them. Red, green, blue, alpha.
COLOURS: Dict[str, Tuple[float, float, float, float]] = {
    "base": (0.62, 0.64, 0.66, 1.0),
    "left_arm": (0.36, 0.55, 0.72, 1.0),
    "right_arm": (0.72, 0.52, 0.36, 1.0),
    "tool": (0.32, 0.34, 0.36, 1.0),
    "bar": (0.12, 0.52, 0.74, 1.0),
    "joint": (0.85, 0.72, 0.25, 1.0),
}


def link_colour(link_name: str) -> str:
    """Return which colour group a robot link belongs to.

    Args:
        link_name (str): the link's name.

    Returns:
        str: a key of :data:`COLOURS`.
    """
    if link_name.startswith("left_"):
        return "left_arm"
    if link_name.startswith("right_"):
        return "right_arm"
    return "base"


def export_robot_assets(root: str, source_key: str, bar_id: str, *,
                        out_dir: Optional[str] = None,
                        verbose: bool = True) -> Dict[str, Any]:
    """Write ``scene.glb`` and ``kinematics.json`` for the benchmark robot.

    Args:
        root (str): the benchmark data root.
        source_key (str): which trimmed-scene cache to read the robot from.
        bar_id (str): any bar; only the robot and tools are exported, and they
            are the same in every trimmed scene.
        out_dir (Optional[str]): where to write; ``<root>/assets/<source_key>``
            by default.
        verbose (bool): print what was written.

    Returns:
        Dict[str, Any]: the paths written and the sizes involved.
    """
    problem, cell, _template = load_trimmed_problem(root, source_key, bar_id)
    out_dir = out_dir or os.path.join(root, "assets", source_key)
    os.makedirs(out_dir, exist_ok=True)

    model = cell.robot_model
    builder = GlbBuilder()
    materials = {name: builder.add_material(name, colour)
                 for name, colour in COLOURS.items()}

    # Which joint moves which link, so a link node can carry its joint's offset.
    joint_by_child: Dict[str, Any] = {joint.child.link: joint for joint in model.joints}
    children_of: Dict[str, List[str]] = {}
    for joint in model.joints:
        children_of.setdefault(joint.parent.link, []).append(joint.child.link)

    link_by_name = {link.name: link for link in model.links}
    root_links = [link.name for link in model.links if link.name not in joint_by_child]

    node_of: Dict[str, int] = {}
    n_triangles = 0

    def build_link(link_name: str) -> int:
        """Create this link's node and everything below it.

        Args:
            link_name (str): which link.

        Returns:
            int: the node index for this link.
        """
        nonlocal n_triangles
        link = link_by_name[link_name]
        joint = joint_by_child.get(link_name)
        translation, rotation = frame_to_translation_rotation(
            joint.origin if joint is not None else None)

        mesh_nodes: List[int] = []
        for index, visual in enumerate(link.visual or []):
            shape = visual.geometry.shape
            meshes = list(getattr(shape, "meshes", None) or [])
            if not meshes:
                continue
            scale = getattr(shape, "scale", None)
            visual_translation, visual_rotation = frame_to_translation_rotation(
                visual.origin)
            for mesh_index, mesh in enumerate(meshes):
                vertices, faces = mesh_arrays(mesh)
                if faces.size == 0:
                    continue
                if scale is not None:
                    vertices = vertices * np.asarray(scale, dtype=np.float32)
                n_triangles += faces.shape[0]
                mesh_id = builder.add_mesh(
                    vertices, faces, materials[link_colour(link_name)])
                mesh_nodes.append(builder.add_node(
                    f"V_{link_name}_{index}_{mesh_index}",
                    translation=visual_translation,
                    rotation=visual_rotation,
                    mesh=mesh_id,
                ))

        child_nodes = [build_link(child) for child in children_of.get(link_name, [])]
        # ! Node names carry no slashes or dots: the glTF loader rewrites those
        # ! characters, and the page looks nodes up by name.
        node = builder.add_node(f"L_{link_name}", translation=translation,
                                rotation=rotation)
        builder.set_children(node, mesh_nodes + child_nodes)
        node_of[link_name] = node
        return node

    roots = [build_link(name) for name in root_links]

    # The tools ride on the arms' tool0 links, so they hang off those nodes.
    for tool_name, tool_model in (cell.tool_models or {}).items():
        attach_link = ("left_ur_arm_tool0" if tool_name.endswith("L")
                       else "right_ur_arm_tool0")
        if attach_link not in node_of:
            continue
        tool_nodes: List[int] = []
        for link in tool_model.links:
            for index, visual in enumerate(link.visual or []):
                meshes = list(getattr(visual.geometry.shape, "meshes", None) or [])
                visual_translation, visual_rotation = frame_to_translation_rotation(
                    visual.origin)
                for mesh_index, mesh in enumerate(meshes):
                    vertices, faces = mesh_arrays(mesh)
                    if faces.size == 0:
                        continue
                    n_triangles += faces.shape[0]
                    mesh_id = builder.add_mesh(vertices, faces, materials["tool"])
                    tool_nodes.append(builder.add_node(
                        f"T_{tool_name}_{link.name}_{index}_{mesh_index}",
                        translation=visual_translation,
                        rotation=visual_rotation,
                        mesh=mesh_id,
                    ))
        if tool_nodes:
            existing = builder.nodes[node_of[attach_link]].get("children", [])
            builder.set_children(node_of[attach_link], list(existing) + tool_nodes)

    glb_path = os.path.join(out_dir, "scene.glb")
    size = builder.write(glb_path, roots)

    kinematics = {
        "schema_version": 1,
        "units": "m",
        "up": "z",
        "asset": "scene.glb",
        "joint_order": problem.joint_names_12,
        "joints": [_joint_record(model, name) for name in problem.joint_names_12],
        "tool_links": {
            "left": "left_ur_arm_tool0",
            "right": "right_ur_arm_tool0",
        },
    }
    kinematics_path = os.path.join(out_dir, "kinematics.json")
    with open(kinematics_path, "w", encoding="utf-8") as handle:
        json.dump(kinematics, handle, indent=2)

    if verbose:
        print(f"[assets] {glb_path} ({size / 1e6:.1f} MB, "
              f"{n_triangles} triangles, {len(builder.meshes)} distinct meshes)")
        print(f"[assets] {kinematics_path} ({len(problem.joint_names_12)} joints)")
    return {"glb": glb_path, "kinematics": kinematics_path,
            "bytes": size, "n_triangles": n_triangles}


def _joint_record(model, joint_name: str) -> Dict[str, Any]:
    """Describe one movable joint for the browser.

    Args:
        model: the compas robot model.
        joint_name (str): which joint.

    Returns:
        Dict[str, Any]: its node name, rotation axis, fixed offset and limits.

    Raises:
        ValueError: if the model has no such joint. The dashboard cannot pose
            a robot it only half understands, so this is not tolerated.
    """
    joint = next((j for j in model.joints if j.name == joint_name), None)
    if joint is None:
        raise ValueError(
            f"the robot model has no joint named {joint_name!r}, so the "
            "dashboard cannot be told how to pose it"
        )
    translation, rotation = frame_to_translation_rotation(joint.origin)
    limit = getattr(joint, "limit", None)
    return {
        "name": joint_name,
        "node": f"L_{joint.child.link}",
        "child_link": joint.child.link,
        "parent_link": joint.parent.link,
        "axis": [float(v) for v in joint.axis.vector],
        "translation": translation,
        "rotation": rotation,
        "lower": float(getattr(limit, "lower", -np.pi)) if limit else -float(np.pi),
        "upper": float(getattr(limit, "upper", np.pi)) if limit else float(np.pi),
    }
