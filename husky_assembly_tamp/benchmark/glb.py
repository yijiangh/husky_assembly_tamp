"""Write a GLB file: robot geometry a browser can load in one request.

The dashboard draws the robot in 3D, which means getting its meshes into the
page. GLB is the one format browsers read natively, and it packs the whole
scene -- geometry, hierarchy, materials -- into a single binary file.

This writes glTF by hand rather than pulling in a library. The format is small
enough for that (a header, a JSON description and one binary blob) and it keeps
the benchmark's dependencies to what the planner already needs.

The node hierarchy deliberately mirrors the robot's joint tree, so the browser
can pose the robot by setting twelve rotations instead of being sent a matrix
for every link of every frame. That is the difference between a trajectory
costing forty kilobytes and costing two megabytes.
"""
from __future__ import annotations

import json
import struct
from hashlib import sha1
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# glTF constants, named so the code below reads as something other than magic
# numbers.
COMPONENT_FLOAT = 5126
COMPONENT_UNSIGNED_INT = 5125
TARGET_ARRAY_BUFFER = 34962
TARGET_ELEMENT_ARRAY_BUFFER = 34963


class GlbBuilder:
    """Collect meshes and nodes, then write them out as one GLB file.

    Meshes are de-duplicated by content, which matters here: the two arms are
    the same robot mirrored, so every arm mesh would otherwise be stored twice.

    Attributes:
        nodes (List[Dict[str, Any]]): the glTF node list being built.
        materials (List[Dict[str, Any]]): the glTF material list.
    """

    def __init__(self):
        """Start an empty scene."""
        self.buffer = bytearray()
        self.buffer_views: List[Dict[str, Any]] = []
        self.accessors: List[Dict[str, Any]] = []
        self.meshes: List[Dict[str, Any]] = []
        self.nodes: List[Dict[str, Any]] = []
        self.materials: List[Dict[str, Any]] = []
        self._mesh_cache: Dict[str, int] = {}

    def _align(self) -> None:
        """Pad the buffer so the next view starts on a four-byte boundary."""
        while len(self.buffer) % 4:
            self.buffer.append(0)

    def _add_view(self, data: bytes, target: Optional[int] = None) -> int:
        """Append raw bytes to the buffer and return their view index.

        Args:
            data (bytes): the bytes to append.
            target (Optional[int]): the glTF buffer target, when it matters.

        Returns:
            int: the index of the new buffer view.
        """
        self._align()
        view = {"buffer": 0, "byteOffset": len(self.buffer), "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self.buffer.extend(data)
        self.buffer_views.append(view)
        return len(self.buffer_views) - 1

    def add_material(self, name: str, colour: Sequence[float],
                     *, metallic: float = 0.05, roughness: float = 0.6) -> int:
        """Add a plain coloured material.

        Args:
            name (str): a name, for debugging in glTF viewers.
            colour (Sequence[float]): red, green, blue and alpha, each 0 to 1.
            metallic (float): how metallic the surface looks.
            roughness (float): how rough the surface looks.

        Returns:
            int: the index of the new material.
        """
        self.materials.append({
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": [float(v) for v in colour],
                "metallicFactor": float(metallic),
                "roughnessFactor": float(roughness),
            },
            "doubleSided": True,
        })
        return len(self.materials) - 1

    def add_mesh(self, vertices: np.ndarray, faces: np.ndarray, material: int) -> int:
        """Add a triangle mesh, reusing an identical one if it is already here.

        Args:
            vertices (np.ndarray): an ``(n, 3)`` array of positions, metres.
            faces (np.ndarray): an ``(m, 3)`` array of vertex indices.
            material (int): which material to draw it with.

        Returns:
            int: the index of the mesh.
        """
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        faces = np.ascontiguousarray(faces, dtype=np.uint32)
        key = sha1(vertices.tobytes() + faces.tobytes()
                   + str(material).encode()).hexdigest()
        if key in self._mesh_cache:
            return self._mesh_cache[key]

        position_view = self._add_view(vertices.tobytes(), TARGET_ARRAY_BUFFER)
        self.accessors.append({
            "bufferView": position_view,
            "componentType": COMPONENT_FLOAT,
            "count": int(vertices.shape[0]),
            "type": "VEC3",
            # glTF requires the bounds of a position accessor; viewers use them
            # to frame the scene without reading every vertex.
            "min": [float(v) for v in vertices.min(axis=0)],
            "max": [float(v) for v in vertices.max(axis=0)],
        })
        position_accessor = len(self.accessors) - 1

        index_view = self._add_view(faces.tobytes(), TARGET_ELEMENT_ARRAY_BUFFER)
        self.accessors.append({
            "bufferView": index_view,
            "componentType": COMPONENT_UNSIGNED_INT,
            "count": int(faces.size),
            "type": "SCALAR",
        })
        index_accessor = len(self.accessors) - 1

        self.meshes.append({"primitives": [{
            "attributes": {"POSITION": position_accessor},
            "indices": index_accessor,
            "material": material,
        }]})
        mesh_index = len(self.meshes) - 1
        self._mesh_cache[key] = mesh_index
        return mesh_index

    def add_node(self, name: str, *, translation: Optional[Sequence[float]] = None,
                 rotation: Optional[Sequence[float]] = None,
                 mesh: Optional[int] = None,
                 children: Optional[Sequence[int]] = None) -> int:
        """Add a node to the scene tree.

        Position and rotation are given separately rather than as one matrix so
        the browser can change the rotation of a joint node and let the
        renderer recompute the rest -- which is exactly how the robot is posed.

        Args:
            name (str): the node's name; the page looks nodes up by it.
            translation (Optional[Sequence[float]]): offset from the parent.
            rotation (Optional[Sequence[float]]): rotation as ``(x, y, z, w)``.
            mesh (Optional[int]): a mesh to draw at this node.
            children (Optional[Sequence[int]]): child node indices.

        Returns:
            int: the index of the new node.
        """
        node: Dict[str, Any] = {"name": name}
        if translation is not None:
            node["translation"] = [float(v) for v in translation]
        if rotation is not None:
            node["rotation"] = [float(v) for v in rotation]
        if mesh is not None:
            node["mesh"] = mesh
        if children:
            node["children"] = [int(v) for v in children]
        self.nodes.append(node)
        return len(self.nodes) - 1

    def set_children(self, node_index: int, children: Sequence[int]) -> None:
        """Give a node its children after the fact.

        Nodes have to be created before they can be referenced, so a tree is
        built leaves-first and the parents are joined up afterwards.

        Args:
            node_index (int): which node.
            children (Sequence[int]): its child node indices.
        """
        if children:
            self.nodes[node_index]["children"] = [int(v) for v in children]

    def write(self, path: str, roots: Sequence[int]) -> int:
        """Write the scene to a GLB file.

        Args:
            path (str): the file to write.
            roots (Sequence[int]): the node indices at the top of the tree.

        Returns:
            int: the file size in bytes.
        """
        self._align()
        gltf = {
            "asset": {"version": "2.0", "generator": "husky_assembly_tamp benchmark"},
            "scene": 0,
            "scenes": [{"nodes": [int(v) for v in roots]}],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "materials": self.materials,
            "accessors": self.accessors,
            "bufferViews": self.buffer_views,
            "buffers": [{"byteLength": len(self.buffer)}],
        }
        json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        # Both chunks must be four-byte aligned; JSON pads with spaces so it
        # stays valid text, the binary chunk pads with zeros.
        json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
        binary = bytes(self.buffer)
        binary += b"\0" * ((4 - len(binary) % 4) % 4)

        total = 12 + 8 + len(json_bytes) + 8 + len(binary)
        with open(path, "wb") as handle:
            handle.write(b"glTF")
            handle.write(struct.pack("<II", 2, total))
            handle.write(struct.pack("<I", len(json_bytes)))
            handle.write(b"JSON")
            handle.write(json_bytes)
            handle.write(struct.pack("<I", len(binary)))
            handle.write(b"BIN\0")
            handle.write(binary)
        return total


def mesh_arrays(mesh) -> Tuple[np.ndarray, np.ndarray]:
    """Turn a compas mesh into plain vertex and triangle arrays.

    Faces with more than three corners are fanned into triangles, since glTF
    draws triangles only.

    Args:
        mesh: a compas ``Mesh``.

    Returns:
        Tuple[np.ndarray, np.ndarray]: an ``(n, 3)`` float array of positions
        and an ``(m, 3)`` integer array of triangle indices.
    """
    vertex_index = {key: i for i, key in enumerate(mesh.vertices())}
    vertices = np.array(
        [mesh.vertex_coordinates(key) for key in mesh.vertices()], dtype=np.float32)
    triangles: List[List[int]] = []
    for face in mesh.faces():
        corners = [vertex_index[key] for key in mesh.face_vertices(face)]
        for offset in range(1, len(corners) - 1):
            triangles.append([corners[0], corners[offset], corners[offset + 1]])
    return vertices, np.array(triangles, dtype=np.uint32).reshape(-1, 3)


def frame_to_translation_rotation(frame) -> Tuple[List[float], List[float]]:
    """Split a compas frame into a position and a rotation.

    Args:
        frame: a compas ``Frame``.

    Returns:
        Tuple[List[float], List[float]]: the position, and the rotation as
        ``(x, y, z, w)`` -- the order glTF and three.js both use.
    """
    if frame is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
    translation = [float(v) for v in frame.point]
    rotation_matrix = np.array([
        [frame.xaxis[0], frame.yaxis[0], frame.zaxis[0]],
        [frame.xaxis[1], frame.yaxis[1], frame.zaxis[1]],
        [frame.xaxis[2], frame.yaxis[2], frame.zaxis[2]],
    ], dtype=float)
    return translation, matrix_to_quaternion(rotation_matrix)


def matrix_to_quaternion(matrix: np.ndarray) -> List[float]:
    """Convert a rotation matrix into a quaternion.

    Uses the branch of Shepperd's method with the largest denominator, which
    avoids the precision loss the naive formula suffers near half turns.

    Args:
        matrix (np.ndarray): a 3x3 rotation matrix.

    Returns:
        List[float]: the rotation as ``(x, y, z, w)``.
    """
    trace = float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2])
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    return [float(x), float(y), float(z), float(w)]
