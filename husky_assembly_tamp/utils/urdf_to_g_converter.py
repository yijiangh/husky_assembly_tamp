#!/usr/bin/env python3
"""
URDF to .g file converter for RAI (Robotic AI) library.
Based on analysis of panda.g and panda_clean.g file formats.

.g file format key points:
- Links are frames: `link_name: { }`
- Visual meshes: `link_name_0(link_name): { shape: mesh, mesh: <path>, visual: True }`
- Joint origins: `joint_origin(parent): { Q: [x, y, z, qw, qx, qy, qz] }`
- Joints: `joint_name(joint_origin): { joint: hingeZ/transX/rigid, limits: [...] }`
- Q (local transform): `Q: [x, y, z, qw, qx, qy, qz]` or `Q: [x, y, z]`

Outputs:
- Standard .g file with mesh references
- Standalone .g file with simple shapes (no external mesh dependencies)
"""

import xml.etree.ElementTree as ET
import numpy as np
import math
import os
import re


def rpy_to_quaternion(roll, pitch, yaw):
    """Convert roll-pitch-yaw angles (in radians) to quaternion [qw, qx, qy, qz]."""
    cr = math.cos(roll / 2)
    sr = math.sin(roll / 2)
    cp = math.cos(pitch / 2)
    sp = math.sin(pitch / 2)
    cy = math.cos(yaw / 2)
    sy = math.sin(yaw / 2)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return [qw, qx, qy, qz]


def parse_origin(origin_elem):
    """Parse URDF origin element to get xyz and rpy."""
    if origin_elem is None:
        return [0, 0, 0], [0, 0, 0]
    
    xyz_str = origin_elem.get('xyz', '0 0 0')
    rpy_str = origin_elem.get('rpy', '0 0 0')
    
    xyz = [float(x) for x in xyz_str.split()]
    rpy = [float(x) for x in rpy_str.split()]
    
    return xyz, rpy


def format_Q(xyz, quat=None):
    """Format the Q attribute for .g file.
    
    Q can be just translation [x, y, z] or full pose [x, y, z, qw, qx, qy, qz]
    """
    # Clean up near-zero values
    def clean(val):
        if abs(val) < 1e-10:
            return 0
        return round(val, 7)
    
    xyz = [clean(x) for x in xyz]
    
    # If no rotation or identity rotation, just use translation
    if quat is None:
        return f"[{xyz[0]}, {xyz[1]}, {xyz[2]}]"
    
    quat = [clean(q) for q in quat]
    
    # Check if rotation is identity (qw=1, qx=qy=qz=0)
    is_identity = abs(quat[0] - 1.0) < 1e-6 and all(abs(q) < 1e-6 for q in quat[1:])
    
    if is_identity:
        return f"[{xyz[0]}, {xyz[1]}, {xyz[2]}]"
    else:
        return f"[{xyz[0]}, {xyz[1]}, {xyz[2]}, {quat[0]}, {quat[1]}, {quat[2]}, {quat[3]}]"


def urdf_joint_type_to_g(urdf_type, axis):
    """Convert URDF joint type to .g joint type."""
    if urdf_type == 'fixed':
        return 'rigid'
    elif urdf_type in ['revolute', 'continuous']:
        # Determine axis
        if axis is None:
            return 'hingeZ'
        axis_xyz = [float(x) for x in axis.get('xyz', '0 0 1').split()]
        if abs(axis_xyz[0]) > 0.9:
            return 'hingeX'
        elif abs(axis_xyz[1]) > 0.9:
            return 'hingeY'
        else:
            return 'hingeZ'
    elif urdf_type == 'prismatic':
        if axis is None:
            return 'transZ'
        axis_xyz = [float(x) for x in axis.get('xyz', '0 0 1').split()]
        if abs(axis_xyz[0]) > 0.9:
            return 'transX'
        elif abs(axis_xyz[1]) > 0.9:
            return 'transY'
        else:
            return 'transZ'
    else:
        return 'rigid'


def format_limits(limit_elem):
    """Format joint limits for .g file."""
    if limit_elem is None:
        return None
    
    lower = float(limit_elem.get('lower', '-3.14159'))
    upper = float(limit_elem.get('upper', '3.14159'))
    velocity = float(limit_elem.get('velocity', '1.0'))
    effort = float(limit_elem.get('effort', '100.0'))
    
    # .g format: [lower, upper, velocity, ?, effort]
    return f"[{lower}, {upper}]"


def convert_mesh_path(urdf_mesh_path, absolute_path_base=None):
    """Convert URDF mesh path to .g format."""
    # Handle package:// URIs
    if urdf_mesh_path.startswith('package://'):
        # Remove package:// prefix
        path = urdf_mesh_path[len('package://'):]
        
        if absolute_path_base:
            # Convert to absolute path
            return f"<{os.path.join(absolute_path_base, path)}>"
        else:
            # Keep as relative path
            return f"<{path}>"
    else:
        return f"<{urdf_mesh_path}>"


# Predefined simple shape approximations for common robot parts
SIMPLE_SHAPE_CONFIGS = {
    # Husky base
    'base_link': {'shape': 'ssBox', 'size': [0.7, 0.5, 0.1, 0.02], 'color': [0.8, 0.8, 0.0, 1], 'Q': [0, 0, 0.05]},
    'wheel': {'shape': 'cylinder', 'size': [0.1143, 0.1651], 'color': [0.15, 0.15, 0.15, 1], 'Q': [0, 0, 0, 0.7071068, 0.7071068, 0, 0]},
    'top_plate': {'shape': 'ssBox', 'size': [0.6, 0.4, 0.01, 0.005], 'color': [0.2, 0.2, 0.2, 1]},
    'bumper': {'shape': 'ssBox', 'size': [0.1, 0.5, 0.05, 0.01], 'color': [0.15, 0.15, 0.15, 1]},
    'top_chassis': {'shape': 'ssBox', 'size': [0.5, 0.4, 0.05, 0.01], 'color': [0.2, 0.2, 0.2, 1]},
    
    # UR arm
    'ur_arm_base': {'shape': 'cylinder', 'size': [0.08, 0.05], 'color': [0.7, 0.7, 0.7, 1], 'Q': [0, 0, 0.04]},
    'ur_arm_shoulder': {'shape': 'cylinder', 'size': [0.1, 0.05], 'color': [0.7, 0.7, 0.7, 1]},
    'ur_arm_upper_arm': {'shape': 'capsule', 'size': [0.425, 0.05], 'color': [0.7, 0.7, 0.7, 1], 'Q': [-0.2125, 0, 0.13, 0.7071068, 0, 0.7071068, 0]},
    'ur_arm_forearm': {'shape': 'capsule', 'size': [0.392, 0.04], 'color': [0.7, 0.7, 0.7, 1], 'Q': [-0.196, 0, 0, 0.7071068, 0, 0.7071068, 0]},
    'ur_arm_wrist': {'shape': 'cylinder', 'size': [0.08, 0.04], 'color': [0.7, 0.7, 0.7, 1]},
    'ur_arm_wrist_3': {'shape': 'cylinder', 'size': [0.06, 0.03], 'color': [0.2, 0.2, 0.2, 1]},
    
    # Generic fallback
    'default': {'shape': 'sphere', 'size': [0.05], 'color': [0.5, 0.5, 0.5, 1]},
}


def get_simple_shape_for_link(link_name):
    """Get a simple shape configuration for a link based on its name."""
    link_lower = link_name.lower()
    
    # Match by patterns
    if 'wheel' in link_lower:
        return SIMPLE_SHAPE_CONFIGS['wheel']
    elif 'base_link' in link_lower and 'ur' not in link_lower:
        return SIMPLE_SHAPE_CONFIGS['base_link']
    elif 'top_plate' in link_lower:
        return SIMPLE_SHAPE_CONFIGS['top_plate']
    elif 'bumper' in link_lower:
        return SIMPLE_SHAPE_CONFIGS['bumper']
    elif 'chassis' in link_lower:
        return SIMPLE_SHAPE_CONFIGS['top_chassis']
    elif 'ur_arm_base' in link_lower or 'base_link_inertia' in link_lower:
        return SIMPLE_SHAPE_CONFIGS['ur_arm_base']
    elif 'shoulder' in link_lower:
        return SIMPLE_SHAPE_CONFIGS['ur_arm_shoulder']
    elif 'upper_arm' in link_lower:
        return SIMPLE_SHAPE_CONFIGS['ur_arm_upper_arm']
    elif 'forearm' in link_lower:
        return SIMPLE_SHAPE_CONFIGS['ur_arm_forearm']
    elif 'wrist_3' in link_lower:
        return SIMPLE_SHAPE_CONFIGS['ur_arm_wrist_3']
    elif 'wrist' in link_lower:
        return SIMPLE_SHAPE_CONFIGS['ur_arm_wrist']
    else:
        return None  # No visual for unknown links


class URDFToGConverter:
    def __init__(self, urdf_path):
        self.urdf_path = urdf_path
        self.tree = ET.parse(urdf_path)
        self.root = self.tree.getroot()
        self.robot_name = self.root.get('name', 'robot')
        
        # Parse all links and joints
        self.links = {}
        self.joints = {}
        self.parent_map = {}  # child_link -> (parent_link, joint)
        self.joint_order = []  # Preserve order of joints as they appear in URDF
        self.child_order_map = {}  # parent_link -> [ordered list of (child_link, joint_name)]
        
        self._parse_urdf()
    
    def _parse_urdf(self):
        """Parse URDF file and build internal representation."""
        # Parse links
        for link in self.root.findall('link'):
            link_name = link.get('name')
            self.links[link_name] = {
                'visual': link.find('visual'),
                'collision': link.find('collision'),
                'inertial': link.find('inertial')
            }
        
        # Parse joints (preserve URDF order)
        for joint in self.root.findall('joint'):
            joint_name = joint.get('name')
            joint_type = joint.get('type')
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')
            origin = joint.find('origin')
            axis = joint.find('axis')
            limit = joint.find('limit')
            
            self.joints[joint_name] = {
                'type': joint_type,
                'parent': parent,
                'child': child,
                'origin': origin,
                'axis': axis,
                'limit': limit
            }
            
            # Preserve joint order as they appear in URDF
            self.joint_order.append(joint_name)
            
            # Build parent_map and child_order_map
            self.parent_map[child] = (parent, joint_name)
            
            # Build ordered list of children for each parent (preserves URDF order)
            if parent not in self.child_order_map:
                self.child_order_map[parent] = []
            self.child_order_map[parent].append((child, joint_name))
    
    def _get_root_links(self):
        """Find root links (links with no parent)."""
        all_children = set(self.parent_map.keys())
        all_links = set(self.links.keys())
        return all_links - all_children
    
    def convert(self, output_path=None, base_frame_name=None, include_collisions=True,
                standalone=False, absolute_mesh_path_base=None):
        """Convert URDF to .g format.
        
        Args:
            output_path: Output file path
            base_frame_name: Name for the multibody base frame
            include_collisions: Include collision geometries
            standalone: Use simple shapes instead of meshes
            absolute_mesh_path_base: Base path for absolute mesh paths
        """
        lines = []
        
        if standalone:
            lines.append(f"## Converted from URDF: {os.path.basename(self.urdf_path)}")
            lines.append(f"## Robot name: {self.robot_name}")
            lines.append(f"## STANDALONE VERSION - Using simple shapes instead of meshes")
        else:
            lines.append(f"## Converted from URDF: {os.path.basename(self.urdf_path)}")
            lines.append(f"## Robot name: {self.robot_name}")
        lines.append("")
        
        # Create base frame if specified
        if base_frame_name:
            lines.append(f"{base_frame_name}: {{ multibody: true }}")
            lines.append("")
        
        # Process links in order (BFS from root, preserving URDF order for siblings)
        root_links = self._get_root_links()
        processed = set()
        queue = list(root_links)
        
        while queue:
            link_name = queue.pop(0)
            if link_name in processed:
                continue
            processed.add(link_name)
            
            # Find children in URDF order (preserves left/right ordering)
            if link_name in self.child_order_map:
                for child_link, joint_name in self.child_order_map[link_name]:
                    if child_link not in processed:
                        queue.append(child_link)
            
            # Generate .g content for this link
            link_lines = self._convert_link(
                link_name, base_frame_name, include_collisions,
                standalone=standalone, absolute_mesh_path_base=absolute_mesh_path_base
            )
            lines.extend(link_lines)
        
        # Process joints in URDF order (preserves left/right ordering)
        lines.append("")
        lines.append("## Joints")
        for joint_name in self.joint_order:
            joint_data = self.joints[joint_name]
            joint_lines = self._convert_joint(joint_name, joint_data)
            lines.extend(joint_lines)
        
        # Add gripper marker and initial joint config for standalone version
        if standalone and base_frame_name:
            lines.append("")
            lines.append("## End effector / Gripper definition")
            lines.append("gripper(ur_arm_tool0): { Q: [0, 0, 0.15], shape: marker, size: [0.03], color: [0.9, 0.9, 0.9, 1], logical: { is_gripper: true } }")
            lines.append("")
            lines.append("## Initial joint configuration")
            lines.append("Edit ur_arm_shoulder_pan_joint: { q: 0.0 }")
            lines.append("Edit ur_arm_shoulder_lift_joint: { q: -1.5708 }")
            lines.append("Edit ur_arm_elbow_joint: { q: 1.5708 }")
            lines.append("Edit ur_arm_wrist_1_joint: { q: -1.5708 }")
            lines.append("Edit ur_arm_wrist_2_joint: { q: -1.5708 }")
            lines.append("Edit ur_arm_wrist_3_joint: { q: 0.0 }")
        
        result = '\n'.join(lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(result)
            print(f"Written to {output_path}")
        
        return result
    
    def convert_both(self, output_path, base_frame_name=None, include_collisions=True,
                     absolute_mesh_path_base=None):
        """Convert URDF to both standard and standalone .g formats.
        
        Args:
            output_path: Base output file path (will generate .g and _standalone.g)
            base_frame_name: Name for the multibody base frame
            include_collisions: Include collision geometries
            absolute_mesh_path_base: Base path for absolute mesh paths
            
        Returns:
            Tuple of (standard_path, standalone_path)
        """
        # Generate standard version
        standard_path = output_path
        if not standard_path.endswith('.g'):
            standard_path = standard_path.replace('.urdf', '.g')
        
        self.convert(
            output_path=standard_path,
            base_frame_name=base_frame_name,
            include_collisions=include_collisions,
            standalone=False,
            absolute_mesh_path_base=absolute_mesh_path_base
        )
        
        # Generate standalone version
        standalone_path = standard_path.replace('.g', '_standalone.g')
        self.convert(
            output_path=standalone_path,
            base_frame_name=base_frame_name,
            include_collisions=include_collisions,
            standalone=True,
            absolute_mesh_path_base=None  # Standalone doesn't use meshes
        )
        
        return standard_path, standalone_path
    
    def _convert_link(self, link_name, base_frame_name, include_collisions,
                      standalone=False, absolute_mesh_path_base=None):
        """Convert a single link to .g format."""
        lines = []
        link_data = self.links[link_name]
        
        # Determine parent frame
        if link_name in self.parent_map:
            parent_link, joint_name = self.parent_map[link_name]
            # Link is parented to the joint
            parent_frame = joint_name
        elif base_frame_name:
            # Root link - parent to base frame
            parent_frame = base_frame_name
        else:
            parent_frame = None
        
        # Create link frame
        if parent_frame:
            lines.append(f"{link_name}({parent_frame}): {{  }}")
        else:
            lines.append(f"{link_name}: {{  }}")
        
        if standalone:
            # Use simple shapes instead of meshes
            self._add_simple_visual(lines, link_name)
            if include_collisions:
                self._add_simple_collision(lines, link_name)
        else:
            # Use mesh geometries
            self._add_mesh_visual(lines, link_name, link_data, absolute_mesh_path_base)
            if include_collisions:
                self._add_mesh_collision(lines, link_name, link_data, absolute_mesh_path_base)
        
        return lines
    
    def _add_simple_visual(self, lines, link_name):
        """Add simple shape visual for standalone version."""
        shape_config = get_simple_shape_for_link(link_name)
        if shape_config is None:
            return
        
        props = []
        
        # Add Q transform if specified
        if 'Q' in shape_config:
            q = shape_config['Q']
            if len(q) == 3:
                props.append(f"Q: [{q[0]}, {q[1]}, {q[2]}]")
            else:
                props.append(f"Q: [{q[0]}, {q[1]}, {q[2]}, {q[3]}, {q[4]}, {q[5]}, {q[6]}]")
        
        # Add shape
        props.append(f"shape: {shape_config['shape']}")
        
        # Add size
        size = shape_config['size']
        props.append(f"size: [{', '.join(str(s) for s in size)}]")
        
        # Add color
        color = shape_config['color']
        props.append(f"color: [{color[0]}, {color[1]}, {color[2]}, {color[3]}]")
        
        lines.append(f"{link_name}_visual({link_name}): {{ {', '.join(props)} }}")
    
    def _add_simple_collision(self, lines, link_name):
        """Add simple shape collision for standalone version."""
        shape_config = get_simple_shape_for_link(link_name)
        if shape_config is None:
            return
        
        props = []
        
        # Add Q transform if specified
        if 'Q' in shape_config:
            q = shape_config['Q']
            if len(q) == 3:
                props.append(f"Q: [{q[0]}, {q[1]}, {q[2]}]")
            else:
                props.append(f"Q: [{q[0]}, {q[1]}, {q[2]}, {q[3]}, {q[4]}, {q[5]}, {q[6]}]")
        
        # Add shape
        props.append(f"shape: {shape_config['shape']}")
        
        # Add size
        size = shape_config['size']
        props.append(f"size: [{', '.join(str(s) for s in size)}]")
        
        # Semi-transparent for collision
        props.append("color: [1, 1, 1, 0.1]")
        props.append("contact: -2")
        
        lines.append(f"{link_name}_coll({link_name}): {{ {', '.join(props)} }}")
    
    def _add_mesh_visual(self, lines, link_name, link_data, absolute_mesh_path_base):
        """Add mesh visual geometry."""
        visual = link_data['visual']
        if visual is None:
            return
        
        geom = visual.find('geometry')
        if geom is None:
            return
        
        mesh = geom.find('mesh')
        if mesh is None:
            return
        
        mesh_path = convert_mesh_path(mesh.get('filename'), absolute_mesh_path_base)
        origin = visual.find('origin')
        xyz, rpy = parse_origin(origin)
        
        props = [f"shape: mesh", f"mesh: {mesh_path}", "visual: true"]
        
        # Add origin transform if not identity
        if any(abs(x) > 1e-10 for x in xyz) or any(abs(r) > 1e-10 for r in rpy):
            quat = rpy_to_quaternion(rpy[0], rpy[1], rpy[2])
            q_str = format_Q(xyz, quat)
            props.insert(0, f"Q: {q_str}")
        
        lines.append(f"{link_name}_visual({link_name}): {{ {', '.join(props)} }}")
    
    def _add_mesh_collision(self, lines, link_name, link_data, absolute_mesh_path_base):
        """Add mesh collision geometry."""
        collision = link_data['collision']
        if collision is None:
            return
        
        geom = collision.find('geometry')
        if geom is None:
            return
        
        origin = collision.find('origin')
        xyz, rpy = parse_origin(origin)
        
        props = []
        
        # Add origin transform if not identity
        if any(abs(x) > 1e-10 for x in xyz) or any(abs(r) > 1e-10 for r in rpy):
            quat = rpy_to_quaternion(rpy[0], rpy[1], rpy[2])
            q_str = format_Q(xyz, quat)
            props.append(f"Q: {q_str}")
        
        mesh = geom.find('mesh')
        box = geom.find('box')
        cylinder = geom.find('cylinder')
        sphere = geom.find('sphere')
        
        if mesh is not None:
            mesh_path = convert_mesh_path(mesh.get('filename'), absolute_mesh_path_base)
            props.extend([f"shape: mesh", f"mesh: {mesh_path}"])
        elif box is not None:
            size = [float(x) for x in box.get('size').split()]
            props.extend([f"shape: ssBox", f"size: [{size[0]}, {size[1]}, {size[2]}, 0.001]"])
        elif cylinder is not None:
            radius = float(cylinder.get('radius'))
            length = float(cylinder.get('length'))
            props.extend([f"shape: capsule", f"size: [{length}, {radius}]"])
        elif sphere is not None:
            radius = float(sphere.get('radius'))
            props.extend([f"shape: sphere", f"size: [{radius}]"])
        
        if len(props) > 0:
            props.append("contact: -2")
            lines.append(f"{link_name}_coll({link_name}): {{ {', '.join(props)} }}")
    
    def _convert_joint(self, joint_name, joint_data):
        """Convert a single joint to .g format."""
        lines = []
        
        parent_link = joint_data['parent']
        child_link = joint_data['child']
        joint_type = joint_data['type']
        origin = joint_data['origin']
        axis = joint_data['axis']
        limit = joint_data['limit']
        
        # Parse origin
        xyz, rpy = parse_origin(origin)
        quat = rpy_to_quaternion(rpy[0], rpy[1], rpy[2])
        q_str = format_Q(xyz, quat)
        
        # Get .g joint type
        g_joint_type = urdf_joint_type_to_g(joint_type, axis)
        
        # Create joint origin frame (relative to parent link)
        lines.append(f"{joint_name}_origin({parent_link}): {{ Q: {q_str} }}")
        
        # Create joint
        joint_props = [f"joint: {g_joint_type}"]
        
        # Add limits for non-fixed joints
        if joint_type in ['revolute', 'prismatic'] and limit is not None:
            limits = format_limits(limit)
            if limits:
                joint_props.append(f"limits: {limits}")
        
        lines.append(f"{joint_name}({joint_name}_origin): {{ {', '.join(joint_props)} }}")
        
        return lines


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert URDF to .g format for RAI library',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion (standard .g file only)
  python urdf_to_g_converter.py robot.urdf
  
  # Generate both standard and standalone versions
  python urdf_to_g_converter.py robot.urdf --both
  
  # With base frame and absolute mesh paths
  python urdf_to_g_converter.py robot.urdf -b robot_base --mesh-base /path/to/meshes --both
  
  # Standalone version only
  python urdf_to_g_converter.py robot.urdf --standalone -o robot_standalone.g
"""
    )
    parser.add_argument('urdf_file', help='Input URDF file path')
    parser.add_argument('-o', '--output', help='Output .g file path')
    parser.add_argument('-b', '--base-frame', help='Base frame name for multibody', default=None)
    parser.add_argument('--no-collision', action='store_true', help='Skip collision geometries')
    parser.add_argument('--standalone', action='store_true', 
                        help='Generate standalone version with simple shapes instead of meshes')
    parser.add_argument('--both', action='store_true',
                        help='Generate both standard and standalone versions')
    parser.add_argument('--mesh-base', help='Base path for absolute mesh paths', default=None)
    
    args = parser.parse_args()
    
    converter = URDFToGConverter(args.urdf_file)
    
    output_path = args.output
    if output_path is None:
        output_path = args.urdf_file.replace('.urdf', '.g')
    
    if args.both:
        # Generate both versions
        standard_path, standalone_path = converter.convert_both(
            output_path=output_path,
            base_frame_name=args.base_frame,
            include_collisions=not args.no_collision,
            absolute_mesh_path_base=args.mesh_base
        )
        print(f"Generated:")
        print(f"  Standard:   {standard_path}")
        print(f"  Standalone: {standalone_path}")
    else:
        # Generate single version
        converter.convert(
            output_path=output_path,
            base_frame_name=args.base_frame,
            include_collisions=not args.no_collision,
            standalone=args.standalone,
            absolute_mesh_path_base=args.mesh_base
        )


if __name__ == '__main__':
    main()
