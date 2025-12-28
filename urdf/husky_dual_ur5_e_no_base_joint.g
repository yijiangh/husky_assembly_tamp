## Converted from URDF: husky_dual_ur5_e_no_base_joint.urdf
## Robot name: husky

world_link: {  }
base_footprint(world_footprint): {  }
base_link(base_footprint_joint): {  }
base_link_visual(base_link): { shape: mesh, mesh: <husky_description/meshes/base_link.stl>, visual: true }
base_link_coll(base_link): { Q: [0, 0, 0.061875], shape: ssBox, size: [0.9874, 0.5709, 0.12375, 0.001], contact: -2 }
inertial_link(inertial_joint): {  }
imu_link(imu_joint): {  }
front_left_wheel_link(front_left_wheel): {  }
front_left_wheel_link_visual(front_left_wheel_link): { shape: mesh, mesh: <husky_description/meshes/wheel.obj>, visual: true }
front_left_wheel_link_coll(front_left_wheel_link): { Q: [0, 0, 0, 0.7071073, 0.7071063, 0, 0], shape: capsule, size: [0.1143, 0.1651], contact: -2 }
front_right_wheel_link(front_right_wheel): {  }
front_right_wheel_link_visual(front_right_wheel_link): { shape: mesh, mesh: <husky_description/meshes/wheel.obj>, visual: true }
front_right_wheel_link_coll(front_right_wheel_link): { Q: [0, 0, 0, 0.7071073, 0.7071063, 0, 0], shape: capsule, size: [0.1143, 0.1651], contact: -2 }
rear_left_wheel_link(rear_left_wheel): {  }
rear_left_wheel_link_visual(rear_left_wheel_link): { shape: mesh, mesh: <husky_description/meshes/wheel.obj>, visual: true }
rear_left_wheel_link_coll(rear_left_wheel_link): { Q: [0, 0, 0, 0.7071073, 0.7071063, 0, 0], shape: capsule, size: [0.1143, 0.1651], contact: -2 }
rear_right_wheel_link(rear_right_wheel): {  }
rear_right_wheel_link_visual(rear_right_wheel_link): { shape: mesh, mesh: <husky_description/meshes/wheel.obj>, visual: true }
rear_right_wheel_link_coll(rear_right_wheel_link): { Q: [0, 0, 0, 0.7071073, 0.7071063, 0, 0], shape: capsule, size: [0.1143, 0.1651], contact: -2 }
top_chassis_link(top_chassis_joint): {  }
top_chassis_link_visual(top_chassis_link): { shape: mesh, mesh: <husky_description/meshes/top_chassis.obj>, visual: true }
front_bumper_link(front_bumper): {  }
front_bumper_link_visual(front_bumper_link): { shape: mesh, mesh: <husky_description/meshes/bumper.obj>, visual: true }
rear_bumper_link(rear_bumper): {  }
rear_bumper_link_visual(rear_bumper_link): { shape: mesh, mesh: <husky_description/meshes/bumper.obj>, visual: true }
dual_arm_bulkhead_link(dual_arm_bulkhead_joint): {  }
dual_arm_bulkhead_link_visual(dual_arm_bulkhead_link): { shape: mesh, mesh: <husky_ur_description/meshes/dual_arm_bulkhead.obj>, visual: true }
dual_arm_bulkhead_link_coll(dual_arm_bulkhead_link): { shape: mesh, mesh: <husky_ur_description/meshes/dual_arm_bulkhead_collision.stl>, contact: -2 }
left_arm_bulkhead_link(left_arm_bulkhead_joint): {  }
right_arm_bulkhead_link(right_arm_bulkhead_joint): {  }
left_ur_arm_base_link(left_arm_mount_joint): {  }
right_ur_arm_base_link(right_arm_mount_joint): {  }
left_ur_arm_base_link_inertia(left_ur_arm_base_link-base_link_inertia): {  }
left_ur_arm_base_link_inertia_visual(left_ur_arm_base_link_inertia): { Q: [0, 0, 0, 0, 0, 0, 1.0], shape: mesh, mesh: <ur_description/meshes/ur5e/visual/base.obj>, visual: true }
left_ur_arm_base_link_inertia_coll(left_ur_arm_base_link_inertia): { Q: [0, 0, 0, 0, 0, 0, 1.0], shape: mesh, mesh: <ur_description/meshes/ur5e/collision/base.stl>, contact: -2 }
left_ur_arm_base(left_ur_arm_base_link-base_fixed_joint): {  }
right_ur_arm_base_link_inertia(right_ur_arm_base_link-base_link_inertia): {  }
right_ur_arm_base_link_inertia_visual(right_ur_arm_base_link_inertia): { Q: [0, 0, 0, 0, 0, 0, 1.0], shape: mesh, mesh: <ur_description/meshes/ur5e/visual/base.obj>, visual: true }
right_ur_arm_base_link_inertia_coll(right_ur_arm_base_link_inertia): { Q: [0, 0, 0, 0, 0, 0, 1.0], shape: mesh, mesh: <ur_description/meshes/ur5e/collision/base.stl>, contact: -2 }
right_ur_arm_base(right_ur_arm_base_link-base_fixed_joint): {  }
left_ur_arm_shoulder_link(left_ur_arm_shoulder_pan_joint): {  }
left_ur_arm_shoulder_link_visual(left_ur_arm_shoulder_link): { Q: [0, 0, 0, 0, 0, 0, 1.0], shape: mesh, mesh: <ur_description/meshes/ur5e/visual/shoulder.obj>, visual: true }
left_ur_arm_shoulder_link_coll(left_ur_arm_shoulder_link): { Q: [0, 0, 0, 0, 0, 0, 1.0], shape: mesh, mesh: <ur_description/meshes/ur5e/collision/shoulder.stl>, contact: -2 }
right_ur_arm_shoulder_link(right_ur_arm_shoulder_pan_joint): {  }
right_ur_arm_shoulder_link_visual(right_ur_arm_shoulder_link): { Q: [0, 0, 0, 0, 0, 0, 1.0], shape: mesh, mesh: <ur_description/meshes/ur5e/visual/shoulder.obj>, visual: true }
right_ur_arm_shoulder_link_coll(right_ur_arm_shoulder_link): { Q: [0, 0, 0, 0, 0, 0, 1.0], shape: mesh, mesh: <ur_description/meshes/ur5e/collision/shoulder.stl>, contact: -2 }
left_ur_arm_upper_arm_link(left_ur_arm_shoulder_lift_joint): {  }
left_ur_arm_upper_arm_link_visual(left_ur_arm_upper_arm_link): { Q: [0, 0, 0.138, 0.5, 0.5, -0.5, -0.5], shape: mesh, mesh: <ur_description/meshes/ur5e/visual/upperarm.obj>, visual: true }
left_ur_arm_upper_arm_link_coll(left_ur_arm_upper_arm_link): { Q: [0, 0, 0.138, 0.5, 0.5, -0.5, -0.5], shape: mesh, mesh: <ur_description/meshes/ur5e/collision/upperarm.stl>, contact: -2 }
right_ur_arm_upper_arm_link(right_ur_arm_shoulder_lift_joint): {  }
right_ur_arm_upper_arm_link_visual(right_ur_arm_upper_arm_link): { Q: [0, 0, 0.138, 0.5, 0.5, -0.5, -0.5], shape: mesh, mesh: <ur_description/meshes/ur5e/visual/upperarm.obj>, visual: true }
right_ur_arm_upper_arm_link_coll(right_ur_arm_upper_arm_link): { Q: [0, 0, 0.138, 0.5, 0.5, -0.5, -0.5], shape: mesh, mesh: <ur_description/meshes/ur5e/collision/upperarm.stl>, contact: -2 }
left_ur_arm_forearm_link(left_ur_arm_elbow_joint): {  }
left_ur_arm_forearm_link_visual(left_ur_arm_forearm_link): { Q: [0, 0, 0.007, 0.5, 0.5, -0.5, -0.5], shape: mesh, mesh: <ur_description/meshes/ur5e/visual/forearm.obj>, visual: true }
left_ur_arm_forearm_link_coll(left_ur_arm_forearm_link): { Q: [0, 0, 0.007, 0.5, 0.5, -0.5, -0.5], shape: mesh, mesh: <ur_description/meshes/ur5e/collision/forearm.stl>, contact: -2 }
right_ur_arm_forearm_link(right_ur_arm_elbow_joint): {  }
right_ur_arm_forearm_link_visual(right_ur_arm_forearm_link): { Q: [0, 0, 0.007, 0.5, 0.5, -0.5, -0.5], shape: mesh, mesh: <ur_description/meshes/ur5e/visual/forearm.obj>, visual: true }
right_ur_arm_forearm_link_coll(right_ur_arm_forearm_link): { Q: [0, 0, 0.007, 0.5, 0.5, -0.5, -0.5], shape: mesh, mesh: <ur_description/meshes/ur5e/collision/forearm.stl>, contact: -2 }
left_ur_arm_wrist_1_link(left_ur_arm_wrist_1_joint): {  }
left_ur_arm_wrist_1_link_visual(left_ur_arm_wrist_1_link): { Q: [0, 0, -0.127, 0.7071068, 0.7071068, 0, 0], shape: mesh, mesh: <ur_description/meshes/ur5e/visual/wrist1.obj>, visual: true }
left_ur_arm_wrist_1_link_coll(left_ur_arm_wrist_1_link): { Q: [0, 0, -0.127, 0.7071068, 0.7071068, 0, 0], shape: mesh, mesh: <ur_description/meshes/ur5e/collision/wrist1.stl>, contact: -2 }
right_ur_arm_wrist_1_link(right_ur_arm_wrist_1_joint): {  }
right_ur_arm_wrist_1_link_visual(right_ur_arm_wrist_1_link): { Q: [0, 0, -0.127, 0.7071068, 0.7071068, 0, 0], shape: mesh, mesh: <ur_description/meshes/ur5e/visual/wrist1.obj>, visual: true }
right_ur_arm_wrist_1_link_coll(right_ur_arm_wrist_1_link): { Q: [0, 0, -0.127, 0.7071068, 0.7071068, 0, 0], shape: mesh, mesh: <ur_description/meshes/ur5e/collision/wrist1.stl>, contact: -2 }
left_ur_arm_wrist_2_link(left_ur_arm_wrist_2_joint): {  }
left_ur_arm_wrist_2_link_visual(left_ur_arm_wrist_2_link): { Q: [0, 0, -0.0997], shape: mesh, mesh: <ur_description/meshes/ur5e/visual/wrist2.obj>, visual: true }
left_ur_arm_wrist_2_link_coll(left_ur_arm_wrist_2_link): { Q: [0, 0, -0.0997], shape: mesh, mesh: <ur_description/meshes/ur5e/collision/wrist2.stl>, contact: -2 }
right_ur_arm_wrist_2_link(right_ur_arm_wrist_2_joint): {  }
right_ur_arm_wrist_2_link_visual(right_ur_arm_wrist_2_link): { Q: [0, 0, -0.0997], shape: mesh, mesh: <ur_description/meshes/ur5e/visual/wrist2.obj>, visual: true }
right_ur_arm_wrist_2_link_coll(right_ur_arm_wrist_2_link): { Q: [0, 0, -0.0997], shape: mesh, mesh: <ur_description/meshes/ur5e/collision/wrist2.stl>, contact: -2 }
left_ur_arm_wrist_3_link(left_ur_arm_wrist_3_joint): {  }
left_ur_arm_wrist_3_link_visual(left_ur_arm_wrist_3_link): { Q: [0, 0, -0.0989, 0.7071068, 0.7071068, 0, 0], shape: mesh, mesh: <ur_description/meshes/ur5e/visual/wrist3.obj>, visual: true }
left_ur_arm_wrist_3_link_coll(left_ur_arm_wrist_3_link): { Q: [0, 0, -0.0989, 0.7071068, 0.7071068, 0, 0], shape: mesh, mesh: <ur_description/meshes/ur5e/collision/wrist3.stl>, contact: -2 }
right_ur_arm_wrist_3_link(right_ur_arm_wrist_3_joint): {  }
right_ur_arm_wrist_3_link_visual(right_ur_arm_wrist_3_link): { Q: [0, 0, -0.0989, 0.7071068, 0.7071068, 0, 0], shape: mesh, mesh: <ur_description/meshes/ur5e/visual/wrist3.obj>, visual: true }
right_ur_arm_wrist_3_link_coll(right_ur_arm_wrist_3_link): { Q: [0, 0, -0.0989, 0.7071068, 0.7071068, 0, 0], shape: mesh, mesh: <ur_description/meshes/ur5e/collision/wrist3.stl>, contact: -2 }
left_ur_arm_flange(left_ur_arm_wrist_3-flange): {  }
right_ur_arm_flange(right_ur_arm_wrist_3-flange): {  }
left_ur_arm_tool0(left_ur_arm_flange-tool0): {  }
right_ur_arm_tool0(right_ur_arm_flange-tool0): {  }

## Joints
world_footprint_origin(world_link): { Q: [0, 0, 0] }
world_footprint(world_footprint_origin): { joint: rigid }
base_footprint_joint_origin(base_footprint): { Q: [0, 0, 0.13228] }
base_footprint_joint(base_footprint_joint_origin): { joint: rigid }
inertial_joint_origin(base_link): { Q: [0, 0, 0] }
inertial_joint(inertial_joint_origin): { joint: rigid }
imu_joint_origin(base_link): { Q: [0.19, 0, 0.149, -2.6e-06, 0.7071081, 2.6e-06, 0.7071055] }
imu_joint(imu_joint_origin): { joint: rigid }
front_left_wheel_origin(base_link): { Q: [0.256, 0.2854, 0.03282] }
front_left_wheel(front_left_wheel_origin): { joint: rigid }
front_right_wheel_origin(base_link): { Q: [0.256, -0.2854, 0.03282] }
front_right_wheel(front_right_wheel_origin): { joint: rigid }
rear_left_wheel_origin(base_link): { Q: [-0.256, 0.2854, 0.03282] }
rear_left_wheel(rear_left_wheel_origin): { joint: rigid }
rear_right_wheel_origin(base_link): { Q: [-0.256, -0.2854, 0.03282] }
rear_right_wheel(rear_right_wheel_origin): { joint: rigid }
top_chassis_joint_origin(base_link): { Q: [0, 0, 0] }
top_chassis_joint(top_chassis_joint_origin): { joint: rigid }
front_bumper_origin(base_link): { Q: [0.48, 0, 0.091] }
front_bumper(front_bumper_origin): { joint: rigid }
rear_bumper_origin(base_link): { Q: [-0.48, 0, 0.091, 1.3e-06, 0, 0, 1.0] }
rear_bumper(rear_bumper_origin): { joint: rigid }
dual_arm_bulkhead_joint_origin(base_link): { Q: [0, 0, 0.224] }
dual_arm_bulkhead_joint(dual_arm_bulkhead_joint_origin): { joint: rigid }
left_arm_bulkhead_joint_origin(dual_arm_bulkhead_link): { Q: [0.1225, 0.14891, 0.13371, 0.6532815, -0.2705981, -0.2705981, -0.6532815] }
left_arm_bulkhead_joint(left_arm_bulkhead_joint_origin): { joint: rigid }
right_arm_bulkhead_joint_origin(dual_arm_bulkhead_link): { Q: [0.1225, -0.14891, 0.13371, 0.6532815, 0.2705981, 0.2705981, -0.6532815] }
right_arm_bulkhead_joint(right_arm_bulkhead_joint_origin): { joint: rigid }
left_arm_mount_joint_origin(left_arm_bulkhead_link): { Q: [0, 0, 0] }
left_arm_mount_joint(left_arm_mount_joint_origin): { joint: rigid }
right_arm_mount_joint_origin(right_arm_bulkhead_link): { Q: [0, 0, 0] }
right_arm_mount_joint(right_arm_mount_joint_origin): { joint: rigid }
right_ur_arm_base_link-base_link_inertia_origin(right_ur_arm_base_link): { Q: [0, 0, 0, 0, 0, 0, 1.0] }
right_ur_arm_base_link-base_link_inertia(right_ur_arm_base_link-base_link_inertia_origin): { joint: rigid }
right_ur_arm_shoulder_pan_joint_origin(right_ur_arm_base_link_inertia): { Q: [0, 0, 0.1625] }
right_ur_arm_shoulder_pan_joint(right_ur_arm_shoulder_pan_joint_origin): { joint: hingeZ, limits: [-3.14159265359, 3.14159265359] }
right_ur_arm_shoulder_lift_joint_origin(right_ur_arm_shoulder_link): { Q: [0, 0, 0, 0.7071068, 0.7071068, 0, 0] }
right_ur_arm_shoulder_lift_joint(right_ur_arm_shoulder_lift_joint_origin): { joint: hingeZ, limits: [-3.14159265359, 0.3] }
right_ur_arm_elbow_joint_origin(right_ur_arm_upper_arm_link): { Q: [-0.425, 0, 0] }
right_ur_arm_elbow_joint(right_ur_arm_elbow_joint_origin): { joint: hingeZ, limits: [-3.14159265359, 3.14159265359] }
right_ur_arm_wrist_1_joint_origin(right_ur_arm_forearm_link): { Q: [-0.3922, 0, 0.1333] }
right_ur_arm_wrist_1_joint(right_ur_arm_wrist_1_joint_origin): { joint: hingeZ, limits: [-4.18879020479, 4.18879020479] }
right_ur_arm_wrist_2_joint_origin(right_ur_arm_wrist_1_link): { Q: [0, -0.0997, 0, 0.7071068, 0.7071068, 0, 0] }
right_ur_arm_wrist_2_joint(right_ur_arm_wrist_2_joint_origin): { joint: hingeZ, limits: [-4.18879020479, 4.18879020479] }
right_ur_arm_wrist_3_joint_origin(right_ur_arm_wrist_2_link): { Q: [0, 0.0996, 0, 0.7071068, -0.7071068, 0, 0] }
right_ur_arm_wrist_3_joint(right_ur_arm_wrist_3_joint_origin): { joint: hingeZ, limits: [-4.18879020479, 4.18879020479] }
right_ur_arm_base_link-base_fixed_joint_origin(right_ur_arm_base_link): { Q: [0, 0, 0, 0.7071068, 0, 0, -0.7071068] }
right_ur_arm_base_link-base_fixed_joint(right_ur_arm_base_link-base_fixed_joint_origin): { joint: rigid }
right_ur_arm_wrist_3-flange_origin(right_ur_arm_wrist_3_link): { Q: [0, 0, 0, 0.5, -0.5, -0.5, -0.5] }
right_ur_arm_wrist_3-flange(right_ur_arm_wrist_3-flange_origin): { joint: rigid }
right_ur_arm_flange-tool0_origin(right_ur_arm_flange): { Q: [0, 0, 0, 0.5, 0.5, 0.5, 0.5] }
right_ur_arm_flange-tool0(right_ur_arm_flange-tool0_origin): { joint: rigid }
left_ur_arm_base_link-base_link_inertia_origin(left_ur_arm_base_link): { Q: [0, 0, 0, 0, 0, 0, 1.0] }
left_ur_arm_base_link-base_link_inertia(left_ur_arm_base_link-base_link_inertia_origin): { joint: rigid }
left_ur_arm_shoulder_pan_joint_origin(left_ur_arm_base_link_inertia): { Q: [0, 0, 0.1625] }
left_ur_arm_shoulder_pan_joint(left_ur_arm_shoulder_pan_joint_origin): { joint: hingeZ, limits: [-3.14159265359, 3.14159265359] }
left_ur_arm_shoulder_lift_joint_origin(left_ur_arm_shoulder_link): { Q: [0, 0, 0, 0.7071068, 0.7071068, 0, 0] }
left_ur_arm_shoulder_lift_joint(left_ur_arm_shoulder_lift_joint_origin): { joint: hingeZ, limits: [-3.14159265359, 0.3] }
left_ur_arm_elbow_joint_origin(left_ur_arm_upper_arm_link): { Q: [-0.425, 0, 0] }
left_ur_arm_elbow_joint(left_ur_arm_elbow_joint_origin): { joint: hingeZ, limits: [-3.14159265359, 3.14159265359] }
left_ur_arm_wrist_1_joint_origin(left_ur_arm_forearm_link): { Q: [-0.3922, 0, 0.1333] }
left_ur_arm_wrist_1_joint(left_ur_arm_wrist_1_joint_origin): { joint: hingeZ, limits: [-4.18879020479, 4.18879020479] }
left_ur_arm_wrist_2_joint_origin(left_ur_arm_wrist_1_link): { Q: [0, -0.0997, 0, 0.7071068, 0.7071068, 0, 0] }
left_ur_arm_wrist_2_joint(left_ur_arm_wrist_2_joint_origin): { joint: hingeZ, limits: [-4.18879020479, 4.18879020479] }
left_ur_arm_wrist_3_joint_origin(left_ur_arm_wrist_2_link): { Q: [0, 0.0996, 0, 0.7071068, -0.7071068, 0, 0] }
left_ur_arm_wrist_3_joint(left_ur_arm_wrist_3_joint_origin): { joint: hingeZ, limits: [-4.18879020479, 4.18879020479] }
left_ur_arm_base_link-base_fixed_joint_origin(left_ur_arm_base_link): { Q: [0, 0, 0, 0, 0, 0, 1.0] }
left_ur_arm_base_link-base_fixed_joint(left_ur_arm_base_link-base_fixed_joint_origin): { joint: rigid }
left_ur_arm_wrist_3-flange_origin(left_ur_arm_wrist_3_link): { Q: [0, 0, 0, 0.5, -0.5, -0.5, -0.5] }
left_ur_arm_wrist_3-flange(left_ur_arm_wrist_3-flange_origin): { joint: rigid }
left_ur_arm_flange-tool0_origin(left_ur_arm_flange): { Q: [0, 0, 0, 0.5, 0.5, 0.5, 0.5] }
left_ur_arm_flange-tool0(left_ur_arm_flange-tool0_origin): { joint: rigid }