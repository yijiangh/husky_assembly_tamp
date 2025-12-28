## Converted from URDF: husky_ur5_e_no_base_joint.urdf
## Robot name: husky

world_link: {  }
base_footprint(world_footprint): {  }
base_link(base_footprint_joint): {  }
base_link_visual(base_link): { shape: mesh, mesh: <husky_description/meshes/base_link.stl>, visual: true }
base_link_coll(base_link): { Q: [0, 0, 0.175625], shape: ssBox, size: [0.78992, 0.5709, 0.10375, 0.001], contact: -2 }
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
top_plate_link(top_plate_joint): {  }
top_plate_link_visual(top_plate_link): { shape: mesh, mesh: <husky_description/meshes/large_top_plate.obj>, visual: true }
top_plate_link_coll(top_plate_link): { shape: mesh, mesh: <husky_description/meshes/large_top_plate_collision.stl>, contact: -2 }
ipad_rack_link(ipad_rack_joint): {  }
ipad_rack_link_visual(ipad_rack_link): { shape: mesh, mesh: <husky_description/meshes/ipad_rack_visual.obj>, visual: true }
ipad_rack_link_coll(ipad_rack_link): { shape: mesh, mesh: <husky_description/meshes/ipad_rack_collision.obj>, contact: -2 }
top_plate_front_link(top_plate_front_joint): {  }
top_plate_rear_link(top_plate_rear_joint): {  }
ur_arm_base_link(arm_mount_joint): {  }
ur_arm_base_link_inertia(ur_arm_base_link-base_link_inertia): {  }
ur_arm_base_link_inertia_visual(ur_arm_base_link_inertia): { Q: [0, 0, 0, 0, 0, 0, 1.0], shape: mesh, mesh: <ur_description/meshes/ur5e/visual/base.obj>, visual: true }
ur_arm_base_link_inertia_coll(ur_arm_base_link_inertia): { Q: [0, 0, 0, 0, 0, 0, 1.0], shape: mesh, mesh: <ur_description/meshes/ur5e/collision/base.stl>, contact: -2 }
ur_arm_base(ur_arm_base_link-base_fixed_joint): {  }
ur_arm_shoulder_link(ur_arm_shoulder_pan_joint): {  }
ur_arm_shoulder_link_visual(ur_arm_shoulder_link): { Q: [0, 0, 0, 0, 0, 0, 1.0], shape: mesh, mesh: <ur_description/meshes/ur5e/visual/shoulder.obj>, visual: true }
ur_arm_shoulder_link_coll(ur_arm_shoulder_link): { Q: [0, 0, 0, 0, 0, 0, 1.0], shape: mesh, mesh: <ur_description/meshes/ur5e/collision/shoulder.stl>, contact: -2 }
ur_arm_upper_arm_link(ur_arm_shoulder_lift_joint): {  }
ur_arm_upper_arm_link_visual(ur_arm_upper_arm_link): { Q: [0, 0, 0.138, 0.5, 0.5, -0.5, -0.5], shape: mesh, mesh: <ur_description/meshes/ur5e/visual/upperarm.obj>, visual: true }
ur_arm_upper_arm_link_coll(ur_arm_upper_arm_link): { Q: [0, 0, 0.138, 0.5, 0.5, -0.5, -0.5], shape: mesh, mesh: <ur_description/meshes/ur5e/collision/upperarm.stl>, contact: -2 }
ur_arm_forearm_link(ur_arm_elbow_joint): {  }
ur_arm_forearm_link_visual(ur_arm_forearm_link): { Q: [0, 0, 0.007, 0.5, 0.5, -0.5, -0.5], shape: mesh, mesh: <ur_description/meshes/ur5e/visual/forearm.obj>, visual: true }
ur_arm_forearm_link_coll(ur_arm_forearm_link): { Q: [0, 0, 0.007, 0.5, 0.5, -0.5, -0.5], shape: mesh, mesh: <ur_description/meshes/ur5e/collision/forearm.stl>, contact: -2 }
ur_arm_wrist_1_link(ur_arm_wrist_1_joint): {  }
ur_arm_wrist_1_link_visual(ur_arm_wrist_1_link): { Q: [0, 0, -0.127, 0.7071068, 0.7071068, 0, 0], shape: mesh, mesh: <ur_description/meshes/ur5e/visual/wrist1.obj>, visual: true }
ur_arm_wrist_1_link_coll(ur_arm_wrist_1_link): { Q: [0, 0, -0.127, 0.7071068, 0.7071068, 0, 0], shape: mesh, mesh: <ur_description/meshes/ur5e/collision/wrist1.stl>, contact: -2 }
ur_arm_wrist_2_link(ur_arm_wrist_2_joint): {  }
ur_arm_wrist_2_link_visual(ur_arm_wrist_2_link): { Q: [0, 0, -0.0997], shape: mesh, mesh: <ur_description/meshes/ur5e/visual/wrist2.obj>, visual: true }
ur_arm_wrist_2_link_coll(ur_arm_wrist_2_link): { Q: [0, 0, -0.0997], shape: mesh, mesh: <ur_description/meshes/ur5e/collision/wrist2.stl>, contact: -2 }
ur_arm_wrist_3_link(ur_arm_wrist_3_joint): {  }
ur_arm_wrist_3_link_visual(ur_arm_wrist_3_link): { Q: [0, 0, -0.0989, 0.7071068, 0.7071068, 0, 0], shape: mesh, mesh: <ur_description/meshes/ur5e/visual/wrist3.obj>, visual: true }
ur_arm_wrist_3_link_coll(ur_arm_wrist_3_link): { Q: [0, 0, -0.0989, 0.7071068, 0.7071068, 0, 0], shape: mesh, mesh: <ur_description/meshes/ur5e/collision/wrist3.stl>, contact: -2 }
ur_arm_flange(ur_arm_wrist_3-flange): {  }
ur_arm_tool0(ur_arm_flange-tool0): {  }
robotiq_85_mount(mount_on_flange): {  }

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
top_plate_joint_origin(base_link): { Q: [0.0812, 0, 0.225] }
top_plate_joint(top_plate_joint_origin): { joint: rigid }
ipad_rack_joint_origin(base_link): { Q: [0.0812, 0, 0.225] }
ipad_rack_joint(ipad_rack_joint_origin): { joint: rigid }
top_plate_front_joint_origin(top_plate_link): { Q: [0.4125, 0, 0.00672] }
top_plate_front_joint(top_plate_front_joint_origin): { joint: rigid }
top_plate_rear_joint_origin(top_plate_link): { Q: [-0.4125, 0, 0.00672] }
top_plate_rear_joint(top_plate_rear_joint_origin): { joint: rigid }
arm_mount_joint_origin(top_plate_front_link): { Q: [-0.105, 0, 0, -0.0, 0, 0, 1.0] }
arm_mount_joint(arm_mount_joint_origin): { joint: rigid }
ur_arm_base_link-base_link_inertia_origin(ur_arm_base_link): { Q: [0, 0, 0, 0.7071068, 0, 0, -0.7071068] }
ur_arm_base_link-base_link_inertia(ur_arm_base_link-base_link_inertia_origin): { joint: rigid }
ur_arm_shoulder_pan_joint_origin(ur_arm_base_link_inertia): { Q: [0, 0, 0.1625] }
ur_arm_shoulder_pan_joint(ur_arm_shoulder_pan_joint_origin): { joint: hingeZ, limits: [-6.28318530718, 6.28318530718] }
ur_arm_shoulder_lift_joint_origin(ur_arm_shoulder_link): { Q: [0, 0, 0, 0.7071068, 0.7071068, 0, 0] }
ur_arm_shoulder_lift_joint(ur_arm_shoulder_lift_joint_origin): { joint: hingeZ, limits: [-6.28318530718, 6.28318530718] }
ur_arm_elbow_joint_origin(ur_arm_upper_arm_link): { Q: [-0.425, 0, 0] }
ur_arm_elbow_joint(ur_arm_elbow_joint_origin): { joint: hingeZ, limits: [-3.14159265359, 3.14159265359] }
ur_arm_wrist_1_joint_origin(ur_arm_forearm_link): { Q: [-0.3922, 0, 0.1333] }
ur_arm_wrist_1_joint(ur_arm_wrist_1_joint_origin): { joint: hingeZ, limits: [-6.28318530718, 6.28318530718] }
ur_arm_wrist_2_joint_origin(ur_arm_wrist_1_link): { Q: [0, -0.0997, 0, 0.7071068, 0.7071068, 0, 0] }
ur_arm_wrist_2_joint(ur_arm_wrist_2_joint_origin): { joint: hingeZ, limits: [-6.28318530718, 6.28318530718] }
ur_arm_wrist_3_joint_origin(ur_arm_wrist_2_link): { Q: [0, 0.0996, 0, 0.7071068, -0.7071068, 0, 0] }
ur_arm_wrist_3_joint(ur_arm_wrist_3_joint_origin): { joint: hingeZ, limits: [-6.28318530718, 6.28318530718] }
ur_arm_base_link-base_fixed_joint_origin(ur_arm_base_link): { Q: [0, 0, 0, 0.7071068, 0, 0, -0.7071068] }
ur_arm_base_link-base_fixed_joint(ur_arm_base_link-base_fixed_joint_origin): { joint: rigid }
ur_arm_wrist_3-flange_origin(ur_arm_wrist_3_link): { Q: [0, 0, 0, 0.5, -0.5, -0.5, -0.5] }
ur_arm_wrist_3-flange(ur_arm_wrist_3-flange_origin): { joint: rigid }
ur_arm_flange-tool0_origin(ur_arm_flange): { Q: [0, 0, 0, 0.5, 0.5, 0.5, 0.5] }
ur_arm_flange-tool0(ur_arm_flange-tool0_origin): { joint: rigid }
mount_on_flange_origin(ur_arm_tool0): { Q: [0, 0, 0] }
mount_on_flange(mount_on_flange_origin): { joint: rigid }