export ROS_PACKAGE_PATH=$(rospack find husky_assembly)/data/husky_urdf:$ROS_PACKAGE_PATH
roslaunch husky_assembly robot.launch