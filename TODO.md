# Reminders: 

Major
- [x] Do smoothing as Caelan suggested
- [] Add tool collision geo, handle ACM properly
- [] Hardware test with mockups on both end-factors
- [] Hardware test with real tools and bars
- [] integrate into the monitor to allow live replanning

Minor
- [] If we use the analytical ik from [ur-analytic-ik](https://github.com/Victorlouisdg/ur-analytic-ik?tab=readme-ov-file), remember to use the factory calibrated model
- [] self collision is not handled correctly, arm and base

# Prompt
X Now the stage 1, 2, 3 all work pretty nicely, but the only caveat is that the joint continuity is not considered. In some of the paths, I can see that the robot can flip its joint dramatically between two neighboring configurations. For example, look at the attached image.  I don't think you can resolve it in the RRT algorithm because, by nature, it's doing warm start per capsule? But correct me if you think I am wrong. If this happens, there are two solutions:
1. We can increase the interpolation resolution (like the position resolution and the angular resolutions) and see if that helps.
2. We can use the ladder letter graph approach. For each capsule found in the path, we solve all possible IK combinations, and then we do a ladder graph type of search. Each node in this graph represents a pair of configurations, and then we can do a DAG search to find the shortest path that minimizes accumulated joint difference in the path. This will give us a kind of the optimal solutions in terms of ensuring joint smoothness. You can find an useful implementation I had here: 
In general, I think we can just do this first coarse resolution to ensure solution exists, and then we can refine it by first increasing the resolution. If that still doesn't work, we fall back to the letter graph approach.

X Report now: the success rate should include joint continuity. So now it shows it succeeded, but actually the draw continuity failed. 

X In general, I think DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD = 0.5 is too large to be tracked nicely on hardware. A reasonable  value should be around 0.2. Let's re-run the report to see what would happen, And also figure out what would be the ideal
  position and rotational resolution to achieve this

Do case study on different goal bar poses.

Benchmark changing pybullet IK to Track IK. 


❯ now let's plan to add a smoothing function to post process the planned path so it can be optimized. The logic is
  similar to this function: /Users/huangyijiang/Code/husky-assembly-teleop/external/pybullet_planning/src/pybullet_plann
  ing/motion_planners/smoothing.py:35 but we need to adapt it to respect the constraints.

## Smoothing
  This is how it works:
  - randomly pick two indices in the path, and then do a shortcut (linear interpolation in the task space of the bar) if
  the resulting path cost is improved.
  - to ensure that the dual-arm constraints are respected (joint continuity, end effector constraints, and
  collision-free), we take the first index from which the shortcut path starts, and we take the previously solved joint
  conf there and we warm start IK solving there using @husky_assembly_tamp/motion_planner/stage1/minimal_rrt.py:689
  solve_dual_arm_pose_ik and we update the rest of the bar task space path until the very end (not just the end of the
  shortcut path). If IK cannot be found or collision is violated, we skip this one and try sample a index pair again.

  Let's discuss

## Tool modeling

I want to add the tools that are moounted on the robot's boyth arms' tool0 and also include its collision to the planning.
This means in addition to the collision checks we already have now in @minimal_rrt, we also check the following:
- tool-robot body collision, this means left arm tool collision with any part of the robot body (including the left arm, right arm, the robot base, etc.)
- left tool - right tool collision
- tool - environment static collision obstacles

Note that since the tool is mounted on the tool0, collisions between the left (right) tool and left (right)_ur_arm_wrist_3_link should be ignored.
Also, collisions between the tool and the bar should be ignored.

For loading the geometry of the tool, we want to load from the real tool geometry from the RobotCell and RobotCellState. You can refer to what is done in /Users/huangyijiang/Code/husky-assembly-teleop/husky_assembly_teleop/common.py:185
