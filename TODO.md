Reminders: 

- [] If we use the analytical ik from [ur-analytic-ik](https://github.com/Victorlouisdg/ur-analytic-ik?tab=readme-ov-file), remember to use the factory calibrated model

- [] self collision is not handled correctly, arm and base
- [] Make the GUI possible to export and save the start and end configuration. 

# Prompt
Now the stage 1, 2, 3 all work pretty nicely, but the only caveat is that the joint continuity is not considered. In some of the paths, I can see that the robot can flip its joint dramatically between two neighboring configurations. For example, look at the attached image.  I don't think you can resolve it in the RRT algorithm because, by nature, it's doing warm start per capsule? But correct me if you think I am wrong. If this happens, there are two solutions:
1. We can increase the interpolation resolution (like the position resolution and the angular resolutions) and see if that helps.
2. We can use the ladder letter graph approach. For each capsule found in the path, we solve all possible IK combinations, and then we do a ladder graph type of search. Each node in this graph represents a pair of configurations, and then we can do a DAG search to find the shortest path that minimizes accumulated joint difference in the path. This will give us a kind of the optimal solutions in terms of ensuring joint smoothness. You can find an useful implementation I had here: 
In general, I think we can just do this first coarse resolution to ensure solution exists, and then we can refine it by first increasing the resolution. If that still doesn't work, we fall back to the letter graph approach.