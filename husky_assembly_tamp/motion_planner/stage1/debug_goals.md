Observability and Debugging Goals  
Workspace tree visualisation. Plot the RRT tree’s Cartesian bar poses in the pybullet scene at each stage so the spatial exploration pattern is visible.  

Failure distribution analysis. Run random restarts with a fixed per-attempt time budget and distinct random seeds. Record, for each attempt, whether the failure occurred at the task-space level (RRT did not connect), IK level (no valid config found), or collision level (all paths are blocked). Visualise the failure distribution across attempts to identify the dominant bottleneck.  

Per-stage comparison. The three development stages are specifically designed so that differences in tree structure and planning success rate between stages directly reveal the contribution of each constraint (IK reachability, collision avoidance) to overall planning difficulty. 