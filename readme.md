Notes:
/scene: manage 2d map to 3d mujoco scene convert, robot and lidar initialisation in scene
/envs: 
1. exploration_env： overall wrapper for env interaction



Simulation / Real Robot
        ↓
ROS-compatible sensor interface
        ↓
Mapping backend
   ├── 现在：occupancy update
   └── 以后：ROS slam_toolbox / costmap
        ↓
AgentKnowledge
        ↓
Representation
        ↓
Algorithm