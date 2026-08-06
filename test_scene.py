from scene.map_generator import MapGenerator
from scene.map_to_scene import MapToScene
from scene.robot_definition import RobotConfig
from scene.mujoco_builder import MujocoBuilder

import math
import mujoco
import mujoco.viewer
import jax.numpy as jnp
from mujoco_lidar import MjLidarWrapper, scan_gen


if __name__ == "__main__":
    print("================")
    print("Generate map")
    print("================")

    generator = MapGenerator()
    semantic_map = generator.generate()

    # 可视化2D map 
    try:
        semantic_map.visualise() # 其实这里应该是generator，但是我不想跑这个
    except Exception as e:
        print(
            "skip map visualization:",
            e                         
        )                              
    print("================")    
    print("Convert scene")
    print("================")

    converter = MapToScene()
    scene = converter.convert(semantic_map)
  

    print("walls:",len(scene.walls))
    print("floors:",len(scene.floors))

    print("================")    
    print("Initialise Robots")
    print("================")
    
    spawn_pos = converter.sample_spawn_position(semantic_map,n_agents=2)
    print("spawn_pos_red:",spawn_pos[0])
    print("spawn_pos_blue:",spawn_pos[1])

    robot_list = [
        RobotConfig(
            name="robot_0",
            position=spawn_pos[0],
            color=(1.0, 0.0, 0.0, 1.0),  # red
            model_type="box",
            size=(0.2, 0.2, 0.5),
        ),

        RobotConfig(
            name="robot_1",
            position=spawn_pos[1],
            color=(0.0, 0.0, 1.0, 1.0),  # blue
            model_type="box",
            size=(0.2, 0.2, 0.5),
        )
    ]

    print("================")
    print("Build MuJoCo")
    print("================")

    builder = MujocoBuilder()
    model = builder.build(scene,robot_list)
    data = mujoco.MjData(model)

    # builder.save_xml("test_scene.xml")
    print("geom number:",model.ngeom)

    print("================")
    print("Lidar")
    print("================")

    lidar1 = MjLidarWrapper(model, site_name = "robot_0_lidar_site", backend="cpu",cutoff_dist = 5.0)

    # batch_data = jnp.stack([data.qpos for _ in range(1)])
    # ranges = lidar1.trace_rays_batch(batch_data, theta, phi)
    # Generate scan pattern
    theta, phi = scan_gen.generate_HDL64()

    # Trace rays
    ranges = lidar1.trace_rays(data, theta, phi)
    print(f"Scanned {len(ranges)} points")
    print(type(ranges))
    print(ranges.shape)

    print("================")
    print("Launch viewer")
    print("================") 

    with mujoco.viewer.launch_passive(
        model,
        data
    ) as viewer:

        # 调整初始视角
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -90
        viewer.cam.distance = 30

        viewer.cam.lookat[:] = [10,10,0]

        while viewer.is_running():
            mujoco.mj_step(model,data)
            viewer.sync()