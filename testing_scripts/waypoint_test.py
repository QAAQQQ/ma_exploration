import time
import numpy as np
import mujoco
import mujoco.viewer

from envs.exploration_env import ExplorationEnv


def set_camera(viewer, env):
    """
    Automatically place the camera above the generated map.
    """

    x_min, x_max, y_min, y_max = env.world_bounds

    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)

    world_size = max(
        x_max - x_min,
        y_max - y_min,
    )

    viewer.cam.lookat[:] = [
        center_x,
        center_y,
        0.0,
    ]

    viewer.cam.distance = world_size * 1.2
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -85


def main():

    env = ExplorationEnv(
        n_agents=2,
        frame_skip=1,
        max_episode_steps=500,
    )

    obs, shared_obs, info = env.reset(seed=0)

    print("Environment created.")
    print("Close the viewer to exit.")

    with mujoco.viewer.launch_passive(
        env.model,
        env.data,
    ) as viewer:

        set_camera(viewer, env)

        rng = np.random.default_rng(0)

        while viewer.is_running():

            # Random waypoint action
            actions = rng.uniform(
                low=-1.0,
                high=1.0,
                size=(env.n_agents, 2),
            )

            obs, shared_obs, rewards, dones, infos = env.step(
                actions
            )

            viewer.sync()

            time.sleep(0.03)

            if np.all(dones):

                print("Episode finished.")

                break


if __name__ == "__main__":
    main()