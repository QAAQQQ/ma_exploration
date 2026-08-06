# random movement test，验证环境不会坏以及lidar和action有没有正常的在跑

import numpy as np
from envs.exploration_env import ExplorationEnv

import time
import mujoco
import mujoco.viewer



def get_robot_positions(env: ExplorationEnv) -> np.ndarray:
    positions = []

    for agent_id in range(env.n_agents):
        body_id = mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"robot_{agent_id}",
        )

        if body_id == -1:
            raise RuntimeError(
                f"Cannot find body robot_{agent_id}"
            )

        positions.append(
            env.data.xpos[body_id].copy()
        )

    return np.asarray(positions)


def main() -> None:
    env = ExplorationEnv(
        n_agents=2,
        max_episode_steps=500,
        frame_skip=20,
    )

    obs, shared_obs, info = env.reset(seed=0)

    print("========== RESET ==========")
    print("obs shape:", obs.shape)
    print("shared obs shape:", shared_obs.shape)
    print("spawn positions:", info["spawn_positions"])

    rng = np.random.default_rng(seed=0)

    actions = np.zeros(
        (env.n_agents, 3),
        dtype=np.float32,
    )

    with mujoco.viewer.launch_passive(
        env.model,
        env.data,
    ) as viewer:

        viewer.cam.azimuth = 90
        viewer.cam.elevation = -90
        viewer.cam.distance = 30
        viewer.cam.lookat[:] = [10.0, 10.0, 0.0]

        # 单一action测试
        # actions = np.array([
        #         [1.0, 0.0, 0.0],   # robot_0 沿 x 正方向移动
        #         [0.0, 1.0, 0.0],   # robot_1 沿 y 正方向移动
        #     ],dtype=np.float32)

        actions = np.array(
        [
            [0.8, 0.5],    # robot_0 前进并向左转
            [0.8, -0.5],   # robot_1 前进并向右转
        ],
        dtype=np.float32,
    )

        # 检查observation
        previous_obs = obs.copy()

        # actions = np.zeros(
        #     (env.n_agents, 2),
        #     dtype=np.float32,
        # )
        
        for step in range(env.max_episode_steps):
            if not viewer.is_running():
                break

            # 不要每个 step 都换动作，否则机器人会疯狂抖动。
            # # 每 100 个环境 step 换一个随机方向。一个step是0.002sec 
            # if step % 80 == 0:
            #     actions[:, 0] = rng.uniform(
            #         0.5,
            #         1.0,
            #         size=env.n_agents,
            #     )

            #     actions[:, 1] = rng.uniform(
            #         -0.5,
            #         0.5,
            #         size=env.n_agents,
            #     )

            #     print(
            #         f"new actions at step {step}:",
            #         actions.round(2),
            #     )

            (obs,shared_obs,rewards,dones,infos,) = env.step(actions)

            viewer.sync()

            # 限制显示速度，否则 viewer 中可能跑得太快。
            time.sleep(0.02)

            ########################## 
             # 1. 检查 NaN / Inf
            if not np.isfinite(obs).all():
                raise RuntimeError(
                    f"Invalid obs detected at step {step}: "
                    f"nan={np.isnan(obs).sum()}, "
                    f"inf={np.isinf(obs).sum()}"
                )

            if not np.isfinite(shared_obs).all():
                raise RuntimeError(
                    f"Invalid shared_obs detected at step {step}"
                )

            if not np.isfinite(rewards).all():
                raise RuntimeError(
                    f"Invalid rewards detected at step {step}"
                )

            # 2. 检查 LiDAR observation 是否变化
            obs_change = np.abs(obs - previous_obs)

            mean_change = obs_change.mean()
            max_change = obs_change.max()
            changed_ratio = np.mean(obs_change > 1e-6)

            if step % 10 == 0:
                positions = get_robot_positions(env)

                print(
                    f"step={step:03d} "
                    f"positions={positions[:, :2].round(3).tolist()} "
                    f"obs_min={obs.min():.4f} "
                    f"obs_max={obs.max():.4f} "
                    f"obs_mean={obs.mean():.4f} "
                    f"mean_change={mean_change:.6f} "
                    f"max_change={max_change:.6f} "
                    f"changed_ratio={changed_ratio:.3f}"
                )

            previous_obs = obs.copy()

            ##########################    
            #显示机器人位置 
            if step % 10 == 0:
                positions = get_robot_positions(env)
                coverage_ratio = infos[0]["coverage_ratio"]
                newly_explored = infos[0]["newly_explored"]

                print(
                    f"step={step:03d}",
                    f"positions={positions[:, :2].round(2).tolist()}",
                    f"obs_mean={obs.mean():.3f}",
                    f"new_cells={newly_explored}",
                    f"coverage={coverage_ratio:.3%}",
                    f"reward={rewards[0, 0]:.2f}",
                )



            if dones.all():
                print("Episode finished.")
                break


if __name__ == "__main__":
    main()