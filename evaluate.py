from __future__ import annotations

import argparse
import time

import mujoco.viewer
import numpy as np

from algorithms.mappo_v01 import (
    MAPPO,
    MAPPOConfig,
)
from envs.exploration_env import ExplorationEnv


def set_overview_camera(
    viewer,
    env: ExplorationEnv,
) -> None:
    """
    Configure the top-down camera used in the current 20 m x 20 m scene.

    Replace this with a bounds-based camera later if scene sizes vary.
    """
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -90
    viewer.cam.distance = 30
    viewer.cam.lookat[:] = [
        10.0,
        10.0,
        0.0,
    ]


def evaluate(
    checkpoint_path: str,
    *,
    seed: int = 0,
    episode_length: int = 200,
    step_sleep: float = 0.03,
) -> None:
    env = ExplorationEnv(
        n_agents=2,
    )

    obs, shared_obs, _ = env.reset(
        seed=seed
    )

    obs_dim = int(obs.shape[1])

    global_obs_dim = int(
        shared_obs.shape[1]
    )

    action_dim = env.action_dim

    agents = [
        f"agent_{i}"
        for i in range(
            env.n_agents
        )
    ]

    mappo = MAPPO(
        obs_dim=obs_dim,
        action_dim=action_dim,
        global_obs_dim=global_obs_dim,
        agents=agents,
        config=MAPPOConfig(),
    )

    checkpoint = mappo.load(
        checkpoint_path,
        load_optimizers=False,
    )

    for agent in agents:
        mappo.policies[agent].eval()
        mappo.critics[agent].eval()

    print(
        "[EVAL] loaded:",
        checkpoint_path,
    )

    print(
        "[EVAL] checkpoint episode:",
        checkpoint.get(
            "episode",
            "unknown",
        ),
    )

    with mujoco.viewer.launch_passive(
        env.model,
        env.data,
    ) as viewer:
        set_overview_camera(
            viewer,
            env,
        )

        episode_reward = np.zeros(
            env.n_agents,
            dtype=np.float32,
        )

        for step in range(
            episode_length
        ):
            if not viewer.is_running():
                break

            actions = mappo.act(
                obs=obs,
                shared_obs=shared_obs,
                training=False,
                action_masks=env.get_action_masks(),
            )

            (
                obs,
                shared_obs,
                rewards,
                dones,
                infos,
            ) = env.step(actions)

            rewards_flat = (
                np.asarray(
                    rewards,
                    dtype=np.float32,
                )
                .reshape(
                    env.n_agents,
                    -1,
                )[:, 0]
            )

            episode_reward += (
                rewards_flat
            )

            viewer.sync()

            print(
                f"[EVAL step {step:03d}] "
                f"reward="
                f"{rewards_flat.mean():.3f} "
                f"coverage="
                f"{env.coverage_tracker.coverage_ratio:.3f}"
            )

            time.sleep(step_sleep)

            if np.all(dones):
                print(
                    "[EVAL] episode finished."
                )
                break

        print(
            "[EVAL] mean episode reward:",
            float(
                episode_reward.mean()
            ),
        )

        print(
            "[EVAL] final coverage:",
            float(
                env.coverage_tracker.coverage_ratio
            ),
        )

        print(
            "[EVAL] viewer remains open. "
            "Close the window to exit."
        )

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Path to MAPPO checkpoint, "
            "e.g. runs/.../checkpoints/final.pt"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--episode-length",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--step-sleep",
        type=float,
        default=0.03,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        episode_length=args.episode_length,
        step_sleep=args.step_sleep,
    )


if __name__ == "__main__":
    main()
