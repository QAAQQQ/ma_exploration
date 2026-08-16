from __future__ import annotations

import argparse
import os
import time

import matplotlib.pyplot as plt
import mujoco.viewer
import numpy as np

from algorithms.algorithm_factory import build_algorithm
from envs.exploration_env import ExplorationEnv


def set_overview_camera(
    viewer,
    env: ExplorationEnv,
) -> None:
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -90
    viewer.cam.distance = 30
    viewer.cam.lookat[:] = [
        10.0,
        10.0,
        0.0,
    ]


def save_agent_maps(
    env: ExplorationEnv,
    save_dir: str,
) -> None:
    """
    Save each agent's final knowledge map after evaluation.

    Saves:
        agent_X_visited_map.png
        agent_X_local_patch.png
    """
    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    positions = env._get_robot_positions()

    for agent_id in range(
        env.n_agents
    ):
        knowledge = env.agent_knowledge[
            agent_id
        ]

        # -------------------------------------------------
        # 1. Full visited-memory map
        # -------------------------------------------------
        full_map = (
            knowledge.local_map.grid
            .copy()
        )

        plt.figure(
            figsize=(6, 6)
        )

        plt.imshow(
            full_map,
            origin="lower",
            vmin=0.0,
            vmax=1.0,
        )

        plt.title(
            f"Agent {agent_id} - Visited Map"
        )

        plt.xlabel("Grid X")
        plt.ylabel("Grid Y")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                save_dir,
                f"agent_{agent_id}_visited_map.png",
            ),
            dpi=150,
        )

        plt.close()

        # -------------------------------------------------
        # 2. Final robot-centred local patch
        # -------------------------------------------------
        patch = (
            knowledge.get_local_patch(
                positions[agent_id]
            )
        )

        plt.figure(
            figsize=(5, 5)
        )

        plt.imshow(
            patch,
            origin="lower",
            vmin=0.0,
            vmax=1.0,
        )

        centre = (
            knowledge.local_map
            .patch_radius_cells
        )

        plt.scatter(
            [centre],
            [centre],
            marker="x",
            s=80,
        )

        plt.title(
            f"Agent {agent_id} - Final Local Patch"
        )

        plt.xlabel("Local Grid X")
        plt.ylabel("Local Grid Y")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                save_dir,
                f"agent_{agent_id}_local_patch.png",
            ),
            dpi=150,
        )

        plt.close()

    print(
        f"[EVAL] agent maps saved to: "
        f"{save_dir}"
    )


def evaluate(
    checkpoint_path: str | None,
    *,
    algorithm_name: str = "mappo",
    seed: int = 0,
    episode_length: int = 200,
    step_sleep: float = 0.03,
    save_dir: str = "evaluation_results",
) -> None:

    env = ExplorationEnv(
        n_agents=2,
    )

    obs, shared_obs, _ = env.reset(
        seed=seed
    )

    algorithm = build_algorithm(
        algorithm_name,
        env=env,
        obs=obs,
        shared_obs=shared_obs,
        seed=seed,
    )

    if checkpoint_path is not None:
        checkpoint = algorithm.load(
            checkpoint_path,
            load_optimizers=False,
        )

        if checkpoint is not None:
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

            actions = algorithm.act(
                obs=obs,
                shared_obs=shared_obs,
                training=False,
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

            if step_sleep > 0:
                time.sleep(
                    step_sleep
                )

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
                env.coverage_tracker
                .coverage_ratio
            ),
        )

        # Save knowledge maps BEFORE waiting for viewer closure.
        map_save_dir = os.path.join(
            save_dir,
            f"{algorithm_name}_seed_{seed}",
        )

        save_agent_maps(
            env,
            map_save_dir,
        )

        print(
            "[EVAL] Close viewer to exit."
        )

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algorithm",
        default="mappo_v01",
        choices=[
            "random",
            "mappo_v01",
        ],
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
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

    parser.add_argument(
        "--save-dir",
        default="evaluation_results",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        algorithm_name=args.algorithm,
        seed=args.seed,
        episode_length=args.episode_length,
        step_sleep=args.step_sleep,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
