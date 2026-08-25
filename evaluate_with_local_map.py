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

    Saves the unified map, its clustered frontiers, and the final local patch.
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
        # 1. Full unified occupancy map
        # -------------------------------------------------
        full_map = (
            knowledge.map.grid
            .copy()
        )

        plt.figure(
            figsize=(6, 6)
        )

        plt.imshow(
            full_map,
            origin="lower",
            vmin=-1.0,
            vmax=1.0,
        )

        plt.title(
            f"Agent {agent_id} - Agent Map"
        )

        plt.xlabel("Grid X")
        plt.ylabel("Grid Y")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                save_dir,
                f"agent_{agent_id}_map.png",
            ),
            dpi=150,
        )

        plt.close()

        # -------------------------------------------------
        # 2. LiDAR occupancy map and frontier candidates
        # -------------------------------------------------
        snapshot = env.get_mapping_snapshot(agent_id)
        occupancy = knowledge.map
        display_grid = snapshot["occupancy_grid"].astype(np.float32)
        # Display convention: unknown=0.5 grey, free=1 white, occupied=0 black.
        display_grid = np.where(display_grid < 0, 0.5, 1.0 - display_grid)

        plt.figure(figsize=(7, 6))
        plt.imshow(
            display_grid,
            origin="lower",
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            extent=(occupancy.xmin, occupancy.xmax, occupancy.ymin, occupancy.ymax),
        )

        frontier_rows, frontier_cols = np.nonzero(snapshot["frontier_mask"])
        if len(frontier_rows):
            frontier_xy = np.asarray(
                [occupancy.grid_to_world(row, col)
                 for row, col in zip(frontier_rows, frontier_cols)]
            )
            plt.scatter(
                frontier_xy[:, 0], frontier_xy[:, 1],
                s=5, c="deepskyblue", label="frontier cells",
            )

        candidates = snapshot["frontier_candidates"]
        if len(candidates):
            plt.scatter(
                candidates[:, 0], candidates[:, 1],
                s=70, c="red", marker="x", label="candidates",
            )
        plt.scatter(
            positions[agent_id, 0], positions[agent_id, 1],
            s=60, c="lime", marker="o", edgecolors="black", label="robot",
        )
        plt.title(f"Agent {agent_id} - Occupancy and Frontiers")
        plt.xlabel("World X (m)")
        plt.ylabel("World Y (m)")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"agent_{agent_id}_occupancy_frontiers.png"),
            dpi=150,
        )
        plt.close()

        # -------------------------------------------------
        # 3. Final robot-centred local patch
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
            knowledge.map
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
