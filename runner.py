from __future__ import annotations

from dataclasses import dataclass
import argparse
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from algorithms.algorithm_factory import build_algorithm

@dataclass
class RunnerConfig:
    algorithm: str = "mappo_v01"
    experiment_name: str = "mappo_v01_smoke(50,100)"

    total_episodes: int = 50
    episode_length: int = 200
    checkpoint_interval: int = 100                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

    n_agents: int = 2
    seed: int = 0
    verbose_options: bool = False

def run_experiment(env, algorithm, cfg: RunnerConfig) -> None:
    run_dir = os.path.join("runs", cfg.experiment_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=run_dir, flush_secs=5)

    try:
        for episode in range(cfg.total_episodes):
            obs, shared_obs, _ = env.reset(seed=cfg.seed + episode)
            episode_reward = np.zeros(env.n_agents, dtype=np.float32)
            option_durations = []
            option_reached = []
            distance_progress = []
            valid_candidate_counts = []
            minimum_candidate_distances = []
            rewards_by_action: dict[int, list[float]] = {}

            for step in range(cfg.episode_length):
                actions = algorithm.act(
                    obs=obs,
                    shared_obs=shared_obs,
                    training=True,
                    action_masks=env.get_action_masks(),
                )

                next_obs, next_shared_obs, rewards, dones, infos = env.step(actions)

                algorithm.observe(
                    obs=obs,
                    shared_obs=shared_obs,
                    actions=actions,
                    rewards=rewards,
                    dones=dones,
                    next_obs=next_obs,
                    next_shared_obs=next_shared_obs,
                    infos=infos,
                )

                rewards_flat = np.asarray(
                    rewards,
                    dtype=np.float32,
                ).reshape(env.n_agents, -1)[:, 0]

                episode_reward += rewards_flat
                for agent_id, info in enumerate(infos):
                    action_id = int(
                        info.get("frontier_index", info.get("selected_action", actions[agent_id]))
                    )
                    option_durations.append(float(info.get("option_duration", 0)))
                    option_reached.append(float(info.get("waypoint_reached", False)))
                    distance_progress.append(float(info.get("distance_progress", 0.0)))
                    valid_candidate_counts.append(
                        float(info.get("valid_frontier_candidates", 0))
                    )
                    minimum_candidate_distances.append(
                        float(info.get("minimum_candidate_distance", 0.0))
                    )
                    rewards_by_action.setdefault(action_id, []).append(
                        float(rewards_flat[agent_id])
                    )

                obs = next_obs
                shared_obs = next_shared_obs

                if np.all(dones):
                    break

            metrics = algorithm.update(
                next_obs=obs,
                next_shared_obs=shared_obs,
            )

            avg_reward = float(episode_reward.mean())
            coverage = float(env.coverage_tracker.coverage_ratio)
            episode_length = step + 1
            collision_count = int(env.episode_collision_count)
            mean_option_duration = float(np.mean(option_durations))
            target_reached_fraction = float(np.mean(option_reached))
            mean_distance_progress = float(np.mean(distance_progress))
            mean_valid_candidates = float(np.mean(valid_candidate_counts))
            positive_pairwise = [x for x in minimum_candidate_distances if x > 0.0]
            mean_min_candidate_distance = (
                float(np.mean(positive_pairwise)) if positive_pairwise else 0.0
            )
            mean_reward_by_action = {
                action: float(np.mean(values))
                for action, values in sorted(rewards_by_action.items())
            }

            print(
                f"[{cfg.algorithm.upper()}] "
                f"[EP {episode:03d}] "
                f"reward={avg_reward:.3f} "
                f"steps={episode_length} "
                f"coverage={coverage:.3f} "
                f"collisions={collision_count} "
                f"reached={target_reached_fraction:.2f} "
                f"progress={mean_distance_progress:.2f}"
            )
            if cfg.verbose_options:
                print(
                    f"[OPTION] duration={mean_option_duration:.1f} "
                    f"valid_candidates={mean_valid_candidates:.1f} "
                    f"min_candidate_distance={mean_min_candidate_distance:.3f} "
                    f"reward_by_action={mean_reward_by_action}"
                )

            writer.add_scalar("Training/Reward", avg_reward, episode)
            writer.add_scalar("Training/Coverage", coverage, episode)
            writer.add_scalar("Training/EpisodeLength", episode_length, episode)
            writer.add_scalar("Training/CollisionCount", collision_count, episode)
            writer.add_scalar("Option/MeanDuration", mean_option_duration, episode)
            writer.add_scalar("Option/TargetReachedFraction", target_reached_fraction, episode)
            writer.add_scalar("Option/MeanDistanceProgress", mean_distance_progress, episode)
            writer.add_scalar("Option/MeanValidCandidates", mean_valid_candidates, episode)
            writer.add_scalar(
                "Option/MeanMinimumCandidateDistance",
                mean_min_candidate_distance,
                episode,
            )
            for action_id, action_reward in mean_reward_by_action.items():
                writer.add_scalar(f"Option/RewardByAction/{action_id}", action_reward, episode)

            for metric_name, metric_value in metrics.items():
                writer.add_scalar(metric_name, metric_value, episode)

            writer.flush()

            if (
                cfg.checkpoint_interval > 0
                and (episode + 1) % cfg.checkpoint_interval == 0
            ):
                algorithm.save(
                    os.path.join(
                        checkpoint_dir,
                        f"ep_{episode + 1}.pt",
                    ),
                    episode=episode + 1,
                )
                algorithm.save(
                    os.path.join(
                        checkpoint_dir,
                        "latest.pt",
                    ),
                    episode=episode + 1,
                )

        algorithm.save(
            os.path.join(checkpoint_dir, "final.pt"),
            episode=cfg.total_episodes,
        )

    finally:
        writer.close()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="mappo_v01", choices=["mappo_v01", "random"])
    parser.add_argument("--env", default="mujoco", choices=["mujoco", "simple2d"])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--n-agents", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment-name", default="mappo_discrete_frontier")
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument(
        "--verbose-options",
        action="store_true",
        help="Print detailed option diagnostics and reward-by-action values.",
    )
    args = parser.parse_args()
    cfg = RunnerConfig(
        algorithm=args.algorithm,
        experiment_name=args.experiment_name,
        total_episodes=args.episodes,
        episode_length=args.episode_length,
        checkpoint_interval=args.checkpoint_interval,
        n_agents=args.n_agents,
        seed=args.seed,
        verbose_options=args.verbose_options,
    )

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if args.env == "simple2d":
        from envs.simple_2d_env import Simple2DExplorationEnv

        env = Simple2DExplorationEnv(
            n_agents=cfg.n_agents,
            max_episode_steps=cfg.episode_length,
            action_mode="discrete_frontier",
        )
    else:
        from envs.exploration_env import ExplorationEnv

        env = ExplorationEnv(
            n_agents=cfg.n_agents,
            max_episode_steps=cfg.episode_length,
        )
    obs, shared_obs, _ = env.reset(seed=cfg.seed)

    algorithm = build_algorithm(
        cfg.algorithm,
        env=env,
        obs=obs,
        shared_obs=shared_obs,
        seed=cfg.seed,
    )

    print("[RUNNER] algorithm:", cfg.algorithm)
    print("[RUNNER] environment:", args.env)
    print("[RUNNER] obs shape:", obs.shape)
    print("[RUNNER] shared obs shape:", shared_obs.shape)
    print("[RUNNER] action dim:", env.action_dim)
    print("[RUNNER] actor output shape: (batch,", env.action_dim, ")")

    run_experiment(env, algorithm, cfg)

if __name__ == "__main__":
    main()
