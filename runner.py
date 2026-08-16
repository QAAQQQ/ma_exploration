from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from algorithms.algorithm_factory import build_algorithm
from envs.exploration_env import ExplorationEnv

@dataclass
class RunnerConfig:
    algorithm: str = "mappo_v01"
    experiment_name: str = "mappo_v01_smoke(50,100)"

    total_episodes: int = 50
    episode_length: int = 200
    checkpoint_interval: int = 100                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

    n_agents: int = 2
    seed: int = 0

def run_experiment(env, algorithm, cfg: RunnerConfig) -> None:
    run_dir = os.path.join("runs", cfg.experiment_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=run_dir, flush_secs=5)

    try:
        for episode in range(cfg.total_episodes):
            obs, shared_obs, _ = env.reset(seed=cfg.seed + episode)
            episode_reward = np.zeros(env.n_agents, dtype=np.float32)

            for step in range(cfg.episode_length):
                actions = algorithm.act(
                    obs=obs,
                    shared_obs=shared_obs,
                    training=True,
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

            print(
                f"[{cfg.algorithm.upper()}] "
                f"[EP {episode:03d}] "
                f"reward={avg_reward:.3f} "
                f"steps={episode_length} "
                f"coverage={coverage:.3f}"
            )

            writer.add_scalar("Training/Reward", avg_reward, episode)
            writer.add_scalar("Training/Coverage", coverage, episode)
            writer.add_scalar("Training/EpisodeLength", episode_length, episode)

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
    cfg = RunnerConfig()

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = ExplorationEnv(n_agents=cfg.n_agents)
    obs, shared_obs, _ = env.reset(seed=cfg.seed)

    algorithm = build_algorithm(
        cfg.algorithm,
        env=env,
        obs=obs,
        shared_obs=shared_obs,
        seed=cfg.seed,
    )

    print("[RUNNER] algorithm:", cfg.algorithm)
    print("[RUNNER] obs shape:", obs.shape)
    print("[RUNNER] shared obs shape:", shared_obs.shape)
    print("[RUNNER] action dim:", env.action_dim)

    run_experiment(env, algorithm, cfg)

if __name__ == "__main__":
    main()
