from __future__ import annotations

from .mappo_v01 import MAPPO, MAPPOConfig
from .random_policy import RandomPolicy

def build_algorithm(name: str, *, env, obs, shared_obs, seed: int = 0):
    name = name.lower()

    if name == "random":
        return RandomPolicy(
            n_agents=env.n_agents,
            action_dim=env.action_dim,
            seed=seed,
        )

    if name == "mappo_v01":
        agents = [f"agent_{i}" for i in range(env.n_agents)]
        return MAPPO(
            obs_dim=int(obs.shape[1]),
            action_dim=env.action_dim,
            global_obs_dim=int(shared_obs.shape[1]),
            agents=agents,
            config=MAPPOConfig(),
        )

    raise ValueError(f"Unknown algorithm: {name}")
