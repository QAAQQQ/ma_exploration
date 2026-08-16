from __future__ import annotations
import numpy as np
from .base_algorithm import BaseAlgorithm

class RandomPolicy(BaseAlgorithm):
    def __init__(self, n_agents: int, action_dim: int = 2, seed: int | None = None):
        self.n_agents = int(n_agents)
        self.action_dim = int(action_dim)
        self.rng = np.random.default_rng(seed)

    def act(self, obs, shared_obs=None, training: bool = True) -> np.ndarray:
        return self.rng.uniform(
            low=-1.0,
            high=1.0,
            size=(self.n_agents, self.action_dim),
        ).astype(np.float32)