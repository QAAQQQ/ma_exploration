from __future__ import annotations
import numpy as np
from .base_algorithm import BaseAlgorithm

class RandomPolicy(BaseAlgorithm):
    def __init__(
        self,
        n_agents: int,
        action_dim: int = 2,
        seed: int | None = None,
        discrete: bool = True,
    ):
        self.n_agents = int(n_agents)
        self.action_dim = int(action_dim)
        self.rng = np.random.default_rng(seed)
        self.discrete = bool(discrete)

    def act(
        self,
        obs,
        shared_obs=None,
        training: bool = True,
        action_masks=None,
    ) -> np.ndarray:
        if not self.discrete:
            return self.rng.uniform(
                -1.0, 1.0, size=(self.n_agents, self.action_dim)
            ).astype(np.float32)
        if action_masks is None:
            action_masks = np.ones((self.n_agents, self.action_dim), dtype=bool)
        action_masks = np.asarray(action_masks, dtype=bool)
        actions = np.zeros(self.n_agents, dtype=np.int64)
        for agent_id in range(self.n_agents):
            valid_actions = np.flatnonzero(action_masks[agent_id])
            actions[agent_id] = (
                self.rng.choice(valid_actions) if len(valid_actions) else 0
            )
        return actions
