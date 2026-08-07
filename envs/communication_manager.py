# use to alter communication in future
from dataclasses import dataclass
import numpy as np


@dataclass
class CommunicationConfig:
    enabled: bool = False
    message_dim: int = 0
    max_range: float = np.inf
    interval: int = 1


class CommunicationManager:
    def __init__(self, config: CommunicationConfig, n_agents: int):
        self.config = config
        self.n_agents = n_agents

    def reset(self) -> None:
        pass

    def communication_allowed(self, step_count: int) -> bool:
        if not self.config.enabled:
            return False
        return step_count % self.config.interval == 0