from .base_algorithm import BaseAlgorithm
from .random_policy import RandomPolicy
from .algorithm_factory import build_algorithm

__all__ = [
    "BaseAlgorithm", "RandomPolicy", "build_algorithm", "MAPPO", "MAPPOConfig", "DEVICE"
]


def __getattr__(name):
    """Keep lightweight/random usage independent from importing CUDA MAPPO."""
    if name in {"MAPPO", "MAPPOConfig", "DEVICE"}:
        from . import mappo_v01

        return getattr(mappo_v01, name)
    raise AttributeError(name)
