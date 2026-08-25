from __future__ import annotations

import numpy as np

from .agent_map import AgentMap


class AgentKnowledge:
    """Thin owner/wrapper around one agent's single knowledge map."""

    def __init__(
        self,
        world_bounds: tuple[float, float, float, float],
        map_resolution: float = 0.25,
        visit_radius: float = 0.4,
        patch_size_m: float = 5.0,
    ) -> None:
        # Kept only for call-site compatibility; there is no visited layer now.
        del visit_radius
        self.map = AgentMap(
            world_bounds=world_bounds,
            resolution=map_resolution,
            patch_size_m=patch_size_m,
        )

    @property
    def observation_dim(self) -> int:
        return self.map.observation_dim

    def reset(self) -> None:
        self.map.reset()

    def update_from_lidar(
        self,
        hit_points_local: np.ndarray,
        robot_position: np.ndarray,
        robot_yaw: float,
    ) -> dict[str, int]:
        return self.map.update_from_lidar(
            hit_points_local=hit_points_local,
            robot_position=robot_position,
            robot_yaw=robot_yaw,
        )

    def get_frontier_candidates(self) -> np.ndarray:
        return self.map.compute_frontier_candidates()

    def get_map_observation(self, robot_position: np.ndarray) -> np.ndarray:
        return self.map.get_map_observation(robot_position)

    def get_local_patch(self, robot_position: np.ndarray) -> np.ndarray:
        return self.map.get_local_patch(robot_position)
