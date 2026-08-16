# wrapper for all agent knowledge

from __future__ import annotations
import numpy as np
from .local_map import LocalMap
from .occupancy_map import OccupancyMap

class AgentKnowledge:
    """
    Container for one agent's internal knowledge.

    Current v0.1:
        - only contains a visited LocalMap

    Future extensions may include:
        - occupancy map
        - frontier information
        - teammate information
        - received communication
        - semantic memory

    This class is intentionally independent of any RL algorithm.
    """

    def __init__(
        self,
        world_bounds: tuple[float, float, float, float],
        map_resolution: float = 0.25,
        visit_radius: float = 0.4,
        patch_size_m: float = 5.0,
    ) -> None:
        self.local_map = LocalMap(
            world_bounds=world_bounds,
            resolution=map_resolution,
            visit_radius=visit_radius,
            patch_size_m=patch_size_m,
        )
        self.occupancy_map = OccupancyMap(
            world_bounds=world_bounds,
            resolution=map_resolution,
        )

    @property
    def observation_dim(self) -> int:
        """
        Size of the knowledge representation returned to the policy.
        """
        return self.local_map.observation_dim

    def reset(self) -> None:
        """
        Reset all agent-side knowledge.
        """
        self.local_map.reset()
        self.occupancy_map.reset()

    def update_occupancy(
        self,
        hit_points_local: np.ndarray,
        robot_position: np.ndarray,
        robot_yaw: float,
    ) -> dict[str, int]:
        """Update the diagnostic occupancy map without changing policy input."""
        return self.occupancy_map.update(
            hit_points_local=hit_points_local,
            robot_position=robot_position,
            robot_yaw=robot_yaw,
        )

    def update(
        self,
        robot_position: np.ndarray,
    ) -> dict[str, float | int]:
        """
        Update internal knowledge from the agent's latest state.

        Current v0.1 only updates visited memory using robot position.

        Args:
            robot_position:
                [x, y] or [x, y, z] in world coordinates.

        Returns:
            Lightweight diagnostics for logging/debugging.
        """
        newly_visited = self.local_map.update(
            robot_position
        )

        return {
            "newly_visited_cells": newly_visited,
            "visited_ratio": self.local_map.visited_ratio,
        }

    def get_observation(
        self,
        robot_position: np.ndarray,
    ) -> np.ndarray:
        """
        Return the agent-side knowledge representation for the policy.

        Current v0.1:
            flattened robot-centred visited-map patch

        Returns:
            shape (observation_dim,)
        """
        return self.local_map.get_observation(
            robot_position
        )

    def get_local_patch(
        self,
        robot_position: np.ndarray,
    ) -> np.ndarray:
        """
        Return the 2D visited-map patch without flattening.

        Useful for visualization/debugging.
        """
        return self.local_map.get_local_patch(
            robot_position
        )
