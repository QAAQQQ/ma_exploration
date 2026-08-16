from __future__ import annotations
import numpy as np


class LocalMap:
    """
    Agent-local visited-memory map.

    This module intentionally does NOT use environment ground-truth occupancy.
    It only records where the agent itself has been.

    Coordinate convention:
        world position = [x, y, z] or [x, y]

    Grid convention:
        grid[row, col]
        row -> y
        col -> x

    Values:
        0.0 = not visited
        1.0 = visited
    """

    def __init__(
        self,
        world_bounds: tuple[float, float, float, float],
        resolution: float = 0.25,
        visit_radius: float = 0.4,
        patch_size_m: float = 5.0,
    ) -> None:
        xmin, xmax, ymin, ymax = world_bounds

        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"Invalid world_bounds: {world_bounds}")
        if resolution <= 0:
            raise ValueError("resolution must be > 0")
        if visit_radius < 0:
            raise ValueError("visit_radius must be >= 0")
        if patch_size_m <= 0:
            raise ValueError("patch_size_m must be > 0")

        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.ymin = float(ymin)
        self.ymax = float(ymax)

        self.resolution = float(resolution)
        self.visit_radius = float(visit_radius)
        self.patch_size_m = float(patch_size_m)

        self.width = int(np.ceil((self.xmax - self.xmin) / self.resolution))
        self.height = int(np.ceil((self.ymax - self.ymin) / self.resolution))

        patch_cells = int(np.ceil(self.patch_size_m / self.resolution))
        if patch_cells % 2 == 0:
            patch_cells += 1

        self.patch_cells = patch_cells
        self.patch_radius_cells = self.patch_cells // 2

        self.grid = np.zeros(
            (self.height, self.width),
            dtype=np.float32,
        )

    @property
    def observation_dim(self) -> int:
        """Flattened local-map observation size."""
        return self.patch_cells * self.patch_cells

    @property
    def visited_ratio(self) -> float:
        """
        Fraction of this agent-local map marked as visited.

        This is NOT the environment ground-truth coverage ratio.
        """
        return float(np.mean(self.grid))

    def reset(self) -> None:
        """Clear all local visited memory."""
        self.grid.fill(0.0)

    def update(self, robot_position: np.ndarray) -> int:
        """
        Mark the area around the current robot position as visited.

        Args:
            robot_position:
                [x, y] or [x, y, z] in world coordinates.

        Returns:
            Number of newly visited cells.
        """
        position = np.asarray(robot_position, dtype=np.float32)

        if position.size < 2:
            raise ValueError("robot_position must contain at least x and y")

        x = float(position[0])
        y = float(position[1])
        row, col = self.world_to_grid(x, y)

        radius_cells = int(np.ceil(self.visit_radius / self.resolution))

        row_min = max(0, row - radius_cells)
        row_max = min(self.height, row + radius_cells + 1)
        col_min = max(0, col - radius_cells)
        col_max = min(self.width, col + radius_cells + 1)

        newly_visited = 0
        radius_sq = self.visit_radius ** 2

        for r in range(row_min, row_max):
            for c in range(col_min, col_max):
                cell_x, cell_y = self.grid_to_world(r, c)
                distance_sq = (cell_x - x) ** 2 + (cell_y - y) ** 2

                if distance_sq <= radius_sq:
                    if self.grid[r, c] == 0.0:
                        newly_visited += 1
                    self.grid[r, c] = 1.0

        return newly_visited

    def get_local_patch(self, robot_position: np.ndarray) -> np.ndarray:
        """
        Return a fixed-size robot-centred visited patch.

        v0.1 behaviour:
            - world-axis aligned
            - NOT rotated by robot yaw
            - out-of-bounds cells are 0 (unknown / unvisited)

        Returns:
            shape (patch_cells, patch_cells)
        """
        position = np.asarray(robot_position, dtype=np.float32)

        if position.size < 2:
            raise ValueError("robot_position must contain at least x and y")

        row, col = self.world_to_grid(
            float(position[0]),
            float(position[1]),
        )

        patch = np.zeros(
            (self.patch_cells, self.patch_cells),
            dtype=np.float32,
        )

        src_row_min = max(0, row - self.patch_radius_cells)
        src_row_max = min(self.height, row + self.patch_radius_cells + 1)
        src_col_min = max(0, col - self.patch_radius_cells)
        src_col_max = min(self.width, col + self.patch_radius_cells + 1)

        dst_row_min = src_row_min - (row - self.patch_radius_cells)
        dst_col_min = src_col_min - (col - self.patch_radius_cells)
        dst_row_max = dst_row_min + (src_row_max - src_row_min)
        dst_col_max = dst_col_min + (src_col_max - src_col_min)

        patch[
            dst_row_min:dst_row_max,
            dst_col_min:dst_col_max,
        ] = self.grid[
            src_row_min:src_row_max,
            src_col_min:src_col_max,
        ]

        return patch

    def get_observation(self, robot_position: np.ndarray) -> np.ndarray:
        """
        Return the fixed-size policy representation of this local map.

        v0.1 representation:
            flattened visited patch
        """
        return self.get_local_patch(robot_position).reshape(-1).astype(np.float32)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert world coordinates to integer grid index (row, col)."""
        col = int(np.floor((x - self.xmin) / self.resolution))
        row = int(np.floor((y - self.ymin) / self.resolution))

        col = int(np.clip(col, 0, self.width - 1))
        row = int(np.clip(row, 0, self.height - 1))

        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Return world coordinate of the centre of a grid cell."""
        x = self.xmin + (col + 0.5) * self.resolution
        y = self.ymin + (row + 0.5) * self.resolution
        return float(x), float(y)