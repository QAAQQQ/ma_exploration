from __future__ import annotations

from collections import deque

import numpy as np


class OccupancyMap:
    """LiDAR-built 2-D occupancy grid and frontier candidate generator.

    Grid values are UNKNOWN (-1), FREE (0), and OCCUPIED (1).  The map is a
    diagnostic/exploration representation only; it is not a policy input.
    """

    UNKNOWN = np.int8(-1)
    FREE = np.int8(0)
    OCCUPIED = np.int8(1)

    def __init__(
        self,
        world_bounds: tuple[float, float, float, float],
        resolution: float = 0.25,
        min_z: float = -0.2,
        max_z: float = 0.2,
        min_frontier_size: int = 2,
    ) -> None:
        xmin, xmax, ymin, ymax = world_bounds
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"Invalid world_bounds: {world_bounds}")
        if resolution <= 0:
            raise ValueError("resolution must be > 0")
        if min_z > max_z:
            raise ValueError("min_z must be <= max_z")
        if min_frontier_size <= 0:
            raise ValueError("min_frontier_size must be > 0")

        self.xmin, self.xmax = float(xmin), float(xmax)
        self.ymin, self.ymax = float(ymin), float(ymax)
        self.resolution = float(resolution)
        self.min_z, self.max_z = float(min_z), float(max_z)
        self.min_frontier_size = int(min_frontier_size)
        self.width = int(np.ceil((self.xmax - self.xmin) / self.resolution))
        self.height = int(np.ceil((self.ymax - self.ymin) / self.resolution))
        self.grid = np.full((self.height, self.width), self.UNKNOWN, dtype=np.int8)
        self.frontier_mask = np.zeros_like(self.grid, dtype=bool)
        self.frontier_candidates = np.empty((0, 2), dtype=np.float32)

    def reset(self) -> None:
        self.grid.fill(self.UNKNOWN)
        self.frontier_mask.fill(False)
        self.frontier_candidates = np.empty((0, 2), dtype=np.float32)

    def update(
        self,
        hit_points_local: np.ndarray,
        robot_position: np.ndarray,
        robot_yaw: float,
    ) -> dict[str, int]:
        """Fuse local-frame LiDAR hit points using ground-truth robot pose."""
        points = np.asarray(hit_points_local, dtype=np.float32)
        position = np.asarray(robot_position, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected hit points shape (N, 3), got {points.shape}")
        if position.size < 2:
            raise ValueError("robot_position must contain at least x and y")

        valid = np.all(np.isfinite(points), axis=1)
        valid &= (points[:, 2] >= self.min_z) & (points[:, 2] <= self.max_z)
        valid &= np.linalg.norm(points, axis=1) > 1e-6
        points = points[valid]

        origin = self.world_to_grid(float(position[0]), float(position[1]))
        cos_yaw, sin_yaw = float(np.cos(robot_yaw)), float(np.sin(robot_yaw))
        observed_before = int(np.count_nonzero(self.grid != self.UNKNOWN))

        for point in points:
            endpoint_x = float(position[0] + cos_yaw * point[0] - sin_yaw * point[1])
            endpoint_y = float(position[1] + sin_yaw * point[0] + cos_yaw * point[1])
            if not self.contains(endpoint_x, endpoint_y):
                continue
            endpoint = self.world_to_grid(endpoint_x, endpoint_y)
            cells = self._line_cells(origin, endpoint)
            for row, col in cells[:-1]:
                self.grid[row, col] = self.FREE
            end_row, end_col = cells[-1]
            self.grid[end_row, end_col] = self.OCCUPIED

        # The robot's current cell is known free even if the scan has no hits.
        self.grid[origin] = self.FREE
        self._update_frontiers()
        observed_after = int(np.count_nonzero(self.grid != self.UNKNOWN))
        return {
            "newly_observed_cells": observed_after - observed_before,
            "frontier_cells": int(np.count_nonzero(self.frontier_mask)),
            "frontier_candidates": int(len(self.frontier_candidates)),
        }

    def _update_frontiers(self) -> None:
        unknown = self.grid == self.UNKNOWN
        adjacent_unknown = np.zeros_like(unknown)
        adjacent_unknown[1:, :] |= unknown[:-1, :]
        adjacent_unknown[:-1, :] |= unknown[1:, :]
        adjacent_unknown[:, 1:] |= unknown[:, :-1]
        adjacent_unknown[:, :-1] |= unknown[:, 1:]
        raw_mask = (self.grid == self.FREE) & adjacent_unknown

        self.frontier_mask.fill(False)
        visited = np.zeros_like(raw_mask)
        candidates: list[tuple[float, float]] = []
        for start_row, start_col in np.argwhere(raw_mask):
            if visited[start_row, start_col]:
                continue
            queue = deque([(int(start_row), int(start_col))])
            visited[start_row, start_col] = True
            component: list[tuple[int, int]] = []
            while queue:
                row, col = queue.popleft()
                component.append((row, col))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = row + dr, col + dc
                    if (0 <= nr < self.height and 0 <= nc < self.width
                            and raw_mask[nr, nc] and not visited[nr, nc]):
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            if len(component) < self.min_frontier_size:
                continue
            rows, cols = zip(*component)
            self.frontier_mask[rows, cols] = True
            centre = np.mean(np.asarray(component, dtype=np.float32), axis=0)
            representative = min(
                component,
                key=lambda cell: (cell[0] - centre[0]) ** 2 + (cell[1] - centre[1]) ** 2,
            )
            candidates.append(self.grid_to_world(*representative))

        self.frontier_candidates = np.asarray(candidates, dtype=np.float32).reshape(-1, 2)

    def contains(self, x: float, y: float) -> bool:
        return self.xmin <= x < self.xmax and self.ymin <= y < self.ymax

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        col = int(np.clip(np.floor((x - self.xmin) / self.resolution), 0, self.width - 1))
        row = int(np.clip(np.floor((y - self.ymin) / self.resolution), 0, self.height - 1))
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        return (
            self.xmin + (col + 0.5) * self.resolution,
            self.ymin + (row + 0.5) * self.resolution,
        )

    @staticmethod
    def _line_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
        """Integer Bresenham line, including both endpoints."""
        row0, col0 = start
        row1, col1 = end
        cells: list[tuple[int, int]] = []
        dcol, drow = abs(col1 - col0), -abs(row1 - row0)
        step_col = 1 if col0 < col1 else -1
        step_row = 1 if row0 < row1 else -1
        error = dcol + drow
        while True:
            cells.append((row0, col0))
            if row0 == row1 and col0 == col1:
                return cells
            twice_error = 2 * error
            if twice_error >= drow:
                error += drow
                col0 += step_col
            if twice_error <= dcol:
                error += dcol
                row0 += step_row
