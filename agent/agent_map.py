from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class AgentMap:
    """One agent's LiDAR-built knowledge; ``grid`` is its only map state."""

    UNKNOWN = np.int8(-1)
    FREE = np.int8(0)
    OCCUPIED = np.int8(1)
    _NEIGHBOURS_8 = tuple(
        (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc
    )

    def __init__(
        self,
        world_bounds: tuple[float, float, float, float],
        resolution: float = 0.25,
        min_z: float = -0.2,
        max_z: float = 0.2,
        min_frontier_size: int = 2,
        patch_size_m: float = 5.0,
    ) -> None:
        xmin, xmax, ymin, ymax = world_bounds
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"Invalid world_bounds: {world_bounds}")
        if resolution <= 0 or min_frontier_size <= 0 or patch_size_m <= 0:
            raise ValueError("resolution, min_frontier_size and patch_size_m must be > 0")
        if min_z > max_z:
            raise ValueError("min_z must be <= max_z")
        self.xmin, self.xmax = float(xmin), float(xmax)
        self.ymin, self.ymax = float(ymin), float(ymax)
        self.resolution = float(resolution)
        self.min_z, self.max_z = float(min_z), float(max_z)
        self.min_frontier_size = int(min_frontier_size)
        self.width = int(np.ceil((self.xmax - self.xmin) / self.resolution))
        self.height = int(np.ceil((self.ymax - self.ymin) / self.resolution))
        patch_cells = int(np.ceil(patch_size_m / self.resolution))
        self.patch_cells = patch_cells if patch_cells % 2 else patch_cells + 1
        self.patch_radius_cells = self.patch_cells // 2
        self.grid = np.full((self.height, self.width), self.UNKNOWN, dtype=np.int8)

    @property
    def observation_dim(self) -> int:
        return self.patch_cells * self.patch_cells

    @property
    def frontier_candidates(self) -> np.ndarray:
        return self.compute_frontier_candidates()

    def reset(self) -> None:
        self.grid.fill(self.UNKNOWN)

    def update_from_lidar(
        self,
        hit_points_local: np.ndarray,
        robot_position: np.ndarray,
        robot_yaw: float,
    ) -> dict[str, int]:
        """Transform LiDAR hits, mark ray cells FREE, and endpoints OCCUPIED."""
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
            cells = self._line_cells(origin, self.world_to_grid(endpoint_x, endpoint_y))
            for row, col in cells[:-1]:
                self.grid[row, col] = self.FREE
            self.grid[cells[-1]] = self.OCCUPIED
        self.grid[origin] = self.FREE
        frontier_mask = self.compute_frontier_mask()
        candidates = self.compute_frontier_candidates(frontier_mask)
        return {
            "newly_observed_cells": int(np.count_nonzero(self.grid != self.UNKNOWN))
            - observed_before,
            "frontier_cells": int(np.count_nonzero(frontier_mask)),
            "frontier_candidates": int(len(candidates)),
        }

    def compute_frontier_mask(self) -> np.ndarray:
        """Compute FREE cells adjacent to UNKNOWN cells, without caching the mask."""
        unknown = self.grid == self.UNKNOWN
        adjacent_unknown = np.zeros_like(unknown)
        padded = np.pad(unknown, 1, constant_values=False)
        for dr, dc in self._NEIGHBOURS_8:
            adjacent_unknown |= padded[
                1 + dr:1 + dr + self.height,
                1 + dc:1 + dc + self.width,
            ]
        return (self.grid == self.FREE) & adjacent_unknown

    def compute_frontier_candidates(
        self, frontier_mask: np.ndarray | None = None
    ) -> np.ndarray:
        """8-neighbour cluster frontiers and choose centroid-nearest valid cells."""
        mask = self.compute_frontier_mask() if frontier_mask is None else np.asarray(
            frontier_mask, dtype=bool
        )
        if mask.shape != self.grid.shape:
            raise ValueError(f"frontier mask shape {mask.shape} != grid shape {self.grid.shape}")
        seen = np.zeros_like(mask)
        candidates: list[tuple[float, float]] = []
        for start_row, start_col in np.argwhere(mask):
            if seen[start_row, start_col]:
                continue
            queue = deque([(int(start_row), int(start_col))])
            seen[start_row, start_col] = True
            component: list[tuple[int, int]] = []
            while queue:
                row, col = queue.popleft()
                component.append((row, col))
                for dr, dc in self._NEIGHBOURS_8:
                    nr, nc = row + dr, col + dc
                    if (0 <= nr < self.height and 0 <= nc < self.width
                            and mask[nr, nc] and not seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            if len(component) < self.min_frontier_size:
                continue
            centre = np.mean(np.asarray(component, dtype=np.float32), axis=0)
            representative = min(
                component,
                key=lambda cell: (cell[0] - centre[0]) ** 2 + (cell[1] - centre[1]) ** 2,
            )
            candidates.append(self.grid_to_world(*representative))
        return np.asarray(candidates, dtype=np.float32).reshape(-1, 2)

    def get_local_patch(self, robot_position: np.ndarray) -> np.ndarray:
        position = np.asarray(robot_position, dtype=np.float32)
        if position.size < 2:
            raise ValueError("robot_position must contain at least x and y")
        row, col = self.world_to_grid(float(position[0]), float(position[1]))
        patch = np.full((self.patch_cells, self.patch_cells), self.UNKNOWN, dtype=np.int8)
        r0, r1 = max(0, row - self.patch_radius_cells), min(
            self.height, row + self.patch_radius_cells + 1
        )
        c0, c1 = max(0, col - self.patch_radius_cells), min(
            self.width, col + self.patch_radius_cells + 1
        )
        dr, dc = r0 - row + self.patch_radius_cells, c0 - col + self.patch_radius_cells
        patch[dr:dr + r1 - r0, dc:dc + c1 - c0] = self.grid[r0:r1, c0:c1]
        return patch

    def get_map_observation(self, robot_position: np.ndarray) -> np.ndarray:
        return self.get_local_patch(robot_position).reshape(-1).astype(np.float32)

    def export_region(
        self, bounds: tuple[float, float, float, float] | None = None
    ) -> dict[str, Any]:
        """Export a copy for communication; bounds use (xmin, xmax, ymin, ymax)."""
        if bounds is None:
            row0, row1, col0, col1 = 0, self.height, 0, self.width
        else:
            xmin, xmax, ymin, ymax = bounds
            if xmax <= xmin or ymax <= ymin:
                raise ValueError(f"Invalid export bounds: {bounds}")
            row0, col0 = self.world_to_grid(xmin, ymin)
            row1, col1 = self.world_to_grid(
                np.nextafter(xmax, -np.inf), np.nextafter(ymax, -np.inf)
            )
            row1, col1 = row1 + 1, col1 + 1
        region_bounds = (
            self.xmin + col0 * self.resolution,
            self.xmin + col1 * self.resolution,
            self.ymin + row0 * self.resolution,
            self.ymin + row1 * self.resolution,
        )
        return {
            "grid": self.grid[row0:row1, col0:col1].copy(),
            "bounds": region_bounds,
            "resolution": self.resolution,
        }

    def merge_grid(
        self,
        incoming_grid: np.ndarray,
        *,
        bounds: tuple[float, float, float, float] | None = None,
        transform: Any | None = None,
    ) -> int:
        """Merge an aligned grid; known replaces UNKNOWN and OCCUPIED wins conflicts."""
        if transform is not None:
            raise NotImplementedError("Transformed map fusion is not implemented yet")
        incoming = np.asarray(incoming_grid, dtype=np.int8)
        row0, col0 = (0, 0) if bounds is None else self.world_to_grid(bounds[0], bounds[2])
        row1, col1 = row0 + incoming.shape[0], col0 + incoming.shape[1]
        if incoming.ndim != 2 or row1 > self.height or col1 > self.width:
            raise ValueError("Incoming grid shape/bounds do not align with this map")
        if bounds is None and incoming.shape != self.grid.shape:
            raise ValueError("A full incoming grid must match this map's shape")
        if not np.all(np.isin(incoming, (self.UNKNOWN, self.FREE, self.OCCUPIED))):
            raise ValueError("Incoming grid contains invalid cell values")
        target = self.grid[row0:row1, col0:col1]
        before = target.copy()
        fill = (target == self.UNKNOWN) & (incoming != self.UNKNOWN)
        target[fill] = incoming[fill]
        target[incoming == self.OCCUPIED] = self.OCCUPIED
        return int(np.count_nonzero(target != before))

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
