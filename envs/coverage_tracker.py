from __future__ import annotations
import numpy as np


class CoverageTracker:
    """
    在连续世界坐标上维护独立的二维 coverage grid。

    这张 grid 不参与 MuJoCo 场景生成，只用于：
    - exploration reward
    - coverage ratio
    - episode termination
    - evaluation logging
    """

    def __init__(
        self,
        world_bounds: tuple[float, float, float, float],
        resolution: float = 0.25,
        exploration_radius: float = 0.3,
    ) -> None:
        """
        Args:
            world_bounds:
                (xmin, xmax, ymin, ymax)，单位为米。Mujoco的单位不知道是多少但是就先这样
            resolution:
                每个 coverage cell 对应的世界尺寸，单位为米。
            exploration_radius:
                机器人位置周围多少米算作已探索。这个需要和机器人的ldiar range有一个对应关系，但也可能没有 NOTE
        """
        xmin, xmax, ymin, ymax = world_bounds

        # sanity check
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(
                f"Invalid world bounds: {world_bounds}"
            )
        if resolution <= 0:
            raise ValueError(
                "resolution must be greater than zero"
            )
        if exploration_radius < 0:
            raise ValueError(
                "exploration_radius cannot be negative"
            )

        self.world_bounds = world_bounds
        self.resolution = float(resolution)
        self.exploration_radius = float(exploration_radius)

        self.width = int(np.ceil((xmax - xmin) / self.resolution))
        self.height = int(np.ceil((ymax - ymin) / self.resolution))

        self.explored_map = np.zeros((self.height, self.width),dtype=bool)

        # 当前先认为整个 world bounds 都属于候选探索区域。
        # 以后可以由 MuJoCo geometry 或 SLAM map 提供。TODO
        self.traversable_mask = np.ones(
            (self.height, self.width),
            dtype=bool,
        )

    def reset(self,traversable_mask: np.ndarray | None = None)-> None:
        self.explored_map.fill(False)

        if traversable_mask is None:
            self.traversable_mask.fill(True)
            return

        traversable_mask = np.asarray(
            traversable_mask,
            dtype=bool,
        )

        if traversable_mask.shape != self.explored_map.shape:
            raise ValueError(
                "traversable_mask shape mismatch: "
                f"expected {self.explored_map.shape}, "
                f"received {traversable_mask.shape}"
            )

        self.traversable_mask = traversable_mask.copy()

    def world_to_grid(self,x: float,y: float,)-> tuple[int, int] | None:
        xmin, xmax, ymin, ymax = self.world_bounds

        if not (xmin <= x < xmax and ymin <= y < ymax):
            return None

        col = int(np.floor((x - xmin) / self.resolution))
        row = int(np.floor((y - ymin) / self.resolution))

        if not (0 <= row < self.height and 0 <= col < self.width):
            return None

        return row, col

    def update(self, positions: np.ndarray) -> dict[str, int | float]:
        """
        根据多个机器人的世界坐标更新 coverage 
        TODO 这个大概可能还会需要改，但是呢，位置更新+lidar圈圈，好像也没错，但这个太理想化了，完全没有考虑lidar扫出来可能有问题?
        Args:
            positions:
                shape 为 (n_agents, 2) 或 (n_agents, 3)。
        Returns:
            本次新探索的 cell 数量。
        """
        positions = np.asarray(
            positions,
            dtype=np.float32,
        )

        if positions.ndim != 2:
            raise ValueError(
                "positions must be a 2D array"
            )

        if positions.shape[1] < 2:
            raise ValueError(
                "positions must contain at least x and y"
            )

        explored_before = int(
            np.logical_and(
                self.explored_map,
                self.traversable_mask,
            ).sum()
        )

        radius_cells = int(
            np.ceil(
                self.exploration_radius
                / self.resolution
            )
        )

        for position in positions:
            x = float(position[0])
            y = float(position[1])

            grid_position = self.world_to_grid(x, y)

            if grid_position is None:
                continue

            center_row, center_col = grid_position

            for row_offset in range(
                -radius_cells,
                radius_cells + 1,
            ):
                for col_offset in range(
                    -radius_cells,
                    radius_cells + 1,
                ):
                    row = center_row + row_offset
                    col = center_col + col_offset

                    if not (
                        0 <= row < self.height
                        and 0 <= col < self.width
                    ):
                        continue

                    cell_center_x, cell_center_y = (
                        self.grid_to_world(row, col)
                    )

                    distance = np.hypot(
                        cell_center_x - x,
                        cell_center_y - y,
                    )

                    if distance > self.exploration_radius:
                        continue

                    if not self.traversable_mask[row, col]:
                        continue

                    self.explored_map[row, col] = True

        explored_after = int(
            np.logical_and(
                self.explored_map,
                self.traversable_mask,
            ).sum()
        )

        newly_explored = explored_after - explored_before

        total_traversable = int(self.traversable_mask.sum())

        if total_traversable > 0:
            coverage_ratio = explored_after / total_traversable
        else:
            coverage_ratio = 0.0

        return {
            "newly_explored": newly_explored,
            "explored_cells": explored_after,
            "total_traversable_cells": total_traversable,
            "coverage_ratio": float(coverage_ratio),
        }

        # return explored_after - explored_before # 简单版本

    def grid_to_world(
        self,
        row: int,
        col: int,
    ) -> tuple[float, float]:
        xmin, _, ymin, _ = self.world_bounds

        x = xmin + (col + 0.5) * self.resolution
        y = ymin + (row + 0.5) * self.resolution

        return x, y

    @property
    def explored_cells(self) -> int:
        return int(
            np.logical_and(
                self.explored_map,
                self.traversable_mask,
            ).sum()
        )

    @property
    def total_traversable_cells(self) -> int:
        return int(self.traversable_mask.sum())

    @property
    def coverage_ratio(self) -> float:
        total = self.total_traversable_cells
        if total == 0:
            return 0.0
        return self.explored_cells / total