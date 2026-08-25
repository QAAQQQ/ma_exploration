from __future__ import annotations
from abc import ABC, abstractmethod
import heapq
import numpy as np


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class BaseController(ABC):
    """Common interface for controllers that drive a robot towards a waypoint."""

    @abstractmethod
    def compute_command(
        self,
        robot_position: np.ndarray,
        robot_yaw: float,
        waypoint_world: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Return a robot-specific low-level command and whether the goal is reached."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset controller state. Stateless controllers need no special handling."""
        return None

    def compute_command_for_agent(
        self,
        agent_id: int,
        robot_position: np.ndarray,
        robot_yaw: float,
        waypoint_world: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Agent-aware adapter; legacy stateless controllers remain compatible."""
        del agent_id
        return self.compute_command(robot_position, robot_yaw, waypoint_world)


class AStarPurePursuitController(BaseController):
    """Plan on a known occupancy grid with A* and track it using pure pursuit."""

    def __init__(
        self,
        *,
        max_forward_speed: float = 0.5,
        max_yaw_rate: float = 1.0,
        waypoint_tolerance: float = 0.15,
        forward_gain: float = 1.0,
        yaw_gain: float = 2.0,
        rotate_in_place_threshold: float = np.deg2rad(75.0),
        slow_forward_threshold: float = np.deg2rad(25.0),
        slow_forward_speed: float = 0.15,
        lookahead_distance: float = 0.5,
        unknown_is_obstacle: bool = True,
    ) -> None:
        self.max_forward_speed = float(max_forward_speed)
        self.max_yaw_rate = float(max_yaw_rate)
        self.waypoint_tolerance = float(waypoint_tolerance)
        self.forward_gain = float(forward_gain)
        self.yaw_gain = float(yaw_gain)
        self.rotate_in_place_threshold = float(rotate_in_place_threshold)
        self.slow_forward_threshold = float(slow_forward_threshold)
        self.slow_forward_speed = float(slow_forward_speed)
        if not 0.0 <= self.slow_forward_threshold < self.rotate_in_place_threshold:
            raise ValueError(
                "slow_forward_threshold must be non-negative and smaller than "
                "rotate_in_place_threshold"
            )
        if lookahead_distance <= 0:
            raise ValueError("lookahead_distance must be > 0")
        self.lookahead_distance = float(lookahead_distance)
        self.unknown_is_obstacle = bool(unknown_is_obstacle)
        self.paths: dict[int, np.ndarray] = {}
        self.path_indices: dict[int, int] = {}

    def reset(self) -> None:
        self.paths.clear()
        self.path_indices.clear()

    def compute_command(
        self,
        robot_position: np.ndarray,
        robot_yaw: float,
        waypoint_world: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Single-agent compatibility entry point using the path cached for agent 0."""
        return self.compute_command_for_agent(
            0, robot_position, robot_yaw, waypoint_world
        )

    def plan_path(
        self,
        agent_id: int,
        occupancy_grid: np.ndarray,
        robot_position: np.ndarray,
        waypoint_world: np.ndarray,
        *,
        xmin: float,
        ymin: float,
        resolution: float,
    ) -> np.ndarray | None:
        """Create and cache a collision-free world-coordinate path."""
        grid = np.asarray(occupancy_grid)
        if grid.ndim != 2:
            raise ValueError("occupancy_grid must be two-dimensional")
        if resolution <= 0:
            raise ValueError("resolution must be > 0")

        def world_to_grid(position: np.ndarray) -> tuple[int, int]:
            col = int(np.floor((float(position[0]) - xmin) / resolution))
            row = int(np.floor((float(position[1]) - ymin) / resolution))
            return row, col

        start = world_to_grid(np.asarray(robot_position))
        goal = world_to_grid(np.asarray(waypoint_world))
        height, width = grid.shape
        if not all(
            0 <= row < height and 0 <= col < width
            for row, col in (start, goal)
        ):
            self.paths.pop(agent_id, None)
            self.path_indices.pop(agent_id, None)
            return None

        traversable = grid == 0 if self.unknown_is_obstacle else grid != 1
        traversable = traversable.copy()
        traversable[start] = True
        traversable[goal] = True
        grid_path = self._astar(traversable, start, goal)
        if grid_path is None:
            self.paths.pop(agent_id, None)
            self.path_indices.pop(agent_id, None)
            return None
        grid_path = self._smooth_grid_path(grid_path, traversable)
        path = np.asarray(
            [
                [
                    xmin + (col + 0.5) * resolution,
                    ymin + (row + 0.5) * resolution,
                ]
                for row, col in grid_path
            ],
            dtype=np.float64,
        )
        path[0] = np.asarray(robot_position, dtype=np.float64)[:2]
        path[-1] = np.asarray(waypoint_world, dtype=np.float64)[:2]
        path = self._densify_world_path(
            path, spacing=min(resolution * 0.5, self.lookahead_distance * 0.5)
        )
        self.paths[agent_id] = path
        self.path_indices[agent_id] = 0
        return path.copy()

    def compute_command_for_agent(
        self,
        agent_id: int,
        robot_position: np.ndarray,
        robot_yaw: float,
        waypoint_world: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        position = np.asarray(robot_position, dtype=np.float64)[:2]
        goal = np.asarray(waypoint_world, dtype=np.float64)[:2]
        if float(np.linalg.norm(goal - position)) <= self.waypoint_tolerance:
            return np.zeros(2, dtype=np.float64), True

        path = self.paths.get(agent_id)
        if path is None or len(path) == 0:
            return np.zeros(2, dtype=np.float64), False

        start_index = self.path_indices.get(agent_id, 0)
        remaining = path[start_index:]
        closest_offset = int(np.argmin(np.linalg.norm(remaining - position, axis=1)))
        closest_index = start_index + closest_offset
        target_index = closest_index
        while (
            target_index < len(path) - 1
            and np.linalg.norm(path[target_index] - position) < self.lookahead_distance
        ):
            target_index += 1
        self.path_indices[agent_id] = closest_index
        pursuit_target = path[target_index]

        delta = pursuit_target - position
        target_distance = float(np.linalg.norm(delta))
        desired_yaw = float(np.arctan2(delta[1], delta[0]))
        yaw_error = wrap_to_pi(desired_yaw - float(robot_yaw))

        if abs(yaw_error) >= self.rotate_in_place_threshold:
            forward_speed = 0.0
            yaw_rate = float(
                np.clip(self.yaw_gain * yaw_error, -self.max_yaw_rate, self.max_yaw_rate)
            )
        else:
            if abs(yaw_error) >= self.slow_forward_threshold:
                forward_speed = min(self.slow_forward_speed, self.max_forward_speed)
            else:
                forward_speed = float(
                    np.clip(
                        self.forward_gain * np.linalg.norm(goal - position),
                        0.0,
                        self.max_forward_speed,
                    )
                )
            local_y = -np.sin(robot_yaw) * delta[0] + np.cos(robot_yaw) * delta[1]
            curvature = 2.0 * local_y / max(target_distance ** 2, 1e-6)
            yaw_rate = float(
                np.clip(forward_speed * curvature, -self.max_yaw_rate, self.max_yaw_rate)
            )
        return np.asarray([forward_speed, yaw_rate], dtype=np.float64), False

    @staticmethod
    def _astar(
        traversable: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]] | None:
        neighbours = (
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, np.sqrt(2.0)), (-1, 1, np.sqrt(2.0)),
            (1, -1, np.sqrt(2.0)), (1, 1, np.sqrt(2.0)),
        )
        queue = [(0.0, 0.0, start)]
        costs = {start: 0.0}
        parents: dict[tuple[int, int], tuple[int, int]] = {}
        height, width = traversable.shape
        while queue:
            _, cost, current = heapq.heappop(queue)
            if cost > costs.get(current, np.inf):
                continue
            if current == goal:
                path = [current]
                while path[-1] != start:
                    path.append(parents[path[-1]])
                path.reverse()
                return path
            row, col = current
            for dr, dc, move_cost in neighbours:
                nr, nc = row + dr, col + dc
                if not (0 <= nr < height and 0 <= nc < width and traversable[nr, nc]):
                    continue
                if dr != 0 and dc != 0:
                    if not (traversable[row + dr, col] and traversable[row, col + dc]):
                        continue
                next_cell = (nr, nc)
                next_cost = cost + move_cost
                if next_cost >= costs.get(next_cell, np.inf):
                    continue
                costs[next_cell] = next_cost
                parents[next_cell] = current
                heuristic = float(np.hypot(goal[0] - nr, goal[1] - nc))
                heapq.heappush(queue, (next_cost + heuristic, next_cost, next_cell))
        return None

    @classmethod
    def _smooth_grid_path(
        cls,
        path: list[tuple[int, int]],
        traversable: np.ndarray,
    ) -> list[tuple[int, int]]:
        if len(path) <= 2:
            return path
        smoothed = [path[0]]
        anchor = 0
        while anchor < len(path) - 1:
            furthest = anchor + 1
            for candidate in range(anchor + 2, len(path)):
                if cls._line_is_free(path[anchor], path[candidate], traversable):
                    furthest = candidate
                else:
                    break
            smoothed.append(path[furthest])
            anchor = furthest
        return smoothed

    @staticmethod
    def _densify_world_path(path: np.ndarray, spacing: float) -> np.ndarray:
        dense_points = [path[0]]
        for start, end in zip(path[:-1], path[1:]):
            distance = float(np.linalg.norm(end - start))
            segments = max(1, int(np.ceil(distance / spacing)))
            for fraction in np.linspace(0.0, 1.0, segments + 1)[1:]:
                dense_points.append(start + fraction * (end - start))
        return np.asarray(dense_points, dtype=np.float64)

    @staticmethod
    def _line_is_free(
        start: tuple[int, int],
        end: tuple[int, int],
        traversable: np.ndarray,
    ) -> bool:
        row0, col0 = start
        row1, col1 = end
        steps = max(abs(row1 - row0), abs(col1 - col0))
        if steps == 0:
            return bool(traversable[row0, col0])
        rows = np.rint(np.linspace(row0, row1, steps + 1)).astype(int)
        cols = np.rint(np.linspace(col0, col1, steps + 1)).astype(int)
        return bool(np.all(traversable[rows, cols]))
