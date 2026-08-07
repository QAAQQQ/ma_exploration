from __future__ import annotations
from abc import ABC, abstractmethod
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


class FYController(BaseController):
    """
    Forward-yaw waypoint controller.

    Converts a world-frame waypoint into:
        command = [forward_speed, yaw_rate]

    This controller computes commands only. It does not modify MuJoCo state.
    """

    def __init__(
        self,
        max_forward_speed: float = 0.5,
        max_yaw_rate: float = 1.0,
        waypoint_tolerance: float = 0.15,
        forward_gain: float = 1.0,
        yaw_gain: float = 2.0,
        rotate_in_place_threshold: float = np.deg2rad(75.0),
    ) -> None:
        self.max_forward_speed = float(max_forward_speed)
        self.max_yaw_rate = float(max_yaw_rate)
        self.waypoint_tolerance = float(waypoint_tolerance)
        self.forward_gain = float(forward_gain)
        self.yaw_gain = float(yaw_gain)
        self.rotate_in_place_threshold = float(rotate_in_place_threshold)

    def compute_command(
        self,
        robot_position: np.ndarray,
        robot_yaw: float,
        waypoint_world: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        robot_position = np.asarray(robot_position, dtype=np.float64)
        waypoint_world = np.asarray(waypoint_world, dtype=np.float64)

        if robot_position.shape[0] < 2 or waypoint_world.shape[0] < 2:
            raise ValueError("robot_position and waypoint_world must contain x and y.")

        dx = float(waypoint_world[0] - robot_position[0])
        dy = float(waypoint_world[1] - robot_position[1])
        distance = float(np.hypot(dx, dy))

        if distance <= self.waypoint_tolerance:
            return np.zeros(2, dtype=np.float64), True

        desired_yaw = float(np.arctan2(dy, dx))
        yaw_error = wrap_to_pi(desired_yaw - float(robot_yaw))

        yaw_rate = float(
            np.clip(
                self.yaw_gain * yaw_error,
                -self.max_yaw_rate,
                self.max_yaw_rate,
            )
        )

        if abs(yaw_error) >= self.rotate_in_place_threshold:
            forward_speed = 0.0
        else:
            heading_alignment = max(0.0, float(np.cos(yaw_error)))
            forward_speed = float(
                np.clip(
                    self.forward_gain * distance * heading_alignment,
                    0.0,
                    self.max_forward_speed,
                )
            )

        return np.array([forward_speed, yaw_rate], dtype=np.float64), False