# process raw point cloud data from mujoco lidar

from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np


class BaseLidarProcessor(ABC):
    """
    Convert raw/local LiDAR point cloud into a fixed-format
    observation representation for downstream algorithms.
    """

    @abstractmethod
    def process(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

class Lidar2DProcessor:
    """
    Convert a local-frame 3D LiDAR point cloud into a fixed-size
    planar 2D range observation.

    Input:
        points:
            shape (N, 3), each point is [x, y, z]
            in the LiDAR local frame.

    Output:
        scan:
            shape (n_bins,)
            normalized planar range observation in [0, 1].

            0.0 = obstacle very close
            1.0 = no obstacle within max_range
    """

    def __init__(
        self,
        min_z: float = -0.2,
        max_z: float = 0.2,
        n_bins: int = 360,
        max_range: float = 5.0,
    ) -> None:
        if min_z > max_z:
            raise ValueError("min_z must be <= max_z")

        if n_bins <= 0:
            raise ValueError("n_bins must be > 0")

        if max_range <= 0:
            raise ValueError("max_range must be > 0")

        self.min_z = float(min_z)
        self.max_z = float(max_z)
        self.n_bins = int(n_bins)
        self.max_range = float(max_range)

    def process(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        """
        Process one 3D point cloud into a fixed-size 2D scan.
        """

        points = np.asarray(
            points,
            dtype=np.float32,
        )

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"Expected point cloud shape (N, 3), got {points.shape}"
            )

        # Default:
        # no obstacle detected in any angular bin
        scan = np.full(
            self.n_bins,
            self.max_range,
            dtype=np.float32,
        )

        # --------------------------------------------------
        # 1. Remove invalid points
        # --------------------------------------------------
        valid_mask = np.all(
            np.isfinite(points),
            axis=1,
        )
        points = points[valid_mask]

        if len(points) == 0:
            return scan / self.max_range

        # --------------------------------------------------
        # 2. Keep only points close to the planar slice
        # --------------------------------------------------
        z_mask = (
            (points[:, 2] >= self.min_z)
            & (points[:, 2] <= self.max_z)
        )
        points = points[z_mask]

        if len(points) == 0:
            return scan / self.max_range

        # --------------------------------------------------
        # 3. Project XYZ -> XY
        # --------------------------------------------------
        x = points[:, 0]
        y = points[:, 1]

        ranges = np.hypot(x, y)

        # --------------------------------------------------
        # 4. Remove points outside sensor range
        # --------------------------------------------------
        range_mask = (
            (ranges > 0.0)
            & (ranges <= self.max_range)
        )

        x = x[range_mask]
        y = y[range_mask]
        ranges = ranges[range_mask]

        if len(ranges) == 0:
            return scan / self.max_range

        # --------------------------------------------------
        # 5. Convert XY position to angular direction
        #
        # atan2 gives angle in [-pi, pi]
        # --------------------------------------------------
        angles = np.arctan2(y, x)

        # Map [-pi, pi] -> [0, n_bins)
        bin_indices = (
            (angles + np.pi)
            / (2.0 * np.pi)
            * self.n_bins
        ).astype(np.int32)

        # Handle the numerical edge case angle == pi
        bin_indices = np.clip(
            bin_indices,
            0,
            self.n_bins - 1,
        )

        # --------------------------------------------------
        # 6. Keep nearest obstacle in each angular bin
        # --------------------------------------------------
        np.minimum.at(
            scan,
            bin_indices,
            ranges,
        )

        # --------------------------------------------------
        # 7. Normalize to [0, 1]
        # --------------------------------------------------
        scan = scan / self.max_range

        return scan.astype(np.float32)