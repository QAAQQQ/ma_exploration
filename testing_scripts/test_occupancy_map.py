import numpy as np

from agent.occupancy_map import OccupancyMap


def test_lidar_ray_marks_free_and_endpoint_occupied() -> None:
    occupancy = OccupancyMap(
        world_bounds=(0.0, 6.0, 0.0, 6.0),
        resolution=1.0,
        min_frontier_size=1,
    )
    occupancy.update(
        hit_points_local=np.asarray([[3.0, 0.0, 0.0]], dtype=np.float32),
        robot_position=np.asarray([1.5, 1.5, 0.0], dtype=np.float32),
        robot_yaw=0.0,
    )

    assert occupancy.grid[1, 1] == OccupancyMap.FREE
    assert occupancy.grid[1, 2] == OccupancyMap.FREE
    assert occupancy.grid[1, 3] == OccupancyMap.FREE
    assert occupancy.grid[1, 4] == OccupancyMap.OCCUPIED
    assert occupancy.grid[0, 0] == OccupancyMap.UNKNOWN


def test_robot_yaw_rotates_local_hit_and_generates_frontier_candidate() -> None:
    occupancy = OccupancyMap(
        world_bounds=(0.0, 6.0, 0.0, 6.0),
        resolution=1.0,
        min_frontier_size=1,
    )
    diagnostics = occupancy.update(
        hit_points_local=np.asarray([[2.0, 0.0, 0.0]], dtype=np.float32),
        robot_position=np.asarray([2.5, 1.5], dtype=np.float32),
        robot_yaw=np.pi / 2.0,
    )

    assert occupancy.grid[3, 2] == OccupancyMap.OCCUPIED
    assert diagnostics["frontier_cells"] > 0
    assert occupancy.frontier_candidates.shape[1] == 2


def test_reset_clears_map_and_frontiers() -> None:
    occupancy = OccupancyMap(
        world_bounds=(0.0, 4.0, 0.0, 4.0),
        resolution=1.0,
        min_frontier_size=1,
    )
    occupancy.update(np.asarray([[1.0, 0.0, 0.0]]), np.asarray([1.5, 1.5]), 0.0)
    occupancy.reset()

    assert np.all(occupancy.grid == OccupancyMap.UNKNOWN)
    assert not np.any(occupancy.frontier_mask)
    assert occupancy.frontier_candidates.shape == (0, 2)
