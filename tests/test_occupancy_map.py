import numpy as np

from agent.agent_map import AgentMap


def test_lidar_ray_marks_free_and_endpoint_occupied() -> None:
    occupancy = AgentMap(
        world_bounds=(0.0, 6.0, 0.0, 6.0),
        resolution=1.0,
        min_frontier_size=1,
    )
    occupancy.update_from_lidar(
        hit_points_local=np.asarray([[3.0, 0.0, 0.0]], dtype=np.float32),
        robot_position=np.asarray([1.5, 1.5, 0.0], dtype=np.float32),
        robot_yaw=0.0,
    )

    assert occupancy.grid[1, 1] == AgentMap.FREE
    assert occupancy.grid[1, 2] == AgentMap.FREE
    assert occupancy.grid[1, 3] == AgentMap.FREE
    assert occupancy.grid[1, 4] == AgentMap.OCCUPIED
    assert occupancy.grid[0, 0] == AgentMap.UNKNOWN


def test_robot_yaw_rotates_local_hit_and_generates_frontier_candidate() -> None:
    occupancy = AgentMap(
        world_bounds=(0.0, 6.0, 0.0, 6.0),
        resolution=1.0,
        min_frontier_size=1,
    )
    diagnostics = occupancy.update_from_lidar(
        hit_points_local=np.asarray([[2.0, 0.0, 0.0]], dtype=np.float32),
        robot_position=np.asarray([2.5, 1.5], dtype=np.float32),
        robot_yaw=np.pi / 2.0,
    )

    assert occupancy.grid[3, 2] == AgentMap.OCCUPIED
    assert diagnostics["frontier_cells"] > 0
    assert occupancy.frontier_candidates.shape[1] == 2


def test_reset_clears_map_and_frontiers() -> None:
    occupancy = AgentMap(
        world_bounds=(0.0, 4.0, 0.0, 4.0),
        resolution=1.0,
        min_frontier_size=1,
    )
    occupancy.update_from_lidar(
        np.asarray([[1.0, 0.0, 0.0]]), np.asarray([1.5, 1.5]), 0.0
    )
    occupancy.reset()

    assert np.all(occupancy.grid == AgentMap.UNKNOWN)
    assert not np.any(occupancy.compute_frontier_mask())
    assert occupancy.frontier_candidates.shape == (0, 2)


def test_frontiers_use_eight_neighbour_clustering() -> None:
    agent_map = AgentMap(
        world_bounds=(0.0, 5.0, 0.0, 5.0),
        resolution=1.0,
        min_frontier_size=1,
    )
    mask = np.zeros_like(agent_map.grid, dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True

    candidates = agent_map.compute_frontier_candidates(mask)

    assert candidates.shape == (1, 2)
