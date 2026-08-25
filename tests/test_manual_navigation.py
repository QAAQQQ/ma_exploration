#测试waypoint controller能否成功移动机器人
from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from envs.exploration_env import ExplorationEnv
from envs.waypoint_controller import AStarPurePursuitController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive one MuJoCo robot toward a manual world-coordinate target."
    )
    parser.add_argument("--target-x", type=float, default=None)
    parser.add_argument("--target-y", type=float, default=None)
    parser.add_argument("--auto-min-distance", type=float, default=0.75)
    parser.add_argument("--auto-max-distance", type=float, default=2.0)
    parser.add_argument("--tolerance", type=float, default=0.15)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.002)
    return parser.parse_args()


def set_camera(viewer, env: ExplorationEnv) -> None:
    xmin, xmax, ymin, ymax = env.world_bounds
    viewer.cam.lookat[:] = [(xmin + xmax) / 2, (ymin + ymax) / 2, 0.0]
    viewer.cam.distance = max(xmax - xmin, ymax - ymin) * 1.2
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -85


def add_target_marker(viewer, target_world: np.ndarray) -> None:
    """Add a non-physical magenta sphere to the passive viewer."""
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        raise RuntimeError("Viewer user scene has no free geometry slots")
    marker = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        marker,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([0.18, 0.0, 0.0], dtype=np.float64),
        np.asarray([target_world[0], target_world[1], 0.2], dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray([1.0, 0.0, 1.0, 1.0], dtype=np.float32),
    )
    viewer.user_scn.ngeom += 1


def choose_target_and_path(
    args: argparse.Namespace,
    env: ExplorationEnv,
    controller: AStarPurePursuitController,
    initial_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    occupancy = env.agent_knowledge[0].map

    def plan(target: np.ndarray) -> np.ndarray | None:
        return controller.plan_path(
            agent_id=0,
            occupancy_grid=occupancy.grid,
            robot_position=initial_position,
            waypoint_world=target,
            xmin=occupancy.xmin,
            ymin=occupancy.ymin,
            resolution=occupancy.resolution,
        )

    if (args.target_x is None) != (args.target_y is None):
        raise ValueError("--target-x and --target-y must be provided together")
    if args.target_x is not None:
        target = np.asarray([args.target_x, args.target_y], dtype=np.float64)
        return target, plan(target)

    free_cells = np.argwhere(occupancy.grid == occupancy.FREE)
    candidates = np.asarray(
        [occupancy.grid_to_world(int(row), int(col)) for row, col in free_cells],
        dtype=np.float64,
    )
    if len(candidates) == 0:
        return None, None
    distances = np.linalg.norm(candidates - initial_position, axis=1)
    valid = (
        (distances >= args.auto_min_distance)
        & (distances <= args.auto_max_distance)
    )
    # Prefer a farther target so movement is easy to see in the viewer.
    for candidate_id in np.argsort(distances[valid])[::-1]:
        target = candidates[valid][candidate_id]
        path = plan(target)
        if path is not None:
            return target, path
    return None, None


def main() -> None:
    args = parse_args()
    if args.tolerance <= 0:
        raise ValueError("--tolerance must be > 0")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if not 0 < args.auto_min_distance <= args.auto_max_distance:
        raise ValueError("auto target distance bounds are invalid")

    controller = AStarPurePursuitController(
        max_forward_speed=0.5,
        max_yaw_rate=1.0,
        waypoint_tolerance=args.tolerance,
        forward_gain=1.0,
        yaw_gain=2.0,
        lookahead_distance=0.5,
    )
    env = ExplorationEnv(n_agents=1, frame_skip=10, controller=controller)
    env.reset(seed=args.seed)

    initial_position = env._get_robot_positions()[0, :2].astype(np.float64)
    reached = False
    steps_executed = 0

    print("Initial robot position:", initial_position.tolist())
    target_world, path = choose_target_and_path(
        args, env, controller, initial_position
    )
    if target_world is None or path is None:
        print("Final robot position:", initial_position.tolist())
        print("Final distance: unavailable")
        print("Steps executed:", steps_executed)
        print("FAILED: no reachable known-free target was found")
        return
    print("Target position:", target_world.tolist())
    print("Target selection:", "manual" if args.target_x is not None else "automatic")
    print("A* path points:", len(path))

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        set_camera(viewer, env)
        with viewer.lock():
            add_target_marker(viewer, target_world)
        viewer.sync()
        for step in range(args.max_steps):
            if not viewer.is_running():
                break

            position = env._get_robot_positions()[0]
            yaw = float(env._get_robot_yaws()[0])
            command, controller_reached = controller.compute_command_for_agent(
                agent_id=0,
                robot_position=position,
                robot_yaw=yaw,
                waypoint_world=target_world,
            )
            distance = float(np.linalg.norm(position[:2] - target_world))
            if controller_reached or distance <= args.tolerance:
                reached = True
                break

            env._apply_low_level_commands(command.reshape(1, 2))
            mujoco.mj_step(env.model, env.data)
            env.physics_step_count += 1
            steps_executed = step + 1
            viewer.sync()
            if args.sleep > 0:
                time.sleep(args.sleep)

    final_position = env._get_robot_positions()[0, :2].astype(np.float64)
    final_distance = float(np.linalg.norm(final_position - target_world))
    reached = reached or final_distance <= args.tolerance

    print("Final robot position:", final_position.tolist())
    print("Final distance:", final_distance)
    print("Steps executed:", steps_executed)
    print("SUCCESS" if reached else "FAILED/TIMEOUT")


if __name__ == "__main__":
    main()
