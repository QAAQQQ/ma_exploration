# MuJoCo -> multi-agent RL environment wrapper
# obs, shared_obs, info = env.reset()
# next_obs, next_shared_obs, rewards, dones, infos = env.step(actions)

from __future__ import annotations
from typing import Any
import mujoco
import numpy as np
from mujoco_lidar import MjLidarWrapper, scan_gen

from scene.map_generator import MapGenerator
from scene.map_to_scene import MapToScene
from scene.mujoco_builder import MujocoBuilder
from scene.robot_definition import RobotConfig
from envs.communication_manager import CommunicationConfig, CommunicationManager
from envs.coverage_tracker import CoverageTracker
from envs.waypoint_controller import BaseController, FYController
from sensors.lidar_processor import BaseLidarProcessor, Lidar2DProcessor

class ExplorationEnv:
    """
    Multi-agent exploration environment.

    High-level RL action per agent:
        [dx_local, dy_local] in [-1, 1]^2

    The action is mapped to a robot-local waypoint, transformed into world
    coordinates, and executed through a replaceable waypoint controller.
    """

    def __init__(
        self,
        n_agents: int = 2,
        max_episode_steps: int = 500,
        frame_skip: int = 1,
        lidar_cutoff: float = 5.0,
        render_mode: str | None = None,
        waypoint_radius: float = 2.0,
        max_control_steps_per_action: int = 50,
        controller: BaseController | None = None,
    ) -> None:
        self.n_agents = int(n_agents)
        self.max_episode_steps = int(max_episode_steps)
        self.frame_skip = int(frame_skip)
        self.lidar_cutoff = float(lidar_cutoff)
        self.render_mode = render_mode

        # High-level action specification.
        self.action_dim = 2
        self.action_mode = "local_waypoint"
        self.waypoint_radius = float(waypoint_radius)
        self.max_control_steps_per_action = int(max_control_steps_per_action)

        self.generator = MapGenerator()
        self.converter = MapToScene()
        self.builder = MujocoBuilder()

        # TODO: replace HDL64（CPU）maybe with LIVOX(MJX)
        # 另外，这个返回的是flattern的（11016,）但真实ldiar数据不一定是这样flattern的
        self.theta, self.phi = scan_gen.generate_HDL64()
        self.lidar_processor = Lidar2DProcessor(
            min_z=-0.2,
            max_z=0.2,
            n_bins=360,
            max_range=self.lidar_cutoff,
        )

        self.semantic_map = None
        self.scene = None
        self.robot_list = None

        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.lidars: list[MjLidarWrapper] = []

        self.current_step = 0  # Number of high-level RL decisions.
        self.physics_step_count = 0

        self.robot_dof_addresses: list[dict[str, int]] = []
        self.robot_body_ids: list[int] = []

        self.world_bounds = (0.0, 20.0, 0.0, 20.0)
        self.coverage_resolution = 0.25
        self.exploration_radius = 0.4

        self.coverage_tracker = CoverageTracker(
            world_bounds=self.world_bounds,
            resolution=self.coverage_resolution,
            exploration_radius=self.exploration_radius,
        )

        self.communication_manager = CommunicationManager(
            config=CommunicationConfig(enabled=False, message_dim=0),
            n_agents=self.n_agents,
        )

        # Dependency injection point for future robot-specific controllers.
        self.controller: BaseController = controller or FYController(
            max_forward_speed=0.5,
            max_yaw_rate=1.0,
            waypoint_tolerance=0.15,
            forward_gain=1.0,
            yaw_gain=2.0,
        )



    def reset(
        self,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Create a new episode."""
        self.current_step = 0
        self.physics_step_count = 0

        self.semantic_map = self.generator.generate()
        self.scene = self.converter.convert(self.semantic_map)

        spawn_positions = self.converter.sample_spawn_position(
            self.semantic_map,
            n_agents=self.n_agents,
            seed=seed,
        )

        self.robot_list = self._create_robot_configs(spawn_positions)
        self.model = self.builder.build(self.scene, self.robot_list)
        self.data = mujoco.MjData(self.model)

        self._cache_robot_indices()
        mujoco.mj_forward(self.model, self.data)

        self.lidars = [
            MjLidarWrapper(
                self.model,
                site_name=f"robot_{agent_id}_lidar_site",
                backend="cpu",
                cutoff_dist=self.lidar_cutoff,
            )
            for agent_id in range(self.n_agents)
        ]

        self.coverage_tracker.reset()
        initial_new_cells = self.coverage_tracker.update(
            self._get_robot_positions()
        )

        self.communication_manager.reset()
        self.controller.reset()

        obs = self._get_obs()
        shared_obs = self._get_shared_obs(obs)

        info = {
            "step": self.current_step,
            "physics_steps": self.physics_step_count,
            "spawn_positions": spawn_positions,
            "initial_explored_cells": initial_new_cells,
            "coverage_ratio": self.coverage_tracker.coverage_ratio,
            "action_mode": self.action_mode,
        }
        return obs, shared_obs, info

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ]:
        """
        Execute one high-level RL step.

        Args:
            actions: shape (n_agents, 2), each row is normalized
                [dx_local, dy_local] in the robot frame.
        """
        self._check_ready()
        actions = self._validate_actions(actions)

        waypoints_world = self._actions_to_world_waypoints(actions)
        reached = np.zeros(self.n_agents, dtype=bool)
        last_commands = np.zeros((self.n_agents, 2), dtype=np.float64)
        executed_control_steps = 0
        total_newly_explored = 0
        latest_coverage_status: dict[str, Any] = {}

        # One high-level waypoint action is executed by several low-level steps.
        for _ in range(self.max_control_steps_per_action):
            positions = self._get_robot_positions()
            yaws = self._get_robot_yaws()
            commands = np.zeros((self.n_agents, 2), dtype=np.float64)

            for agent_id in range(self.n_agents):
                if reached[agent_id]:
                    continue

                command, agent_reached = self.controller.compute_command(
                    robot_position=positions[agent_id],
                    robot_yaw=float(yaws[agent_id]),
                    waypoint_world=waypoints_world[agent_id],
                )
                commands[agent_id] = command
                reached[agent_id] = agent_reached

            last_commands = commands.copy()
            if np.all(reached):
                break

            self._apply_low_level_commands(commands)
            for _ in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)
                self.physics_step_count += 1

            executed_control_steps += 1

            # Stop an action early if the episode-level target is already met.
            latest_coverage_status = self.coverage_tracker.update(
                self._get_robot_positions()
            )
            total_newly_explored += int(
                latest_coverage_status.get("newly_explored", 0)
            )
            if self.coverage_tracker.coverage_ratio >= 0.9:
                break

        self.current_step += 1

        # Ensure coverage is updated even when every waypoint was already reached.
        if executed_control_steps == 0:
            latest_coverage_status = self.coverage_tracker.update(
                self._get_robot_positions()
            )
            total_newly_explored += int(
                latest_coverage_status.get("newly_explored", 0)
            )

        coverage_status = {
            **latest_coverage_status,
            "newly_explored": total_newly_explored,
        }
        coverage_ratio = self.coverage_tracker.coverage_ratio

        obs = self._get_obs()
        shared_obs = self._get_shared_obs(obs)
        rewards = self._compute_rewards(
            newly_explored=coverage_status["newly_explored"]
        )

        time_over = self.current_step >= self.max_episode_steps
        coverage_complete = coverage_ratio >= 0.9
        episode_done = time_over or coverage_complete
        dones = np.full(self.n_agents, episode_done, dtype=bool)

        infos = [
            {
                "agent_id": agent_id,
                "step": self.current_step,
                "physics_steps": self.physics_step_count,
                "waypoint_world": waypoints_world[agent_id].copy(),
                "waypoint_reached": bool(reached[agent_id]),
                "low_level_command": last_commands[agent_id].copy(),
                "control_steps": executed_control_steps,
                **coverage_status,
                "time_limit_reached": time_over,
                "coverage_complete": coverage_complete,
            }
            for agent_id in range(self.n_agents)
        ]

        return obs, shared_obs, rewards, dones, infos

    def _cache_robot_indices(self) -> None:
        """Cache MuJoCo body IDs and velocity DoF addresses."""
        self._check_ready()
        self.robot_dof_addresses = []
        self.robot_body_ids = []

        for agent_id in range(self.n_agents):
            robot_name = f"robot_{agent_id}"

            joint_ids = {
                "x": mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"{robot_name}_x_joint",
                ),
                "y": mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"{robot_name}_y_joint",
                ),
                "yaw": mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"{robot_name}_yaw_joint",
                ),
            }

            if any(joint_id == -1 for joint_id in joint_ids.values()):
                raise RuntimeError(f"Cannot find movement joints for {robot_name}")

            self.robot_dof_addresses.append(
                {
                    key: int(self.model.jnt_dofadr[joint_id])
                    for key, joint_id in joint_ids.items()
                }
            )

            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                robot_name,
            )
            if body_id == -1:
                raise RuntimeError(f"Cannot find body {robot_name}")
            self.robot_body_ids.append(int(body_id))

    def _validate_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        expected_shape = (self.n_agents, self.action_dim)
        if actions.shape != expected_shape:
            raise ValueError(
                f"Expected actions shape {expected_shape}, got {actions.shape}."
            )
        if not np.all(np.isfinite(actions)):
            raise ValueError("Actions contain NaN or infinity.")
        return np.clip(actions, -1.0, 1.0)

    def _actions_to_world_waypoints(self, actions: np.ndarray) -> np.ndarray:
        """
        Convert normalized robot-local [dx, dy] actions to world-frame goals.

        Robot frame convention:
            +x: forward
            +y: left
        """
        positions = self._get_robot_positions()
        yaws = self._get_robot_yaws()
        waypoints = np.zeros((self.n_agents, 2), dtype=np.float64)

        for agent_id in range(self.n_agents):
            dx_local = float(actions[agent_id, 0] * self.waypoint_radius)
            dy_local = float(actions[agent_id, 1] * self.waypoint_radius)
            yaw = float(yaws[agent_id])

            cos_yaw = float(np.cos(yaw))
            sin_yaw = float(np.sin(yaw))
            dx_world = cos_yaw * dx_local - sin_yaw * dy_local
            dy_world = sin_yaw * dx_local + cos_yaw * dy_local

            waypoints[agent_id] = [
                positions[agent_id, 0] + dx_world,
                positions[agent_id, 1] + dy_world,
            ]

        return waypoints

    def _apply_low_level_commands(self, commands: np.ndarray) -> None:
        """
        Apply FYController outputs to the current kinematic box robots.

        This is the robot-specific execution boundary. A future robot model can
        replace this function or delegate it to a RobotInterface without changing
        the policy action or waypoint controller interface.
        """
        expected_shape = (self.n_agents, 2)
        commands = np.asarray(commands, dtype=np.float64)
        if commands.shape != expected_shape:
            raise ValueError(
                f"Expected low-level commands shape {expected_shape}, "
                f"got {commands.shape}."
            )

        for agent_id in range(self.n_agents):
            dof = self.robot_dof_addresses[agent_id]
            body_id = self.robot_body_ids[agent_id]
            forward_speed = float(commands[agent_id, 0])
            yaw_rate = float(commands[agent_id, 1])

            rotation_matrix = self.data.xmat[body_id].reshape(3, 3)
            forward_direction = rotation_matrix[:, 0]

            self.data.qvel[dof["x"]] = forward_speed * forward_direction[0]
            self.data.qvel[dof["y"]] = forward_speed * forward_direction[1]
            self.data.qvel[dof["yaw"]] = yaw_rate

    def _get_robot_yaws(self) -> np.ndarray:
        """Read each robot's world-frame yaw from its body rotation matrix."""
        self._check_ready()
        yaws = []
        for body_id in self.robot_body_ids:
            rotation_matrix = self.data.xmat[body_id].reshape(3, 3)
            yaws.append(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
        return np.asarray(yaws, dtype=np.float64)

    def _create_robot_configs(
        self,
        spawn_positions: list[tuple[float, float, float]],
    ) -> list[RobotConfig]:
        colors = [
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
        ]
        return [
            RobotConfig(
                name=f"robot_{agent_id}",
                position=spawn_positions[agent_id],
                color=colors[agent_id % len(colors)],
                model_type="box",
                size=(0.2, 0.2, 0.2),
            )
            for agent_id in range(self.n_agents)
        ]

    def _get_lidar_obs(self) -> np.ndarray:
        """
        Return LiDAR point clouds for all agents.

        Returns:
            point_clouds:
                shape (n_agents, n_points, 3)
                每个点为 LiDAR local frame 下的 [x, y, z]。
        """
        self._check_ready()
        point_clouds = []

        for lidar in self.lidars:
            # Update current scan
            lidar.trace_rays(
                self.data,
                self.theta,
                self.phi,
            )
            
            points = lidar.get_hit_points()
            points = np.asarray(
                points,
                dtype=np.float32,
            )

            point_clouds.append(self.lidar_processor.process(points))
        return np.stack(point_clouds, axis=0)


    def _get_obs(self) -> np.ndarray:
        """
        Return raw LiDAR point clouds as agent observations.
        Returns:
            obs:
                shape (n_agents, n_points, 3)
        """
        return self._get_lidar_obs() 


    def _get_shared_obs(self, obs: np.ndarray) -> np.ndarray:
        """Give every agent the same concatenated centralized observation."""
        global_obs = obs.reshape(-1)
        return np.repeat(global_obs[None, :], self.n_agents, axis=0).astype(
            np.float32
        )

    def _compute_rewards(self, newly_explored: int) -> np.ndarray:
        coverage_reward = float(newly_explored)
        step_penalty = 0.01
        team_reward = coverage_reward - step_penalty
        return np.full((self.n_agents, 1), team_reward, dtype=np.float32)

    def _get_robot_positions(self) -> np.ndarray:
        self._check_ready()
        if not self.robot_body_ids:
            raise RuntimeError("Robot body IDs have not been initialized.")
        return np.asarray(
            [self.data.xpos[body_id].copy() for body_id in self.robot_body_ids],
            dtype=np.float32,
        )

    def _check_ready(self) -> None:
        if self.model is None or self.data is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")