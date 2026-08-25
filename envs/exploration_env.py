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
from envs.waypoint_controller import BaseController, AStarPurePursuitController
from sensors.lidar_processor import BaseLidarProcessor, Lidar2DProcessor
from agent.agent_knowledge import AgentKnowledge

class ExplorationEnv:
    """
    Multi-agent exploration environment.

    High-level RL action per agent is a discrete frontier candidate index.
    """

    def __init__(
        self,
        n_agents: int = 2,
        max_episode_steps: int = 500,
        frame_skip: int = 10,
        lidar_cutoff: float = 5.0,
        render_mode: str | None = None,
        waypoint_radius: float = 2.0,
        max_frontier_candidates: int = 16,
        min_candidate_separation: float = 0.75,
        collision_penalty: float = 0.1,
        collision_enabled: bool = False,
        max_control_steps_per_action: int = 2000,
        block_patience_steps: int = 100,
        block_progress_epsilon: float = 1e-5,
        controller: BaseController | None = None,
    ) -> None:
        self.n_agents = int(n_agents)
        self.max_episode_steps = int(max_episode_steps)
        self.frame_skip = int(frame_skip)
        self.lidar_cutoff = float(lidar_cutoff)
        self.render_mode = render_mode

        # High-level action specification.
        self.action_dim = int(max_frontier_candidates)
        self.action_mode = "discrete_frontier"
        self.waypoint_radius = float(waypoint_radius)
        self.max_control_steps_per_action = int(max_control_steps_per_action)
        self.collision_penalty = float(collision_penalty)
        self.collision_enabled = bool(collision_enabled)
        self.frontier_feature_dim = 6
        self.min_candidate_separation = float(min_candidate_separation)
        self.block_patience_steps = int(block_patience_steps)
        self.block_progress_epsilon = float(block_progress_epsilon)
        self.frontier_candidates_world = np.zeros(
            (self.n_agents, self.action_dim, 2), dtype=np.float32
        )
        self.frontier_action_masks = np.zeros(
            (self.n_agents, self.action_dim), dtype=bool
        )

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
        self._collision_active = np.zeros(self.n_agents, dtype=bool)
        self.episode_collision_count = 0

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
        self.controller: BaseController = controller or AStarPurePursuitController(
            max_forward_speed=0.5,
            max_yaw_rate=1.0,
            waypoint_tolerance=0.15,
            forward_gain=1.0,
            yaw_gain=2.0,
            lookahead_distance=0.5,
        )

        self.agent_knowledge = [
        AgentKnowledge(
            world_bounds=self.world_bounds,
            map_resolution=self.coverage_resolution,
            visit_radius=self.exploration_radius,
            patch_size_m=5.0,
        )for _ in range(self.n_agents)
]


    def reset(self,seed: int | None = None)-> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        self.current_step = 0
        self.physics_step_count = 0
        self.episode_collision_count = 0
        self._collision_active.fill(False)

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

        for agent_id in range(self.n_agents):
            self.agent_knowledge[agent_id].reset()

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
            actions: integer frontier indices, shape (n_agents,) or
                (n_agents, 1).
        
        """
        self._check_ready()
        actions = self._validate_actions(actions)

        waypoints_world = self._actions_to_world_waypoints(actions)
        option_valid_candidates = self.frontier_action_masks.sum(axis=1).astype(np.int32)
        option_min_pairwise_distance = np.asarray(
            [self._minimum_candidate_distance(agent_id) for agent_id in range(self.n_agents)],
            dtype=np.float32,
        )
        option_start_positions = self._get_robot_positions()[:, :2].copy()
        option_start_distances = np.linalg.norm(
            waypoints_world - option_start_positions, axis=1
        )
        reached = np.zeros(self.n_agents, dtype=bool)
        blocked = np.zeros(self.n_agents, dtype=bool)
        active = np.ones(self.n_agents, dtype=bool)
        path_planned = np.ones(self.n_agents, dtype=bool)
        planned_path_lengths = np.zeros(self.n_agents, dtype=np.int32)
        if isinstance(self.controller, AStarPurePursuitController):
            for agent_id in range(self.n_agents):
                occupancy = self.agent_knowledge[agent_id].map
                path = self.controller.plan_path(
                    agent_id,
                    occupancy.grid,
                    option_start_positions[agent_id],
                    waypoints_world[agent_id],
                    xmin=occupancy.xmin,
                    ymin=occupancy.ymin,
                    resolution=occupancy.resolution,
                )
                path_planned[agent_id] = path is not None
                if path is None:
                    blocked[agent_id] = True
                    active[agent_id] = False
                else:
                    planned_path_lengths[agent_id] = len(path)
        option_durations = np.zeros(self.n_agents, dtype=np.int32)
        stalled_steps = np.zeros(self.n_agents, dtype=np.int32)
        previous_positions = option_start_positions.copy()
        last_commands = np.zeros((self.n_agents, 2), dtype=np.float64)
        executed_control_steps = 0
        collision_counts = np.zeros(self.n_agents, dtype=np.int32)
        total_newly_explored = 0
        latest_coverage_status: dict[str, Any] = {}

        # One high-level waypoint action is executed by several low-level steps.
        for _ in range(self.max_control_steps_per_action):
            positions = self._get_robot_positions()
            yaws = self._get_robot_yaws()
            commands = np.zeros((self.n_agents, 2), dtype=np.float64)

            for agent_id in range(self.n_agents):
                if not active[agent_id]:
                    continue

                command, agent_reached = self.controller.compute_command_for_agent(
                    agent_id=agent_id,
                    robot_position=positions[agent_id],
                    robot_yaw=float(yaws[agent_id]),
                    waypoint_world=waypoints_world[agent_id],
                )
                commands[agent_id] = command
                reached[agent_id] = agent_reached
                if agent_reached:
                    active[agent_id] = False

            last_commands = commands.copy()
            if not np.any(active):
                break

            self._apply_low_level_commands(commands)
            for _ in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)
                self.physics_step_count += 1
                if self.collision_enabled:
                    collision_counts += self._update_collision_events()

            option_durations[active] += 1
            positions_after = self._get_robot_positions()
            for agent_id in range(self.n_agents):
                if not active[agent_id]:
                    continue
                if self.collision_enabled and self._collision_active[agent_id]:
                    blocked[agent_id] = True
                    active[agent_id] = False
                    continue
                movement = float(
                    np.linalg.norm(positions_after[agent_id, :2] - previous_positions[agent_id])
                )
                if commands[agent_id, 0] > 1e-6 and movement < self.block_progress_epsilon:
                    stalled_steps[agent_id] += 1
                else:
                    stalled_steps[agent_id] = 0
                if stalled_steps[agent_id] >= self.block_patience_steps:
                    blocked[agent_id] = True
                    active[agent_id] = False
                previous_positions[agent_id] = positions_after[agent_id, :2]

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
        final_positions = self._get_robot_positions()[:, :2]
        final_distances = np.linalg.norm(waypoints_world - final_positions, axis=1)
        distance_progress = option_start_distances - final_distances

        obs = self._get_obs()
        shared_obs = self._get_shared_obs(obs)
        rewards = self._compute_rewards(
            newly_explored=coverage_status["newly_explored"],
            collision_counts=collision_counts,
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
                "option_blocked": bool(blocked[agent_id]),
                "path_planned": bool(path_planned[agent_id]),
                "planned_path_points": int(planned_path_lengths[agent_id]),
                "option_duration": int(option_durations[agent_id]),
                "distance_progress": float(distance_progress[agent_id]),
                "valid_frontier_candidates": int(option_valid_candidates[agent_id]),
                "minimum_candidate_distance": float(
                    option_min_pairwise_distance[agent_id]
                ),
                "low_level_command": last_commands[agent_id].copy(),
                "control_steps": executed_control_steps,
                "frontier_index": int(actions[agent_id]),
                "collision_count": int(collision_counts[agent_id]),
                "collision_enabled": self.collision_enabled,
                "episode_collision_count": int(self.episode_collision_count),
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
        raw_actions = np.asarray(actions)
        if raw_actions.shape == (self.n_agents, 1):
            raw_actions = raw_actions[:, 0]
        expected_shape = (self.n_agents,)
        if raw_actions.shape != expected_shape:
            raise ValueError(
                f"Expected actions shape {expected_shape} or "
                f"({self.n_agents}, 1), got {raw_actions.shape}."
            )
        if not np.all(np.isfinite(raw_actions)):
            raise ValueError("Actions contain NaN or infinity.")
        if not np.all(raw_actions == np.floor(raw_actions)):
            raise ValueError("Discrete frontier actions must be integers.")
        actions = raw_actions.astype(np.int64)
        if np.any(actions < 0) or np.any(actions >= self.action_dim):
            raise ValueError(f"Frontier action must be in [0, {self.action_dim}).")
        selected_valid = self.frontier_action_masks[np.arange(self.n_agents), actions]
        if not np.all(selected_valid):
            raise ValueError("Action selected a padded/invalid frontier candidate.")
        return actions

    def _actions_to_world_waypoints(self, actions: np.ndarray) -> np.ndarray:
        """
        Resolve each discrete frontier index to its cached world-frame target.
        """
        return self.frontier_candidates_world[
            np.arange(self.n_agents), actions
        ].astype(np.float64, copy=True)

    def _apply_low_level_commands(self, commands: np.ndarray) -> None:
        """
        Apply waypoint-controller outputs to the current kinematic box robots.

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
        positions = self._get_robot_positions()
        yaws = self._get_robot_yaws()

        for agent_id, lidar in enumerate(self.lidars):
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

            # Mapping is a side channel only: the existing processed LiDAR
            # observation returned below remains byte-for-byte the same shape.
            self.agent_knowledge[agent_id].update_from_lidar(
                hit_points_local=points,
                robot_position=positions[agent_id],
                robot_yaw=float(yaws[agent_id]),
            )

            point_clouds.append(self.lidar_processor.process(points))
        return np.stack(point_clouds, axis=0)

    def get_mapping_snapshot(self, agent_id: int) -> dict[str, np.ndarray]:
        """Return copies of map products for inspection/visualisation."""
        if not 0 <= agent_id < self.n_agents:
            raise IndexError(f"agent_id must be in [0, {self.n_agents}), got {agent_id}")
        occupancy = self.agent_knowledge[agent_id].map
        frontier_mask = occupancy.compute_frontier_mask()
        return {
            "occupancy_grid": occupancy.grid.copy(),
            "frontier_mask": frontier_mask,
            "frontier_candidates": occupancy.compute_frontier_candidates(
                frontier_mask
            ),
        }

    def get_action_masks(self) -> np.ndarray:
        """Return the current fixed-size frontier validity masks."""
        return self.frontier_action_masks.copy()

    def _build_frontier_features(
        self,
        agent_id: int,
        robot_position: np.ndarray,
        robot_yaw: float,
    ) -> np.ndarray:
        """Cache padded candidates and return their fixed-size local features."""
        occupancy = self.agent_knowledge[agent_id].map
        candidates = occupancy.compute_frontier_candidates()
        features = np.zeros(
            (self.action_dim, self.frontier_feature_dim), dtype=np.float32
        )
        self.frontier_candidates_world[agent_id].fill(0.0)
        self.frontier_action_masks[agent_id].fill(False)

        if len(candidates) == 0:
            # Safe Categorical fallback: selecting slot zero means stay put.
            self.frontier_candidates_world[agent_id, 0] = robot_position[:2]
            self.frontier_action_masks[agent_id, 0] = True
            return features

        scored_candidates = []
        for candidate in candidates:
            row, col = occupancy.world_to_grid(float(candidate[0]), float(candidate[1]))
            radius = 2
            patch = occupancy.grid[
                max(0, row - radius):min(occupancy.height, row + radius + 1),
                max(0, col - radius):min(occupancy.width, col + radius + 1),
            ]
            information_gain = float(np.mean(patch == occupancy.UNKNOWN))
            distance = float(np.linalg.norm(candidate - robot_position[:2]))
            scored_candidates.append((information_gain, -distance, candidate))

        scored_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = []
        for scored_candidate in scored_candidates:
            candidate = scored_candidate[2]
            if all(
                np.linalg.norm(candidate - kept[2]) >= self.min_candidate_separation
                for kept in selected
            ):
                selected.append(scored_candidate)
            if len(selected) == self.action_dim:
                break
        cos_yaw, sin_yaw = float(np.cos(robot_yaw)), float(np.sin(robot_yaw))
        for index, (information_gain, _, candidate) in enumerate(selected):
            dx_world = float(candidate[0] - robot_position[0])
            dy_world = float(candidate[1] - robot_position[1])
            dx_local = cos_yaw * dx_world + sin_yaw * dy_world
            dy_local = -sin_yaw * dx_world + cos_yaw * dy_world
            distance = float(np.hypot(dx_local, dy_local))
            relative_angle = float(np.arctan2(dy_local, dx_local))
            features[index] = (
                dx_local,
                dy_local,
                distance,
                np.sin(relative_angle),
                np.cos(relative_angle),
                information_gain,
            )
            self.frontier_candidates_world[agent_id, index] = candidate
            self.frontier_action_masks[agent_id, index] = True
        return features

    def _minimum_candidate_distance(self, agent_id: int) -> float:
        candidates = self.frontier_candidates_world[
            agent_id, self.frontier_action_masks[agent_id]
        ]
        if len(candidates) < 2:
            return 0.0
        deltas = candidates[:, None, :] - candidates[None, :, :]
        distances = np.linalg.norm(deltas, axis=-1)
        distances[np.eye(len(candidates), dtype=bool)] = np.inf
        return float(np.min(distances))


    def _get_obs(self) -> np.ndarray:
        """
        Build policy observation for each agent.

        Current observation:
            LiDAR scan
            +
            agent-local visited memory
        """
        lidar_obs = self._get_lidar_obs()
        positions = self._get_robot_positions()
        yaws = self._get_robot_yaws()

        observations = []

        for agent_id in range(self.n_agents):

            knowledge_obs = (
                self.agent_knowledge[
                    agent_id
                ].get_map_observation(
                    positions[agent_id]
                )
            )
            frontier_features = self._build_frontier_features(
                agent_id,
                positions[agent_id],
                float(yaws[agent_id]),
            )

            obs = np.concatenate(
                [
                    lidar_obs[agent_id],
                    knowledge_obs,
                    np.asarray(
                        [np.sin(yaws[agent_id]), np.cos(yaws[agent_id])],
                        dtype=np.float32,
                    ),
                    frontier_features.reshape(-1),
                ]
            ).astype(np.float32)

            observations.append(obs)

        return np.stack(
            observations,
            axis=0,
        ) 


    def _get_shared_obs(self, obs: np.ndarray) -> np.ndarray:
        """Give every agent the same concatenated centralized observation."""
        global_obs = obs.reshape(-1)
        return np.repeat(global_obs[None, :], self.n_agents, axis=0).astype(
            np.float32
        )

    def _compute_rewards(
        self,
        newly_explored: int,
        collision_counts: np.ndarray,
    ) -> np.ndarray:
        coverage_reward = float(newly_explored)
        step_penalty = 0.01
        team_reward = coverage_reward - step_penalty
        rewards = np.full((self.n_agents, 1), team_reward, dtype=np.float32)
        rewards[:, 0] -= self.collision_penalty * collision_counts
        return rewards

    def _update_collision_events(self) -> np.ndarray:
        """Count new robot contact events while ignoring the floor."""
        contact_active = np.zeros(self.n_agents, dtype=bool)
        body_to_agent = {
            body_id: agent_id for agent_id, body_id in enumerate(self.robot_body_ids)
        }
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geom_ids = (int(contact.geom1), int(contact.geom2))
            bodies = [int(self.model.geom_bodyid[geom_id]) for geom_id in geom_ids]
            body_names = [
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
                for body_id in bodies
            ]
            if any(name == "floor" for name in body_names):
                continue
            for body_id in bodies:
                if body_id in body_to_agent:
                    contact_active[body_to_agent[body_id]] = True

        new_events = contact_active & ~self._collision_active
        self._collision_active = contact_active
        count = new_events.astype(np.int32)
        self.episode_collision_count += int(np.sum(count))
        return count

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
