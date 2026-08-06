# 这个负责包装mujoco-> gym 
# env.rest,env.step(actions)
# obs, shared_obs, info = env.reset()
# next_obs, next_shared_obs, rewards, dones, infos = env.step(actions)

from __future__ import annotations

from typing import Any
from mujoco_lidar import MjLidarWrapper, scan_gen

import mujoco
import numpy as np

from scene.map_generator import MapGenerator
from scene.map_to_scene import MapToScene
from scene.robot_definition import RobotConfig
from scene.mujoco_builder import MujocoBuilder
from envs.coverage_tracker import CoverageTracker

class ExplorationEnv:
    def __init__(
        self,
        n_agents: int = 2,
        max_episode_steps: int = 500,
        frame_skip: int = 1,
        lidar_cutoff: float = 5.0,
        render_mode: str | None = None,
    ) -> None:

        self.n_agents = n_agents
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip 
        self.lidar_cutoff = lidar_cutoff
        self.render_mode = render_mode

        self.generator = MapGenerator()
        self.converter = MapToScene()
        self.builder = MujocoBuilder()

        # LiDAR 扫描模式暂时用 HDL64 (而且用的cpu，没有用到tachi或者mjx，后面要改动) TODO
        self.theta, self.phi = scan_gen.generate_HDL64()

        self.semantic_map = None
        self.scene = None
        self.robot_list = None

        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None

        self.lidars: list[MjLidarWrapper] = []

        self.current_step = 0

        self.world_bounds = (
            0.0,
            20.0,
            0.0,
            20.0,
        )

        self.coverage_resolution = 0.25
        self.exploration_radius = 0.4

        self.coverage_tracker = CoverageTracker(
            world_bounds=self.world_bounds,
            resolution=self.coverage_resolution,
            exploration_radius=self.exploration_radius,
        )

    def reset(self,seed: int | None = None,) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        create new episode
        Returns:
            obs:每个 agent 的局部 observation
            shared_obs:centralized critic 使用的全局 observation
            info:环境附加信息，主要是传递训练用不到但是调试和评估有用的东西
        """
        self.current_step = 0

        # 1. Generate map
        self.semantic_map = self.generator.generate() 

        # 2. Convert semantic map to scene
        self.scene = self.converter.convert(self.semantic_map)

        # 3. Sample robot spawn positions
        spawn_positions = self.converter.sample_spawn_position(
            self.semantic_map,
            n_agents=self.n_agents,
            seed=seed,
        )

        # 4. Create robot definitions
        self.robot_list = self._create_robot_configs(spawn_positions)

        # 5. Build MuJoCo model and data
        self.model = self.builder.build(
            self.scene,
            self.robot_list,
        )
        self.data = mujoco.MjData(self.model)

        # 初始化 robot joint（但是可能是box only，后续再改）
        self.robot_dof_addresses = []
        for agent_id in range(self.n_agents):
            robot_name = f"robot_{agent_id}"

            x_joint_id = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,f"{robot_name}_x_joint",)

            y_joint_id = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,f"{robot_name}_y_joint",)

            yaw_joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{robot_name}_yaw_joint",
            )

            if (x_joint_id == -1 or y_joint_id == -1 or yaw_joint_id == -1):
                raise RuntimeError(f"Cannot find movement joints for {robot_name}")

            self.robot_dof_addresses.append({
                "x": int(self.model.jnt_dofadr[x_joint_id]),
                "y": int(self.model.jnt_dofadr[y_joint_id]),
                "yaw": int(self.model.jnt_dofadr[yaw_joint_id]),
            })


        # 确保派生状态被正确计算，例如 site_xpos
        mujoco.mj_forward(self.model, self.data)

        # 6. Create one LiDAR wrapper per agent
        self.lidars = [
            MjLidarWrapper(
                self.model,
                site_name=f"robot_{agent_id}_lidar_site",
                backend="cpu",
                cutoff_dist=self.lidar_cutoff,
            )
            for agent_id in range(self.n_agents)
        ]

        # 7. Initial observations
        obs = self._get_obs()
        shared_obs = self._get_shared_obs(obs)

        # 8. Initialise coverage map
        self.coverage_tracker.reset()
        robot_positions = self._get_robot_positions()
        initial_new_cells = self.coverage_tracker.update(robot_positions)

        # 9. storage robot body id:
        self.robot_body_ids = []
        for agent_id in range(self.n_agents):
            robot_name = f"robot_{agent_id}"

            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                robot_name,
            )

            if body_id == -1:
                raise RuntimeError(
                    f"Cannot find body {robot_name}"
                )

            self.robot_body_ids.append(body_id)


        info = {
            "step": self.current_step,
            "spawn_positions": spawn_positions,
            "initial_explored_cells": initial_new_cells,
            "coverage_ratio": (self.coverage_tracker.coverage_ratio),
        }

        return obs, shared_obs, info

    def step(self,actions: np.ndarray) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ]:
        """
        执行一次环境 step。

        Args:
            actions:
                shape 通常为 (n_agents, action_dim)。

        Returns:
            obs:
                shape (n_agents, obs_dim)
            shared_obs:
                shape (n_agents, shared_obs_dim)
            rewards:
                shape (n_agents, 1)
            dones:
                shape (n_agents,)
            infos:
                每个 agent 一个 info dict
        """
        self._check_ready()

        actions = np.asarray(actions, dtype=np.float32)

        if actions.shape[0] != self.n_agents:
            raise ValueError(
                f"Expected actions for {self.n_agents} agents, "
                f"but received shape {actions.shape}."
            )

        # 1. Apply agent actions
        self._apply_actions(actions)

        # 2. Advance MuJoCo physics
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.current_step += 1

        # 2.5： Get robot pos + calcuate coverage
        robot_positions = self._get_robot_positions()
        coverage_status = self.coverage_tracker.update(robot_positions)
        coverage_ratio = (self.coverage_tracker.coverage_ratio)

        # 3. Read new state
        obs = self._get_obs()
        shared_obs = self._get_shared_obs(obs)

        # 4. Compute reward
        rewards = self._compute_rewards(newly_explored = coverage_status["newly_explored"])

        # 5. Episode termination
        time_over = self.current_step >= self.max_episode_steps
        coverage_complete = coverage_ratio >= 0.9 
        episode_done = (time_over or coverage_complete)

        dones = np.full(
            shape=(self.n_agents,),
            fill_value=episode_done,
            dtype=bool,
        )


        infos = [
            {
                "agent_id": agent_id,
                "step": self.current_step,
                **coverage_status,
                "time_limit_reached": time_over,
                "coverage_complete": coverage_complete,
            }
            for agent_id in range(self.n_agents)
        ]

        return obs, shared_obs, rewards, dones, infos

    def _create_robot_configs(
        self,
        spawn_positions: list[tuple[float, float, float]],
    ) -> list[RobotConfig]:
        colors = [
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
        ]

        robot_configs = []

        for agent_id in range(self.n_agents):
            color = colors[agent_id % len(colors)]

            robot_configs.append(
                RobotConfig(
                    name=f"robot_{agent_id}",
                    position=spawn_positions[agent_id],
                    color=color,
                    model_type="box",
                    size=(0.2, 0.2, 0.2),
                )
            )

        return robot_configs

    def _get_obs(self) -> np.ndarray:
        """
        暂时直接返回 LiDAR ranges。
        后面会把它改成固定数量的二维射线，并加入速度、方向等状态。TODO
        """
        self._check_ready()

        observations = []

        for agent_id in range(self.n_agents):
            ranges = self.lidars[agent_id].trace_rays(
                self.data,
                self.theta,
                self.phi,
            )

            ranges = np.asarray(ranges, dtype=np.float32)

            # 将无穷值和异常值限制在 cutoff_dist
            ranges = np.nan_to_num(
                ranges,
                nan=self.lidar_cutoff,
                posinf=self.lidar_cutoff,
                neginf=0.0,
            )

            ranges = np.clip(
                ranges,
                0.0,
                self.lidar_cutoff,
            )

            # 归一化到 [0, 1]
            ranges = ranges / self.lidar_cutoff

            observations.append(ranges)

        return np.stack(observations, axis=0)

    def _get_shared_obs(self,obs: np.ndarray)-> np.ndarray:
        """
        第一版 centralized observation：把所有 agent 的局部 observation 拼起来。
        每个 agent 都得到同一份 shared observation。
        """
        global_obs = obs.reshape(-1)

        shared_obs = np.repeat(
            global_obs[None, :],
            repeats=self.n_agents,
            axis=0,
        )

        return shared_obs.astype(np.float32)

    def _apply_actions(self, actions: np.ndarray)-> None:
        """
        必须根据机器人使用的 joint 类型决定：
        - freejoint
        - slide joint
        - mocap
        - actuator
        - 直接修改 qvel
        我要更改他，从最开始的变成2维，我之前没有自己细致写到过这个程度
        """  
        # raise NotImplementedError("Action application has not been implemented yet.")
        expected_shape = (self.n_agents, 2)

        if actions.shape != expected_shape:
            raise ValueError(
                f"Expected actions shape {expected_shape}, "
                f"but received {actions.shape}"
            )

        actions = np.clip(actions, -1.0, 1.0)

        max_forward_velocity = 1.5
        max_yaw_rate = 1.0

        for agent_id in range(self.n_agents):
            dof = self.robot_dof_addresses[agent_id]
            body_id = self.robot_body_ids[agent_id]

            forward_velocity = (
                actions[agent_id, 0]
                * max_forward_velocity
            )

            yaw_rate = (
                actions[agent_id, 1]
                * max_yaw_rate
            )

            # MuJoCo 保存的是 body 从局部坐标到世界坐标的旋转矩阵
            rotation_matrix = self.data.xmat[
                body_id
            ].reshape(3, 3)

            # 假设机器人局部 +x 轴是前方。
            # 取局部 x 轴在世界坐标中的方向。
            forward_direction = rotation_matrix[:, 0]

            vx = forward_velocity * forward_direction[0]
            vy = forward_velocity * forward_direction[1]

            self.data.qvel[dof["x"]] = vx
            self.data.qvel[dof["y"]] = vy
            self.data.qvel[dof["yaw"]] = yaw_rate

    def _compute_rewards(self,newly_explored: int)->np.ndarray:
        """
        临时 reward。
        环境框架测试期间先返回 0。
        后面再加入 exploration coverage、collision、step penalty。
        """
        # 测试期间返回0
        # return np.zeros(
        #     shape=(self.n_agents, 1),
        #     dtype=np.float32,
        # )
        coverage_reward = float(newly_explored)
        step_penalty = 0.01

        team_reward = (
            coverage_reward
            - step_penalty
        )

        return np.full(
            (self.n_agents, 1),
            team_reward,
            dtype=np.float32,
        )

    def _get_robot_positions(self) -> np.ndarray:
        self._check_ready()

        positions = []

        for agent_id in range(self.n_agents):
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"robot_{agent_id}",
            )

            if body_id == -1:
                raise RuntimeError(
                    f"Cannot find body robot_{agent_id}"
                )

            positions.append(
                self.data.xpos[body_id].copy()
            )

        return np.asarray(
            positions,
            dtype=np.float32,
        )

    def _check_ready(self) -> None:
        if self.model is None or self.data is None:
            raise RuntimeError(
                "Environment is not initialized. Call reset() first."
            )
