from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from .base_algorithm import BaseAlgorithm

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for MAPPO training.")

DEVICE = torch.device("cuda:0")

@dataclass
class MAPPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    ppo_epochs: int = 4
    grad_norm: float = 0.5

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5, dtype=torch.float32))

    def forward(self, obs: torch.Tensor):
        features = self.encoder(obs)
        mean = self.mean_head(features)
        log_std = torch.clamp(self.log_std, min=-5.0, max=2.0)
        std = torch.exp(log_std).expand_as(mean)
        return mean, std

class ValueNetwork(nn.Module):
    def __init__(self, global_obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, global_obs: torch.Tensor):
        return self.net(global_obs)

class Buffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs = []
        self.global_obs = []
        self.actions = []
        self.raw_actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

def sample_squashed_action(mean, std):
    dist = torch.distributions.Normal(mean, std)
    raw_action = dist.rsample()
    action = torch.tanh(raw_action)
    log_prob = dist.log_prob(raw_action)
    log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
    log_prob = log_prob.sum(dim=-1)
    return action, raw_action, log_prob

def compute_gae(rewards, values, dones, next_value, gamma, lam):
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(len(rewards))):
        next_val = next_value if t == len(rewards) - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * non_terminal - values[t]
        gae = delta + gamma * lam * non_terminal * gae
        advantages[t] = gae

    returns = advantages + values

    advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=DEVICE)
    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=DEVICE)

    advantages_t = (advantages_t - advantages_t.mean()) / (
        advantages_t.std(unbiased=False) + 1e-8
    )
    return advantages_t, returns_t

class MAPPO(BaseAlgorithm):
    def __init__(self, obs_dim, action_dim, global_obs_dim, agents, config=None):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.global_obs_dim = int(global_obs_dim)
        self.agents = list(agents)
        self.config = config or MAPPOConfig()

        self.policies = {
            a: PolicyNetwork(self.obs_dim, self.action_dim).to(DEVICE)
            for a in self.agents
        }
        self.critics = {
            a: ValueNetwork(self.global_obs_dim).to(DEVICE)
            for a in self.agents
        }
        self.optimizers = {
            a: torch.optim.Adam(
                list(self.policies[a].parameters()) + list(self.critics[a].parameters()),
                lr=self.config.lr,
            )
            for a in self.agents
        }
        self.buffers = {a: Buffer() for a in self.agents}
        self._current_records = None
        self._last_shared_obs = None

    def _select_action(self, agent, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            mean, std = self.policies[agent](obs_t)
            action, raw_action, log_prob = sample_squashed_action(mean, std)
        return (
            action.cpu().numpy().astype(np.float32),
            raw_action.cpu().numpy().astype(np.float32),
            float(log_prob.item()),
        )

    def _predict(self, agent, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            mean, _ = self.policies[agent](obs_t)
            action = torch.tanh(mean)
        return action.cpu().numpy().astype(np.float32)

    def _value(self, agent, global_obs):
        g = torch.as_tensor(global_obs, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            value = self.critics[agent](g)
        return float(value.squeeze().item())

    def act(self, obs, shared_obs=None, training: bool = True):
        actions = np.zeros((len(self.agents), self.action_dim), dtype=np.float32)

        if training:
            if shared_obs is None:
                raise ValueError("MAPPO requires shared_obs during training.")
            records = []

            for agent_id, agent in enumerate(self.agents):
                action, raw_action, log_prob = self._select_action(agent, obs[agent_id])
                value = self._value(agent, shared_obs[agent_id])
                actions[agent_id] = action
                records.append({
                    "obs": obs[agent_id].copy(),
                    "global_obs": shared_obs[agent_id].copy(),
                    "action": action.copy(),
                    "raw_action": raw_action.copy(),
                    "log_prob": log_prob,
                    "value": value,
                })

            self._current_records = records

        else:
            for agent_id, agent in enumerate(self.agents):
                actions[agent_id] = self._predict(agent, obs[agent_id])

        return actions

    def observe(self, **transition):
        if self._current_records is None:
            return

        rewards = np.asarray(transition["rewards"], dtype=np.float32).reshape(
            len(self.agents), -1
        )[:, 0]
        dones = np.asarray(transition["dones"], dtype=bool)
        self._last_shared_obs = transition.get("next_shared_obs")

        for agent_id, agent in enumerate(self.agents):
            record = self._current_records[agent_id]
            buf = self.buffers[agent]
            buf.obs.append(record["obs"])
            buf.global_obs.append(record["global_obs"])
            buf.actions.append(record["action"])
            buf.raw_actions.append(record["raw_action"])
            buf.log_probs.append(record["log_prob"])
            buf.values.append(record["value"])
            buf.rewards.append(float(rewards[agent_id]))
            buf.dones.append(float(dones[agent_id]))

        self._current_records = None

    def update(self, **kwargs):
        next_shared_obs = kwargs.get("next_shared_obs", self._last_shared_obs)
        if next_shared_obs is None:
            return {}

        all_losses = []
        for agent_id, agent in enumerate(self.agents):
            losses = self._update_agent(agent, next_shared_obs[agent_id])
            if losses:
                all_losses.extend(losses)

        if not all_losses:
            return {}

        return {
            "Loss/Total": float(np.mean([x["total"] for x in all_losses])),
            "Loss/Policy": float(np.mean([x["policy"] for x in all_losses])),
            "Loss/Value": float(np.mean([x["value"] for x in all_losses])),
            "Loss/Entropy": float(np.mean([x["entropy"] for x in all_losses])),
        }

    def _update_agent(self, agent, next_global_obs):
        buf = self.buffers[agent]
        if len(buf.rewards) == 0:
            return None

        next_value = 0.0 if bool(buf.dones[-1]) else self._value(agent, next_global_obs)
        advantages, returns = compute_gae(
            buf.rewards, buf.values, buf.dones, next_value,
            self.config.gamma, self.config.lam
        )

        obs_batch = torch.as_tensor(np.asarray(buf.obs), dtype=torch.float32, device=DEVICE)
        global_obs_batch = torch.as_tensor(np.asarray(buf.global_obs), dtype=torch.float32, device=DEVICE)
        raw_action_batch = torch.as_tensor(np.asarray(buf.raw_actions), dtype=torch.float32, device=DEVICE)
        old_log_probs = torch.as_tensor(np.asarray(buf.log_probs), dtype=torch.float32, device=DEVICE)

        losses = []

        for _ in range(self.config.ppo_epochs):
            mean, std = self.policies[agent](obs_batch)
            dist = torch.distributions.Normal(mean, std)

            action_batch = torch.tanh(raw_action_batch)
            new_log_probs = dist.log_prob(raw_action_batch)
            new_log_probs -= torch.log(1.0 - action_batch.pow(2) + 1e-6)
            new_log_probs = new_log_probs.sum(dim=-1)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 1.0 - self.config.clip, 1.0 + self.config.clip
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            values = self.critics[agent](global_obs_batch).squeeze(-1)
            value_loss = ((values - returns) ** 2).mean()

            entropy = dist.entropy().sum(dim=-1).mean()

            total_loss = (
                policy_loss
                + self.config.value_coef * value_loss
                - self.config.entropy_coef * entropy
            )

            optimizer = self.optimizers[agent]
            optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(self.policies[agent].parameters())
                + list(self.critics[agent].parameters()),
                self.config.grad_norm,
            )
            optimizer.step()

            losses.append({
                "total": float(total_loss.item()),
                "policy": float(policy_loss.item()),
                "value": float(value_loss.item()),
                "entropy": float(entropy.item()),
            })

        buf.clear()
        return losses

    def save(self, path, **kwargs):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "episode": kwargs.get("episode"),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "global_obs_dim": self.global_obs_dim,
            "agents": self.agents,
            "policies": {a: self.policies[a].state_dict() for a in self.agents},
            "critics": {a: self.critics[a].state_dict() for a in self.agents},
            "optimizers": {a: self.optimizers[a].state_dict() for a in self.agents},
        }, path)

    def load(self, path, **kwargs):
        checkpoint = torch.load(path, map_location=DEVICE)
        load_optimizers = kwargs.get("load_optimizers", False)

        for agent in self.agents:
            self.policies[agent].load_state_dict(checkpoint["policies"][agent])
            self.critics[agent].load_state_dict(checkpoint["critics"][agent])

            if load_optimizers and "optimizers" in checkpoint:
                self.optimizers[agent].load_state_dict(checkpoint["optimizers"][agent])

        return checkpoint
