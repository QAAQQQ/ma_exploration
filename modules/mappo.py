import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# Config
# ======================
class Config:
    env_name         = "exploration"   # "exploration" | "minigrid"
    experiment_name  = "baseline"
    total_episodes   = 5000
    episode_length   = 500             # exploration env 用 500，minigrid 用 50
    log_interval     = 50
    gamma            = 0.99
    lam              = 0.95
    clip             = 0.2
    value_coef       = 0.5
    entropy_coef     = 0.01
    lr               = 3e-4
    ppo_epochs       = 4
    grad_norm        = 0.5

# ======================
# Actor
# ======================
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),     nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ======================
# Critic (centralized)
# ======================
class ValueNetwork(nn.Module):
    def __init__(self, global_obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


# ======================
# Buffer
# ======================
class Buffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs       = []
        self.global_obs = []
        self.actions   = []
        self.log_probs = []
        self.rewards   = []
        self.values    = []
        self.dones     = []


# ======================
# GAE
# ======================
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        next_val = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae   = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    adv = torch.tensor(advantages, dtype=torch.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv


# ======================
# PPO Loss
# ======================
def ppo_loss(old_log_probs, new_log_probs, advantages, values, returns,
             clip=0.2, value_coef=0.5, entropy_coef=0.01):
    ratio   = torch.exp(new_log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip)

    policy_loss  = -torch.min(ratio * advantages, clipped * advantages).mean()
    value_loss   = ((values - returns) ** 2).mean()
    entropy_loss = -new_log_probs.mean()

    total = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
    return total, policy_loss, value_loss, entropy_loss


# ======================
# MAPPO
# ======================
class MAPPO:
    def __init__(self, obs_dim, action_dim, global_obs_dim, agents):
        self.agents = agents

        self.policies = {
            a: PolicyNetwork(obs_dim, action_dim).to(device)
            for a in agents
        }
        self.critics = {
            a: ValueNetwork(global_obs_dim).to(device)
            for a in agents
        }
        self.optim = {
            a: torch.optim.Adam(
                list(self.policies[a].parameters()) +
                list(self.critics[a].parameters()),
                lr=3e-4
            )
            for a in agents
        }
        self.buffers = {a: Buffer() for a in agents}

    def select_action(self, agent, obs):
        obs_t  = torch.FloatTensor(obs).to(device)
        logits = self.policies[agent](obs_t)
        dist   = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).detach()

    def value(self, agent, global_obs):
        g = torch.FloatTensor(global_obs).to(device)
        return self.critics[agent](g).squeeze().detach()


# ======================
# Training Loop
# ======================
def train(env, mappo: MAPPO, cfg: Config):
    """
    Works with both ExplorationEnv and MiniGridMAWrapper.
    Both expose the same dict-based API:
        obs  = env.reset()          -> {agent: np.array}
        obs, rewards, dones, infos = env.step(actions)
    """
    os.makedirs(f"runs/{cfg.env_name}", exist_ok=True)
    writer = SummaryWriter(f"runs/{cfg.env_name}/{cfg.experiment_name}")

    agents = mappo.agents

    for episode in range(cfg.total_episodes):

        obs            = env.reset()
        episode_reward = {a: 0.0 for a in agents}

        for step in range(cfg.episode_length):

            # ---- build global obs ----
            global_obs = np.concatenate([obs[a] for a in agents])

            actions = {}
            for a in agents:
                action, log_prob = mappo.select_action(a, obs[a])
                value            = mappo.value(a, global_obs)

                buf = mappo.buffers[a]
                buf.obs.append(obs[a])
                buf.global_obs.append(global_obs.copy())
                buf.actions.append(action)
                buf.log_probs.append(log_prob)
                buf.values.append(value.item())
                # reward from *previous* step stored here; updated below
                buf.rewards.append(env.last_reward.get(a, 0.0))
                buf.dones.append(0.0)

                actions[a] = action

            next_obs, rewards, dones, _ = env.step(actions)

            # overwrite the reward slot we just pushed with the real reward
            for a in agents:
                mappo.buffers[a].rewards[-1] = rewards[a]
                mappo.buffers[a].dones[-1]   = float(dones[a])
                episode_reward[a] += rewards[a]

            obs = next_obs

            if all(dones.values()):
                break

        # ---- UPDATE ----
        ep_total_loss  = []
        ep_policy_loss = []
        ep_value_loss  = []
        ep_ent_loss    = []

        for a in agents:
            buf = mappo.buffers[a]
            if len(buf.rewards) == 0:
                buf.clear()
                continue

            adv     = compute_gae(buf.rewards, buf.values, buf.dones,
                                   cfg.gamma, cfg.lam).to(device)
            returns = adv + torch.tensor(buf.values, dtype=torch.float32).to(device)

            obs_batch   = torch.tensor(np.array(buf.obs),        dtype=torch.float32).to(device)
            g_obs_batch = torch.tensor(np.array(buf.global_obs), dtype=torch.float32).to(device)
            act_batch   = torch.tensor(buf.actions,               dtype=torch.long).to(device)
            old_logp    = torch.stack(buf.log_probs).to(device)

            for _ in range(cfg.ppo_epochs):
                logits   = mappo.policies[a](obs_batch)
                dist     = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(act_batch)

                new_vals = mappo.critics[a](g_obs_batch).squeeze()

                loss, pl, vl, ent = ppo_loss(
                    old_logp, new_logp, adv, new_vals, returns,
                    cfg.clip, cfg.value_coef, cfg.entropy_coef
                )

                mappo.optim[a].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(mappo.policies[a].parameters()) +
                    list(mappo.critics[a].parameters()),
                    cfg.grad_norm
                )
                mappo.optim[a].step()

                ep_total_loss.append(loss.item())
                ep_policy_loss.append(pl.item())
                ep_value_loss.append(vl.item())
                ep_ent_loss.append(ent.item())

            buf.clear()

        avg_reward   = np.mean(list(episode_reward.values()))
        episode_len  = step + 1

        # exploration ratio (only available for ExplorationEnv)
        explore_ratio = None
        if hasattr(env, "explored") and hasattr(env, "total_explorable"):
            explore_ratio = float(np.sum(env.explored)) / max(env.total_explorable, 1)

        if episode % cfg.log_interval == 0:
            writer.add_scalar("Training/Avg_Reward", avg_reward,  episode)
            writer.add_scalar("Training/Ep_Length",  episode_len, episode)
            if explore_ratio is not None:
                writer.add_scalar("Training/Explore_Ratio", explore_ratio, episode)

            if len(ep_total_loss) > 0:
                writer.add_scalar("Loss/Total_Loss",   np.mean(ep_total_loss),  episode)
                writer.add_scalar("Loss/Policy_Loss",  np.mean(ep_policy_loss), episode)
                writer.add_scalar("Loss/Value_Loss",   np.mean(ep_value_loss),  episode)
                writer.add_scalar("Loss/Entropy",      np.mean(ep_ent_loss),    episode)

            msg = f"[EP {episode:5d}] reward={avg_reward:.4f}  steps={episode_len}"
            if len(ep_total_loss) > 0:
                msg += f"  loss={np.mean(ep_total_loss):.4f}"
            if explore_ratio is not None:
                msg += f"  explored={explore_ratio*100:.1f}%"
            print(msg)

    writer.close()