import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

try:
    from .agents import Agent
    from .buffers import ReplayBuffer
except ImportError:
    from agents import Agent
    from buffers import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)


class DQNAgent(Agent):
    """DQN agent for environments with a Box observation space and a
    Discrete action space (e.g. SQLBlockEnv).

    Note: this does NOT call Agent.__init__, because that method hard-
    requires a continuous Box action space (action_low/action_high,
    rescale/clip helpers, etc.) which don't apply to discrete actions.
    Instead it replicates the parts of Agent's setup that are generic
    (name, device, env, obs_shape/obs_dim) and adds its own discrete-
    action bookkeeping (n_actions, epsilon-greedy schedule). It still
    inherits from Agent so play(), evaluate(), save(), load(),
    flatten_obs(), and obs_tensor() work unchanged, since none of those
    depend on the action space being continuous.
    """

    def __init__(self,
        env: gym.Env = None,
        name: str = "bot",
        gamma=0.99,
        tau=0.005,
        lr=1e-3,
        batch_size=64,
        buffer_capacity=100_000,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=50_000,
        target_update_every=1,  # env steps between soft target-network updates
    ):
        if env is None:
            raise ValueError("An environment must be provided, e.g. DQNAgent(env=my_env).")
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("Only Box observation spaces are supported.")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("DQNAgent requires a Discrete action space.")

        # --- generic setup (mirrors Agent.__init__, minus the Box-action bits) ---
        self.env = env
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_shape = self.env.observation_space.shape
        self.obs_dim = int(np.prod(self.obs_shape))

        # discrete-action bookkeeping (replaces action_low/action_high etc.)
        self.n_actions = int(self.env.action_space.n)
        self.action_shape = ()  # a discrete action is a scalar, not a vector

        # --- networks ---
        self.q_net = QNetwork(self.obs_dim, self.n_actions).to(self.device)
        self.tgt_q_net = copy.deepcopy(self.q_net)
        for param in self.tgt_q_net.parameters():
            param.requires_grad = False

        self.q_opt = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.target_update_every = target_update_every

        self.total_updates = 0
        self.total_env_steps = 0

    def epsilon(self):
        """Linear decay from eps_start to eps_end over eps_decay_steps env steps."""
        frac = min(1.0, self.total_env_steps / max(1, self.eps_decay_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, obs, deterministic=False):
        if not deterministic and random.random() < self.epsilon():
            return int(self.env.action_space.sample())

        obs_t = self.obs_tensor(obs)  # inherited from Agent
        self.q_net.eval()
        with torch.no_grad():
            q_values = self.q_net(obs_t)
        self.q_net.train()
        return int(torch.argmax(q_values, dim=-1).item())

    def dqn_update(self):
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)
        obs = obs.to(self.device)
        actions = actions.to(self.device).long().view(-1, 1)  # discrete action indices
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)

        # Bellman target using the (frozen) target network
        with torch.no_grad():
            next_q_values = self.tgt_q_net(next_obs)
            max_next_q = next_q_values.max(dim=-1, keepdim=True).values
            target_q = rewards + self.gamma * (1.0 - dones) * max_next_q

        # Q(s, a) for the actions actually taken
        current_q = self.q_net(obs).gather(1, actions)
        loss = nn.functional.mse_loss(current_q, target_q)

        self.q_opt.zero_grad()
        loss.backward()
        self.q_opt.step()

        self.total_updates += 1
        if self.total_updates % self.target_update_every == 0:
            # Polyak/soft update, same style as TD3Agent's target updates
            for p_main, p_tgt in zip(self.q_net.parameters(), self.tgt_q_net.parameters()):
                p_tgt.data.mul_(1.0 - self.tau)
                p_tgt.data.add_(self.tau * p_main.data)

        return loss.item()

    def train(self, total_timesteps=100_000, learning_starts=1_000, log_every=20, render=True):
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_count = 0
        for timestep in range(1, total_timesteps + 1):
            self.total_env_steps = timestep
            if timestep < learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.act(obs, deterministic=False)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.replay_buffer.add(obs, action, reward, next_obs, float(terminated))
            episode_reward += reward
            obs = next_obs
            if done:
                episode_count += 1
                if episode_count % log_every == 0 and render:
                    print(
                        f"Timestep {timestep:>7} | Episode {episode_count:>4} | "
                        f"Reward: {episode_reward:>8.2f} | Epsilon: {self.epsilon():.3f}"
                    )
                episode_reward = 0.0
                obs, _ = self.env.reset()
            if timestep >= learning_starts and len(self.replay_buffer) >= self.batch_size:
                self.dqn_update()
        print("Training complete.")
        return self
