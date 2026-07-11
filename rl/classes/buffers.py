import numpy as np
import torch


class RolloutBuffer:
    def __init__(self):
        self.clear()
    def clear(self):
        self.obs, self.actions, self.log_probs = [], [], []
        self.rewards, self.values, self.dones = [], [], []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns(self, last_value, gamma=0.99, gae_lambda=0.95):
        advantages = []
        gae = 0.0
        values = self.values + [last_value]
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        return advantages, returns

    def to_tensors(self, advantages, returns, device):
        obs = torch.tensor(np.array(self.obs), dtype=torch.float32, device=device).view(len(self.obs), -1)
        actions = torch.stack(self.actions).to(device).view(len(self.actions), -1)
        log_probs = torch.stack(self.log_probs).to(device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return obs, actions, log_probs, advantages, returns


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, obs, action, reward, next_obs, done):
        transition = (obs, action, reward, next_obs, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.tensor(np.array(obs), dtype=torch.float32).view(batch_size, -1),
            torch.tensor(np.array(actions), dtype=torch.float32).view(batch_size, -1),
            torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_obs), dtype=torch.float32).view(batch_size, -1),
            torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)
