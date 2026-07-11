import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym


class Agent:
    """Base class for continuous-control Gymnasium agents.
    The PPO, TD3, and SAC implementations in this package expect environments
    with Box observation and Box action spaces. Observations and actions are
    flattened for neural-network input, then actions are reshaped back to the
    environment's native action shape before stepping.
    """

    def __init__(self, env: gym.Env, name: str = "bot"):
        if env is None:
            raise ValueError("An environment must be provided, e.g. PPOAgent(env=my_env).")
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("Only Box observation spaces are supported.")
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("Only continuous Box action spaces are supported.")

        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env

        self.obs_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape
        self.obs_dim = int(np.prod(self.obs_shape))
        self.action_dim = int(np.prod(self.action_shape))

        self.action_low_np = self.env.action_space.low.astype(np.float32).reshape(-1)
        self.action_high_np = self.env.action_space.high.astype(np.float32).reshape(-1)
        self.action_low = torch.tensor(self.action_low_np, dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor(self.action_high_np, dtype=torch.float32, device=self.device)

    def flatten_obs(self, obs):
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def obs_tensor(self, obs):
        return torch.tensor(self.flatten_obs(obs), dtype=torch.float32, device=self.device).unsqueeze(0)

    def format_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.action_shape)
        return action

    def clip_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, self.action_low_np, self.action_high_np)
        return action.reshape(self.action_shape).astype(np.float32)

    def rescale(self, raw_action):
        raw_t = torch.as_tensor(raw_action, dtype=torch.float32, device=self.device)
        return self.action_low + (raw_t + 1.0) * 0.5 * (self.action_high - self.action_low)

    def play(self, render=False, deterministic=True, verbose=True):
        obs, _ = self.env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        trajectory = []

        while not done:
            action = self.act(obs, deterministic=deterministic)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            trajectory.append(
                {
                    "obs": obs,
                    "action": action,
                    "reward": float(reward),
                    "next_obs": next_obs,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": info,
                }
            )
            if verbose:
                print(f"Step {steps + 1}: Action: {action}, Reward: {reward:.2f}")

            total_reward += reward
            steps += 1
            obs = next_obs

            if render and hasattr(self.env, "render"):
                self.env.render()

        result = {"total_reward": total_reward, "steps": steps, "trajectory": trajectory}
        if verbose:
            print(f"Episode finished - Reward: {total_reward:.2f}, Steps: {steps}")
        return result

    def evaluate(self, n_episodes=100, env_factory=None, deterministic=True):
        rewards, steps_list = [], []

        for _ in range(n_episodes):
            env = env_factory() if env_factory is not None else self.env
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                action = self.act(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated

            rewards.append(total_reward)
            steps_list.append(steps)

        print(f"Episodes: {n_episodes}")
        print(f"Mean reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
        print(f"Reward range: [{np.min(rewards):.2f}, {np.max(rewards):.2f}]")
        print(f"Mean steps: {np.mean(steps_list):.2f}")
        return {"rewards": rewards, "steps": steps_list}

    def save(self, path=None):
        if path is None:
            path = f"{self.name}_{self.__class__.__name__}.pth"

        payload = {
            "class_name": self.__class__.__name__,
            "name": self.name,
            "modules": {
                attr: module.state_dict()
                for attr, module in self.__dict__.items()
                if isinstance(module, nn.Module)
            },
            "optimizer_states": {
                attr: obj.state_dict()
                for attr, obj in self.__dict__.items()
                if hasattr(obj, "state_dict") and attr.endswith("_opt")
            },
        }
        torch.save(payload, path)
        return path

    def load(self, path, map_location=None):
        payload = torch.load(path, map_location=map_location if map_location is not None else self.device)

        for attr, state_dict in payload.get("modules", {}).items():
            module = getattr(self, attr, None)
            if isinstance(module, nn.Module):
                module.load_state_dict(state_dict)

        for attr, state_dict in payload.get("optimizer_states", {}).items():
            optimizer = getattr(self, attr, None)
            if hasattr(optimizer, "load_state_dict"):
                optimizer.load_state_dict(state_dict)

        return self
