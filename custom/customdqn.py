import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

try:
    from classes.agents import Agent
    from classes.buffers import ReplayBuffer
    from classes.adjenv import N_BLOCKS, MAX_SEQUENCE_LEN, legal_mask_from_obs_batch
except ImportError:
    from classes.agents import Agent
    from classes.buffers import ReplayBuffer
    from classes.adjenv import N_BLOCKS, MAX_SEQUENCE_LEN, legal_mask_from_obs_batch


class QNetwork(nn.Module):
    """Two-headed Q-network for the (block_choice, stop_flag) action space.

    Splitting the flat 56-way action into a block head (over N_BLOCKS) and a
    binary stop head decouples "which block" from "should I stop", instead of
    forcing the network to guess both jointly via block_idx * 2 + stop_flag.

    The built-sequence prefix (the first MAX_SEQUENCE_LEN entries of obs) is a
    sequence of categorical block-type tokens, not a numeric quantity, so it
    goes through an nn.Embedding instead of being fed in as a raw scalar index
    (which would wrongly imply e.g. block 5 is "closer" to block 6 than 19).
    """

    def __init__(self, obs_dim, n_blocks=N_BLOCKS, seq_len=MAX_SEQUENCE_LEN, embed_dim=8, hidden=64):
        super().__init__()
        self.seq_len = seq_len
        self.n_blocks = n_blocks
        # +1 in vocab size for the padding token (0)
        self.embed = nn.Embedding(n_blocks + 1, embed_dim)
        rest_dim = obs_dim - seq_len  # step_count + required-blocks vector + depth
        in_dim = seq_len * embed_dim + rest_dim
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.block_head = nn.Linear(hidden, n_blocks)
        self.stop_head = nn.Linear(hidden, 2)

    def forward(self, obs):
        seq_tokens = obs[:, :self.seq_len].long()
        rest = obs[:, self.seq_len:]
        emb = self.embed(seq_tokens)              # (B, seq_len, embed_dim)
        emb = emb.reshape(emb.size(0), -1)         # (B, seq_len * embed_dim)
        x = torch.cat([emb, rest], dim=-1)
        feat = self.body(x)
        return self.block_head(feat), self.stop_head(feat)


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
        self.q_net = QNetwork(self.obs_dim, hidden=128, embed_dim=16).to(self.device)
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

    def masked_random_action(self):
        """Sample a uniformly random *legal* action instead of
        env.action_space.sample(), which samples over all 56 flat actions
        regardless of grammar validity. With ~90% of actions illegal at a
        typical state, unmasked random exploration fills the replay buffer
        with mostly illegal-move transitions.

        Edge case: some grammar states have ZERO legal next blocks (e.g. the
        built sequence already matches a complete query but stop_flag wasn't
        set correctly on the final step, so the episode didn't terminate).
        legal_next_blocks() returns an empty set there, so np.random.choice
        on the empty legal-blocks array would raise ValueError. In that case
        there's no "correct" move left anyway, so fall back to a uniform
        choice over all blocks -- matching pre-masking behavior -- and let
        the environment's own illegal-block penalty handle it.
        """
        legal_mask = self.env.legal_action_mask()
        legal_blocks = np.flatnonzero(legal_mask)
        if legal_blocks.size == 0:
            block_idx = int(np.random.randint(N_BLOCKS))
        else:
            block_idx = int(np.random.choice(legal_blocks))
        stop_flag = random.randint(0, 1)
        return block_idx * 2 + stop_flag

    def act(self, obs, deterministic=False):
        legal_mask = self.env.legal_action_mask()  # (N_BLOCKS,) bool

        if not deterministic and random.random() < self.epsilon():
            return self.masked_random_action()

        obs_t = self.obs_tensor(obs)  # inherited from Agent
        self.q_net.eval()
        with torch.no_grad():
            block_q, stop_q = self.q_net(obs_t)
            mask_t = torch.tensor(legal_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            if mask_t.any():
                block_q = block_q.masked_fill(~mask_t, float("-inf"))
            # else: no block is grammatically legal here (dead-end state) --
            # leave block_q unmasked rather than forcing every value to -inf,
            # which would make argmax silently and deterministically pick
            # block 0 every time instead of reflecting the net's own values.
            block_idx = int(torch.argmax(block_q, dim=-1).item())
            stop_flag = int(torch.argmax(stop_q, dim=-1).item())
        self.q_net.train()
        return block_idx * 2 + stop_flag

    def dqn_update(self):
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)
        obs = obs.to(self.device)
        actions = actions.to(self.device).long().view(-1)  # flat action = block_idx * 2 + stop_flag
        rewards = rewards.to(self.device).view(-1, 1)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device).view(-1, 1)

        block_actions = (actions // 2).view(-1, 1)
        stop_actions = (actions % 2).view(-1, 1)

        # Reconstruct the legal-block mask for each next_obs so the target
        # can't bootstrap off Q-values for illegal moves the agent would
        # never actually be allowed to take.
        next_legal_mask = torch.tensor(
            legal_mask_from_obs_batch(next_obs), dtype=torch.bool, device=self.device
        )

        with torch.no_grad():
            # Double DQN: pick the next action with the online net (masked),
            # but evaluate its value with the frozen target net, curbing the
            # overestimation bias vanilla DQN has with this wide a reward range.
            next_block_q_online, next_stop_q_online = self.q_net(next_obs)
            # Some states have zero legal continuations (e.g. built_seq already
            # matches a complete query but stop wasn't set correctly on the
            # final step). Masking those rows to all -inf would make argmax
            # deterministically pick block 0 for every one of them instead of
            # reflecting the net's own values, so only mask rows that actually
            # have at least one legal block.
            has_legal = next_legal_mask.any(dim=-1, keepdim=True)
            block_mask = ~next_legal_mask & has_legal
            next_block_q_online = next_block_q_online.masked_fill(block_mask, float("-inf"))
            next_block_action = next_block_q_online.argmax(dim=-1, keepdim=True)
            next_stop_action = next_stop_q_online.argmax(dim=-1, keepdim=True)

            next_block_q_tgt, next_stop_q_tgt = self.tgt_q_net(next_obs)
            next_block_q = next_block_q_tgt.gather(1, next_block_action)
            next_stop_q = next_stop_q_tgt.gather(1, next_stop_action)
            max_next_q = next_block_q + next_stop_q  # joint value of the (block, stop) pair
            target_q = rewards + self.gamma * (1.0 - dones) * max_next_q

        block_q, stop_q = self.q_net(obs)
        current_q = block_q.gather(1, block_actions) + stop_q.gather(1, stop_actions)
        loss = nn.functional.mse_loss(current_q, target_q)

        self.q_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=5.0)
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
                action = self.masked_random_action()
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
