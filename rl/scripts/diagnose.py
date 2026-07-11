"""
Run this BEFORE touching any reward values.
It tells you exactly why the agent is stuck at -75.
"""
import numpy as np
import collections
from functools import partial

try:
    # When running as a package module: `python -m rl.scripts.diagnose`
    from rl.classes.env import IDX_TO_BLOCK, legal_next_blocks as lnb
except ModuleNotFoundError:
    # When running with cwd=`rl`: `python scripts/diagnose.py`
    from classes.env import IDX_TO_BLOCK, legal_next_blocks as lnb

# ── paste your imports here ───────────────────────────────────────────────────
# from subqueries_v4 import (
#     SQLEnvAdvanced, Frame, PROBLEM_BANK,
#     BLOCK_TO_IDX, IDX_TO_BLOCK, N_BLOCKS,
#     MAX_SEQUENCE_LEN, MAX_DEPTH, legal_next_blocks
# )
# from calc_reward import calc_reward
# from wm_dqn_agent import WorldModelDQNAgent
# ─────────────────────────────────────────────────────────────────────────────


def diagnose(agent, env, n_episodes=200):
    """
    Runs the trained agent deterministically and breaks down exactly
    what went wrong on every step of every episode.
    """
    counters = collections.Counter()
    reward_breakdown = collections.defaultdict(list)
    episode_lengths  = []
    episode_rewards  = []
    stop_flag_stats  = {"correct": 0, "wrong": 0}

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_steps  = 0
        done      = False

        while not done:
            action = agent.act(obs, deterministic=True)
            block_idx, stop_flag = divmod(int(action), 2)
            block = IDX_TO_BLOCK[block_idx]

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps  += 1

            # ── classify this step ──────────────────────────────────────────
            frame      = env.state.stack[-1]
            target     = frame.target["blocks"]
            pos        = len(frame.built_seq) - 1  # after step, built_seq already updated if correct
            expected   = target[pos] if not terminated and pos < len(target) else None

            # was the block correct?
            block_correct = (expected is not None and block == expected) or (
                terminated and reward > 0
            )

            # re-read built_seq before this step to check legality
            # (approximate — built_seq may have been updated)
            syntax_ok = block in lnb(frame.built_seq[:-1] if block_correct else frame.built_seq)

            if block_correct:
                counters["correct_block"] += 1
                # check stop_flag appropriateness
                true_done = (len(frame.built_seq) == len(target)) if not terminated else True
                if stop_flag == int(true_done):
                    stop_flag_stats["correct"] += 1
                else:
                    stop_flag_stats["wrong"]   += 1
                    counters["wrong_stop"] += 1
            elif syntax_ok:
                counters["wrong_legal"] += 1
            else:
                counters["wrong_illegal"] += 1

            reward_breakdown["per_step"].append(reward)
            obs = next_obs

        episode_lengths.append(ep_steps)
        episode_rewards.append(ep_reward)

    # ── print report ─────────────────────────────────────────────────────────
    total_steps = sum(counters.values())
    print("=" * 60)
    print(f"DIAGNOSIS REPORT  ({n_episodes} episodes)")
    print("=" * 60)

    print(f"\nReward summary:")
    print(f"  Mean episode reward : {np.mean(episode_rewards):+.2f}")
    print(f"  Std                 : {np.std(episode_rewards):.2f}")
    print(f"  Min / Max           : {np.min(episode_rewards):+.2f} / {np.max(episode_rewards):+.2f}")
    print(f"  Mean episode length : {np.mean(episode_lengths):.1f} steps")

    print(f"\nStep breakdown (total {total_steps} steps):")
    for key, count in sorted(counters.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(total_steps, 1)
        print(f"  {key:<20} {count:>6}  ({pct:.1f}%)")

    print(f"\nStop flag (on correct blocks only):")
    total_correct = stop_flag_stats["correct"] + stop_flag_stats["wrong"]
    for k, v in stop_flag_stats.items():
        pct = 100 * v / max(total_correct, 1)
        print(f"  {k:<10} {v:>6}  ({pct:.1f}%)")

    print(f"\nPer-step reward distribution:")
    rewards_arr = np.array(reward_breakdown["per_step"])
    for val, label in [(-2.0, "illegal block"), (-1.3, "wrong+legal+wrong_stop"),
                       (-0.3, "wrong+legal"), (0.0, "wrong+legal+no_stop"),
                       (1.0, "correct"), (2.0, "correct+good_stop"),
                       (6.0, "correct+stop+subq_pop"), (7.0, "correct+stop+terminate")]:
        count = int(np.sum(np.abs(rewards_arr - val) < 0.05))
        if count > 0:
            print(f"  reward ~{val:+.1f}  → {count} steps")

    print()

    # ── root cause interpretation ─────────────────────────────────────────────
    pct_illegal = 100 * counters["wrong_illegal"] / max(total_steps, 1)
    pct_legal   = 100 * counters["wrong_legal"]   / max(total_steps, 1)
    pct_correct = 100 * counters["correct_block"] / max(total_steps, 1)
    pct_wrong_stop = 100 * counters["wrong_stop"] / max(total_steps, 1)

    print("ROOT CAUSE ANALYSIS:")
    if pct_illegal > 40:
        print(f"  ⚠  {pct_illegal:.0f}% illegal blocks → agent hasn't learned grammar at all.")
        print(f"     Fix: reduce action space, increase learning_starts, check obs encoding.")
    if pct_legal > 40:
        print(f"  ⚠  {pct_legal:.0f}% wrong-but-legal → agent knows grammar but not target.")
        print(f"     Fix: increase required_blocks weight in obs, add shaping reward for")
        print(f"          matching expected block type.")
    if pct_correct > 60 and pct_wrong_stop > 30:
        print(f"  ⚠  {pct_wrong_stop:.0f}% wrong stop_flag on correct blocks → agent can't")
        print(f"     learn when to terminate. Fix: decouple stop_flag penalty or add")
        print(f"     a dedicated stop signal in the obs.")
    if pct_correct > 70:
        print(f"  ✓  Agent mostly places correct blocks ({pct_correct:.0f}%).")
        print(f"     Mean reward is low due to stop_flag or subquery framing issues.")
    if np.mean(episode_lengths) >= 38:
        print(f"  ⚠  Episodes hit MAX_TOTAL_STEPS ({np.mean(episode_lengths):.1f} avg).")
        print(f"     Agent is not terminating — stop_flag=1 is never being triggered.")

    print("=" * 60)
    return counters, episode_rewards


# ── run it ───────────────────────────────────────────────────────────────────
# env   = SQLEnvAdvanced()
# counters, rewards = diagnose(agent, env, n_episodes=200)
