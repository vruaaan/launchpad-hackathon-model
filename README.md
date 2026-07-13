# launchpad-hackathon-rl-model

Toy research repo for **learning to generate SQL (as a sequence of high-level “blocks”)** via:

- **Reinforcement Learning (DQN)** in a Gymnasium environment that enforces a SQL-ish grammar and supports nested subqueries.
- **Autoregressive modeling (LSTM / Transformer)** over the same block tokens (next-token prediction, perplexity, sampling).

This was built for a hackathon-style workflow: notebooks for iteration + a couple of small Python entrypoints.

## Repo layout

- `rl/`
  - `rl/classes/env.py`: `SQLEnvAdvanced` — grammar + reward shaping + nested subquery stack
  - `rl/classes/adjenv.py`: same environment plus **legal action masking utilities**
  - `rl/classes/dqn.py`: baseline discrete-action DQN
  - `rl/custom/customdqn.py`: masked Double-DQN variant with an embedding over the prefix tokens
  - `rl/scripts/diagnose.py`: helper to debug why a trained agent gets stuck (illegal moves / wrong stop flag)
  - `rl/*.ipynb`: training + finetuning notebooks, plus saved `.pth` checkpoints
- `autoregression/`
  - `autoregression/data/*.py`: problem bank + dataset/tokenization utilities
  - `autoregression/models/lstm.py`: LSTM language model wrapper (`RNNAgent`)
  - `autoregression/models/transformer.py`: GPT-style decoder-only Transformer wrapper
  - `autoregression/models/trainlstm.py`: train + eval + sample entrypoint
  - `autoregression/models/traintrans.py`: train + eval + sample entrypoint
  - `autoregression/*.ipynb`: exploration notebooks

## Setup

Create a venv and install deps:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quickstart: RL environment + DQN

The environment is a **block-sequence builder**. Each step selects:

- a `block_idx` (which grammar token to place), and
- a `stop_flag` (whether to “close” the current frame / end the episode).

Actions are encoded as a single discrete integer:

```
flat_action = block_idx * 2 + stop_flag
```

Minimal “train then evaluate” example:

```bash
python -c "from rl.classes.env import SQLEnvAdvanced; from rl.classes.dqn import DQNAgent; env = SQLEnvAdvanced(render_mode=None); agent = DQNAgent(env=env); agent.train(total_timesteps=10_000, learning_starts=1_000, log_every=10, render=False); agent.evaluate(n_episodes=50, deterministic=True); agent.save('dqn_sql_blocks.pth')"
```

If you want legality masking + the two-headed (block, stop) network, use:

```bash
python -c "from rl.classes.adjenv import SQLEnvAdvanced; from rl.custom.customdqn import DQNAgent; env = SQLEnvAdvanced(render_mode=None); agent = DQNAgent(env=env); agent.train(total_timesteps=10_000, learning_starts=1_000, log_every=10, render=False); agent.save('custom_dqn_sql_blocks.pth')"
```

## Debugging an RL policy

`rl/scripts/diagnose.py` is meant to be run **after** you have an agent + env wired up and want to understand:

- are mistakes mostly *illegal* grammar moves vs *legal-but-wrong* moves?
- is the agent failing because of the `stop_flag`?

Typical use:

1. Edit `rl/scripts/diagnose.py` to import/create your `agent` and `env` (the file has a placeholder section).
2. Run:

```bash
python rl/scripts/diagnose.py
```

## Quickstart: autoregressive models

The autoregression side turns each generated problem into a flat token stream:

- tokens are SQL “block types” plus special tokens: `<PAD>`, `<SOS>`, `<EOS>`
- nested subqueries are depth-first spliced between `SUBQUERY_START` / `SUBQUERY_END`

Train/evaluate/sample an LSTM:

```bash
python autoregression/models/trainlstm.py
```

Train/evaluate/sample a Transformer:

```bash
python autoregression/models/traintrans.py
```

Both scripts print validation perplexity and a sampled sequence, and save a `.pth` checkpoint into the working directory.

## Notes / gotchas

- This repo uses **Gymnasium** (not classic `gym`).
- Many experiments live in notebooks under `rl/` and `autoregression/`; use those if you want the “full” training runs and plots.
- If you try `python -m ...` and imports fail, run from the repo root so `rl/` and `autoregression/` are on `sys.path`.
