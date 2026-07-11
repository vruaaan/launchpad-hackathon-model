"""
Converts SCALED_PROBLEM_BANK (a tree-shaped problem bank, built for the RL
env's stack/Frame mechanics) into flat token sequences suitable for
autoregressive (LSTM / Transformer) training.

The one representational change from the RL env: subqueries are INLINED.
In the RL env, a parent's `blocks` list holds SUBQUERY_START immediately
followed by SUBQUERY_END, with the nested query's tokens living in a
separate pushed Frame -- that's the right shape for a stack-based agent,
but wrong for a sequence model, which should see one continuous token
stream that reads the way real nested SQL actually reads:

    RL env (parent blocks):  ... IN SUBQUERY_START SUBQUERY_END
    RL env (child frame):    SELECT COLUMN FROM TABLE WHERE ...

    Flattened for seq model: ... IN SUBQUERY_START SELECT COLUMN FROM TABLE
                              WHERE ... SUBQUERY_END

`flatten(problem)` does this splice recursively, so arbitrarily nested
subqueries become one flat list.

No stop-flag is needed here (unlike the RL action space) -- teacher-forced
next-token prediction with a trailing <EOS> is the standard way to teach a
sequence model when to stop, and there's no "act on a wrong belief"
dynamic to worry about outside of RL.
"""

import random

import torch
from torch.nn.utils.rnn import pad_sequence

from problembank import PROBLEM_BANK
from problembank import BLOCK_TYPES

# ---------------------------------------------------------------------------
# Vocabulary: block types + three special tokens.
# ---------------------------------------------------------------------------
SPECIALS = ["<PAD>", "<SOS>", "<EOS>"]
VOCAB = SPECIALS + list(BLOCK_TYPES)
TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}
ID_TO_TOKEN = {i: tok for tok, i in TOKEN_TO_ID.items()}
PAD_ID, SOS_ID, EOS_ID = TOKEN_TO_ID["<PAD>"], TOKEN_TO_ID["<SOS>"], TOKEN_TO_ID["<EOS>"]
VOCAB_SIZE = len(VOCAB)


def flatten(problem):
    """DFS-serialize a (possibly nested) problem into one flat token list,
    splicing each subquery's tokens in between its SUBQUERY_START/END."""
    out = []
    blocks = problem["blocks"]
    subqueries = problem.get("subqueries", {})
    for i, tok in enumerate(blocks):
        out.append(tok)
        if tok == "SUBQUERY_START":
            out.extend(flatten(subqueries[i]))
    return out


def build_dataset():
    """Returns a list of token-ID sequences: <SOS> ... tokens ... <EOS>."""
    sequences = []
    for problem in PROBLEM_BANK:
        flat = flatten(problem)
        ids = [SOS_ID] + [TOKEN_TO_ID[t] for t in flat] + [EOS_ID]
        sequences.append(torch.tensor(ids, dtype=torch.long))
    return sequences


class SQLSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate(batch):
    """Pad a batch of variable-length sequences, and build the teacher-
    forcing (input, target) pair: input is seq[:-1], target is seq[1:]."""
    padded = pad_sequence(batch, batch_first=True, padding_value=PAD_ID)
    input_ids = padded[:, :-1]
    target_ids = padded[:, 1:]
    return input_ids, target_ids


if __name__ == "__main__":
    print(f"Vocab size: {VOCAB_SIZE}  ({VOCAB})\n")

    sequences = build_dataset()
    lengths = [len(s) for s in sequences]
    print(f"{len(sequences)} sequences | min len {min(lengths)}, "
          f"max len {max(lengths)}, mean len {sum(lengths)/len(lengths):.1f}\n")

    # Show one flattened example with a subquery, to confirm inlining worked.
    example = next(p for p in PROBLEM_BANK if p.get("subqueries"))
    print("Example problem with a subquery:")
    print(f"  RL-env blocks (parent only): {example['blocks']}")
    print(f"  RL-env nested Frame target:  "
          f"{list(example['subqueries'].values())[0]['blocks']}")
    print(f"  Flattened for seq model:     {flatten(example)}\n")

    random.seed(0)
    torch.manual_seed(0)
    dataset = SQLSequenceDataset(sequences)
    n_val = max(1, int(0.05 * len(dataset)))
    train_set, val_set = torch.utils.data.random_split(
        dataset, [len(dataset) - n_val, n_val]
    )
    loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=collate)
    input_ids, target_ids = next(iter(loader))
    print(f"One training batch: input_ids {tuple(input_ids.shape)}, "
          f"target_ids {tuple(target_ids.shape)}")
    print(f"train/val split: {len(train_set)}/{len(val_set)}")
