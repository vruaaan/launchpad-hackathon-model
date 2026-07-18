import random

import torch
from torch.nn.utils.rnn import pad_sequence

from autoregression.data.problembank import PROBLEM_BANK
from autoregression.data.problembank import BLOCK_TYPES

# ---------------------------------------------------------------------------
# Vocabulary: block types + three special tokens.
# ---------------------------------------------------------------------------
SPECIALS = ["<PAD>", "<SOS>", "<EOS>", "<CURSOR>"]
TOKENS = SPECIALS + list(BLOCK_TYPES)
N_TOKENS = len(TOKENS)
TOKEN_TO_ID = {tok: i for i, tok in enumerate(TOKENS)}
ID_TO_TOKEN = {i: tok for tok, i in TOKEN_TO_ID.items()}
PAD_ID = TOKEN_TO_ID["<PAD>"]
SOS_ID = TOKEN_TO_ID["<SOS>"]
EOS_ID = TOKEN_TO_ID["<EOS>"]
CURSOR_ID = TOKEN_TO_ID["<CURSOR>"]


def flatten(problem):
    """DFS-serialize a (possibly nested) problem into one flat token list,
    splicing each subquery's tokens in between its SUBQUERY_START/END."""
    out = []
    blocks = problem["blocks"]
    subqueries = problem.get("subqueries", {})
    for i, tok in enumerate(blocks):
        out.append(tok)
        if tok == "SUBQUERY_START":
            # For intentionally incorrect/incomplete problems, a SUBQUERY_START
            # token may appear without a corresponding nested node.
            nested = subqueries.get(i)
            if nested is not None:
                out.extend(flatten(nested))
    return out

def normalise_data(problem):
    flat = flatten(problem)
    ids = [SOS_ID] + [TOKEN_TO_ID[t] for t in flat] + [EOS_ID]
    return torch.tensor(ids, dtype=torch.long)

def build_dataset():
    sequences = []
    for problem in PROBLEM_BANK:
        sequences.append(normalise_data(problem))
    return sequences


class SQLSeqDataset(torch.utils.data.Dataset):
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



tensorlst = build_dataset()
random.seed(0)
torch.manual_seed(0)
dataset = SQLSeqDataset(tensorlst)
n_val = max(1, int(0.05 * len(dataset)))
train_set, val_set = torch.utils.data.random_split(
        dataset, [len(dataset) - n_val, n_val]
    )
loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=collate)
