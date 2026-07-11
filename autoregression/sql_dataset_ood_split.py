"""
Builds the autoregressive dataset with a STRUCTURAL held-out split instead
of a random one.

The held-out knob: WHERE_VARIANTS indices 6 and 7 -- the AND-chained and
OR-chained condition variants:
    6: WHERE COLUMN OPERATOR VALUE AND COLUMN OPERATOR VALUE
    7: WHERE COLUMN OPERATOR VALUE OR  COLUMN OPERATOR VALUE

Every problem built from those two variants is removed from the training
set entirely and placed in `ood_val` instead. Nothing in `train` ever
contains a WHERE clause with more than one condition. `interp_val` is a
random 5% slice of what's LEFT (i.e. drawn from the same knob-values the
model actually trained on) -- a normal, in-distribution validation set,
kept only as a baseline to compare against `ood_val`.

This directly tests one specific thing: can the model learn "a WHERE
clause condition can repeat, chained with AND/OR" as a general rule from
seeing it applied to individual conditions, or did it only memorize the
specific finite set of clause-combinations it was shown? A random split
(or k-fold over a random split) cannot answer that question, because
AND/OR-chaining would appear in both its train and val slices.
"""

import random

import torch

from problembank import (
    GROUPBY_VARIANTS, JOINS, LIMIT_VARIANTS, ORDERBY_VARIANTS,
    SELECT_ITEMS, SIMPLE_SUBQUERY_POOL, WHERE_VARIANTS, 
    assemble, make_problem
)

from sql_dataset_prep import EOS_ID, SOS_ID, TOKEN_TO_ID, flatten

# Which WHERE_VARIANTS indices are held out of training entirely.
HELD_OUT_WHERE_INDICES = {6, 7}  # the AND-chained and OR-chained variants


def _generate_flat_with_tags():
    """Re-run the same combinatorial sweep as SCALED_PROBLEM_BANK, but
    keep the knob INDICES used to build each problem, so we can filter on
    them afterward. (SCALED_PROBLEM_BANK itself discards this metadata.)"""
    tagged = []
    for si, select_item in enumerate(SELECT_ITEMS):
        for ji, join in enumerate(JOINS):
            for wi, where in enumerate(WHERE_VARIANTS):
                for gi, groupby in enumerate(GROUPBY_VARIANTS):
                    for oi, orderby in enumerate(ORDERBY_VARIANTS):
                        for li, limit in enumerate(LIMIT_VARIANTS):
                            blocks = _assemble(select_item, join, where, groupby, orderby, limit)
                            problem = make_problem(blocks)
                            tagged.append((problem, {"where": wi}))
    return tagged


def _generate_subquery_with_tags():
    """One-level subquery problems, same generation as v3. Tagged
    where="subquery" so they're never accidentally caught by the WHERE
    holdout filter -- they use their own dedicated WHERE...IN template,
    not variants 6/7, so they stay in-distribution for this split."""
    tagged = []
    random.seed(0)
    for select_item in SELECT_ITEMS:
        for join in JOINS:
            for groupby in GROUPBY_VARIANTS:
                for orderby in ORDERBY_VARIANTS[:2]:
                    for limit in LIMIT_VARIANTS[:2]:
                        where = ["WHERE", "COLUMN", "IN", "SUBQUERY_START", "SUBQUERY_END"]
                        blocks = _assemble(select_item, join, where, groupby, orderby, limit)
                        pos = blocks.index("SUBQUERY_START")
                        nested = random.choice(SIMPLE_SUBQUERY_POOL)
                        problem = make_problem(blocks, subqueries={pos: nested})
                        tagged.append((problem, {"where": "subquery"}))
    return tagged


def build_structural_split(val_fraction=0.05, seed=0):
    """Returns (train_problems, interp_val_problems, ood_val_problems)."""
    tagged = _generate_flat_with_tags() + _generate_subquery_with_tags()

    ood = [p for p, tag in tagged if tag["where"] in HELD_OUT_WHERE_INDICES]
    seen = [p for p, tag in tagged if tag["where"] not in HELD_OUT_WHERE_INDICES]

    rng = random.Random(seed)
    rng.shuffle(seen)
    n_val = max(1, int(val_fraction * len(seen)))
    interp_val = seen[:n_val]
    train = seen[n_val:]

    return train, interp_val, ood


def to_token_sequences(problems):
    out = []
    for p in problems:
        flat = flatten(p)
        ids = [SOS_ID] + [TOKEN_TO_ID[t] for t in flat] + [EOS_ID]
        out.append(torch.tensor(ids, dtype=torch.long))
    return out


if __name__ == "__main__":
    train, interp_val, ood_val = build_structural_split()

    print(f"train:      {len(train)} problems  (never contains AND/OR-chained WHERE)")
    print(f"interp_val: {len(interp_val)} problems  (same knob-values as train, random slice)")
    print(f"ood_val:    {len(ood_val)} problems  (AND/OR-chained WHERE -- never seen in training)\n")

    print("Example from train:")
    print(f"  {train[0]['blocks']}\n")

    print("Example from interp_val:")
    print(f"  {interp_val[0]['blocks']}\n")

    print("Example from ood_val (structure never seen in training):")
    print(f"  {ood_val[0]['blocks']}\n")

    # Confirm the holdout is actually clean: no AND/OR-chained WHERE
    # anywhere in train or interp_val.
    def has_chained_where(blocks):
        return "WHERE" in blocks and ("AND" in blocks or "OR" in blocks) and "IN" not in blocks

    leaked = sum(1 for p in train + interp_val if has_chained_where(p["blocks"]))
    print(f"AND/OR-chained WHERE examples leaked into train+interp_val: {leaked} (should be 0)")

    train_seqs = to_token_sequences(train)
    interp_seqs = to_token_sequences(interp_val)
    ood_seqs = to_token_sequences(ood_val)
    print(f"\nToken sequences ready: train={len(train_seqs)}, "
          f"interp_val={len(interp_seqs)}, ood_val={len(ood_seqs)}")
