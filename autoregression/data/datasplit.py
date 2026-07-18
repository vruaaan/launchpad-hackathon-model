"""
Extends sql_dataset_ood_split.py from one held-out knob (WHERE-chaining) to
ALL six knobs at once -- with each held-out choice verified against a
token-exposure audit first, since not every variant is a fair thing to
hold out.

Why some variants can't be held out at all
-------------------------------------------
A variant is only a fair generalization test if the tokens it introduces
are ALSO seen somewhere else in training. If a variant is the SOLE source
of some token, holding it out doesn't test "did the model generalize a
rule" -- it tests "can the model predict a token it has literally never
seen," which no model can do, and which isn't an interesting result either
way. Running that audit (see the `_token_sources` check below) against the
current generator finds:

    SAFE to hold out (token also appears elsewhere):
        SELECT_ITEMS[0,2,3,4], WHERE_VARIANTS[1,4,6,7]*, GROUPBY_VARIANTS[1],
        ORDERBY_VARIANTS[1], LIMIT_VARIANTS[1]
        (*WHERE_VARIANTS[6]/[7] -- the AND/OR-chained conditions -- are only
        safe because sql_problem_bank.py's GROUPBY_VARIANTS now includes a
        HAVING-chained AND/OR variant too, giving AND/OR a second source.
        Before that fix, these were unsafe -- see dataset_generation_explained.md.)

    UNSAFE (sole source of a token -- always kept in training):
        SELECT_ITEMS[1] (DISTINCT), JOINS[1] (JOIN/TABLE/ON -- the only join
        variant at all), WHERE_VARIANTS[2,3,5] (IS_NULL/IS_NOT_NULL/NOT),
        GROUPBY_VARIANTS[2,3,4] (HAVING), ORDERBY_VARIANTS[2,3] (ASC/DESC),
        LIMIT_VARIANTS[2] (OFFSET)

So exactly one SAFE variant is held out per knob (chosen below), except
JOINS, which has no safe option at all (it's a binary present/absent knob
with no redundant source for JOIN/TABLE/ON) and is therefore never
restricted -- both join states appear freely in every split.

What "equal representation" means here
----------------------------------------
Because the bank is a full cartesian product, removing one specific index
from one knob's dimension -- while leaving every OTHER knob's dimension
fully intact -- does not bias the remaining values' relative frequency in
any way: every surviving value of every knob still appears exactly as
often, relative to every other surviving value, as it did before. That
balance is verified explicitly at the bottom of this file rather than
assumed.
"""

import itertools
import random
from collections import defaultdict
from autoregression.data.problembank2 import (
    INCORRECT_CORRUPTION_BANK,
    INCORRECT_TRUNCATION_BANK,
    SIMPLE_SUBQUERY_POOL,
)
from autoregression.data.dataprep import EOS_ID, SOS_ID, TOKEN_TO_ID, flatten

import torch
SELECT_ITEMS = [
    ["COLUMN"],
    ["DISTINCT", "COLUMN"],
    ["STAR"],
    ["AGG_FUNC", "COLUMN"],
    ["AGG_FUNC", "STAR"],
]
 
JOINS = [
    None,
    ["JOIN", "TABLE", "ON", "COLUMN", "OPERATOR", "COLUMN"],
]
 
# Where-variants WITHOUT a subquery (those are generated separately below,
# since they need a nested problem attached via `subqueries=`).
WHERE_VARIANTS = [
    None,
    ["WHERE", "COLUMN", "OPERATOR", "VALUE"],
    ["WHERE", "COLUMN", "IS_NULL"],
    ["WHERE", "COLUMN", "IS_NOT_NULL"],
    ["WHERE", "COLUMN", "LIKE", "VALUE"],
    ["WHERE", "NOT", "COLUMN", "LIKE", "VALUE"],
    ["WHERE", "COLUMN", "OPERATOR", "VALUE", "AND", "COLUMN", "OPERATOR", "VALUE"],
    ["WHERE", "COLUMN", "OPERATOR", "VALUE", "OR", "COLUMN", "OPERATOR", "VALUE"],
]
 
GROUPBY_VARIANTS = [
    None,
    ["GROUP_BY", "COLUMN"],
    ["GROUP_BY", "COLUMN", "HAVING", "AGG_FUNC", "COLUMN", "OPERATOR", "VALUE"],
    # These two exist specifically so AND/OR have a second source besides
    # WHERE_VARIANTS[6]/[7] -- without them, holding out WHERE-chained
    # conditions would zero-expose the AND/OR tokens entirely (see
    # sql_dataset_multi_ood_split.py for why that matters).
    ["GROUP_BY", "COLUMN", "HAVING", "AGG_FUNC", "COLUMN", "OPERATOR", "VALUE",
     "AND", "AGG_FUNC", "COLUMN", "OPERATOR", "VALUE"],
    ["GROUP_BY", "COLUMN", "HAVING", "AGG_FUNC", "COLUMN", "OPERATOR", "VALUE",
     "OR", "AGG_FUNC", "COLUMN", "OPERATOR", "VALUE"],
]
 
ORDERBY_VARIANTS = [
    None,
    ["ORDER_BY", "COLUMN"],
    ["ORDER_BY", "COLUMN", "ASC"],
    ["ORDER_BY", "COLUMN", "DESC"],
]
 
LIMIT_VARIANTS = [
    None,
    ["LIMIT"],
    ["LIMIT", "OFFSET"],
]
 
 
def assemble(select_item, join, where, groupby, orderby, limit):
    blocks = ["SELECT"] + select_item + ["FROM", "TABLE"]
    for part in (join, where, groupby, orderby, limit):
        if part:
            blocks += part
    return blocks




def make_problem(blocks, subqueries=None):
    return {"blocks": blocks, "subqueries": subqueries or {}}

# (imported above from autoregression.data.dataprep)

KNOBS = {
    "select": SELECT_ITEMS,
    "join": JOINS,
    "where": WHERE_VARIANTS,
    "groupby": GROUPBY_VARIANTS,
    "orderby": ORDERBY_VARIANTS,
    "limit": LIMIT_VARIANTS,
}

HELD_OUT = {
    "select": {4},     # ["AGG_FUNC", "STAR"]
    "where": {6, 7},   # AND-chained, OR-chained conditions
    "groupby": {1},    # bare "GROUP_BY COLUMN", no HAVING
    "orderby": {1},    # bare "ORDER_BY COLUMN", no ASC/DESC
    "limit": {1},      # bare "LIMIT", no OFFSET
}


def audit_token_sources():
    """Confirms every index in HELD_OUT is actually safe (its tokens all
    have a source outside the entries being held out), and prints the
    full safe/unsafe table for visibility."""
    token_sources = defaultdict(set)
    for kname, variants in KNOBS.items(): #iterate through all variants and knobs
        for i, v in enumerate(variants): 
            if not v:
                continue
            for tok in set(v):
                token_sources[tok].add((kname, i))
    all_safe = True
    for kname, held in HELD_OUT.items():
        for idx in held:
            v = KNOBS[kname][idx]
            for tok in set(v):
                other_sources = token_sources[tok] - {(kname, idx)}
                if not other_sources:
                    all_safe = False
                    raise AssertionError(
                        f"HELD_OUT['{kname}']={idx} ({v}) is the SOLE source of "
                        f"token {tok!r} -- holding it out would zero-expose it, "
                        f"not test generalization.")
    if all_safe:
        print("Audit OK: all held-out variants are safe (no token is solely sourced by a held-out variant).")

audit_token_sources()


def generate_flat_with_tags():
    tagged = []
    for si, select_item in enumerate(SELECT_ITEMS):
        for ji, join in enumerate(JOINS):
            for wi, where in enumerate(WHERE_VARIANTS):
                for gi, groupby in enumerate(GROUPBY_VARIANTS):
                    for oi, orderby in enumerate(ORDERBY_VARIANTS):
                        for li, limit in enumerate(LIMIT_VARIANTS):
                            blocks = assemble(select_item, join, where, groupby, orderby, limit)
                            problem = make_problem(blocks)
                            tags = {"select": si, "join": ji, "where": wi,
                                    "groupby": gi, "orderby": oi, "limit": li}
                            tagged.append((problem, tags))
    return tagged


def generate_subquery_with_tags():
    """Subquery problems are tagged the same way for the knobs that still
    apply to the OUTER query, with where='subquery' since they use their
    own dedicated WHERE...IN template rather than a WHERE_VARIANTS index.
    They're never selected by any held-out filter, so they stay in train."""
    tagged = []
    rng = random.Random(0)
    for si, select_item in enumerate(SELECT_ITEMS):
        for ji, join in enumerate(JOINS):
            for gi, groupby in enumerate(GROUPBY_VARIANTS):
                for oi, orderby in enumerate(ORDERBY_VARIANTS):
                    for li, limit in enumerate(LIMIT_VARIANTS):
                        where = ["WHERE", "COLUMN", "IN", "SUBQUERY_START", "SUBQUERY_END"]
                        blocks = assemble(select_item, join, where, groupby, orderby, limit)
                        pos = blocks.index("SUBQUERY_START")
                        nested = rng.choice(SIMPLE_SUBQUERY_POOL)
                        problem = make_problem(blocks, subqueries={pos: nested})
                        tags = {"select": si, "join": ji, "where": "subquery",
                                "groupby": gi, "orderby": oi, "limit": li}
                        tagged.append((problem, tags))
    return tagged


def is_held_out(tags, knob):
    val = tags[knob]
    return isinstance(val, int) and val in HELD_OUT.get(knob, set())


def any_held_out(tags):
    return any(is_held_out(tags, k) for k in HELD_OUT)


def only_this_axis_held_out(tags, knob):
    if not is_held_out(tags, knob):
        return False
    return all(is_held_out(tags, k) is False for k in HELD_OUT if k != knob)


def build_multi_knob_split(val_fraction=0.05, seed=0):
    """
    Returns:
      train              -- no knob at any held-out value
      interp_val         -- random slice of train (standard in-distribution val)
      ood_by_knob        -- dict: knob_name -> problems where ONLY that knob
                             is novel (single-factor generalization test)
      ood_compound       -- problems where 2+ knobs are simultaneously novel
                             (a harder, multi-factor generalization test)
    """
    tagged = generate_flat_with_tags() + generate_subquery_with_tags()
    train_tagged = [(p, t) for p, t in tagged if not any_held_out(t)]

    ood_by_knob = {}
    for knob in HELD_OUT:
        ood_by_knob[knob] = [p for p, t in tagged if only_this_axis_held_out(t, knob)]

    n_held_axes = lambda t: sum(is_held_out(t, k) for k in HELD_OUT)
    ood_compound = [p for p, t in tagged if n_held_axes(t) >= 2]

    rng = random.Random(seed)
    train_problems = [p for p, t in train_tagged]
    rng.shuffle(train_problems)
    n_val = max(1, int(val_fraction * len(train_problems)))
    interp_val = train_problems[:n_val]
    train = train_problems[n_val:]

    return train, interp_val, ood_by_knob, ood_compound, train_tagged

def normalise_data(problem):
    flat = flatten(problem)
    ids = [SOS_ID] + [TOKEN_TO_ID[t] for t in flat] + [EOS_ID]
    return torch.tensor(ids, dtype=torch.long)


def to_token_seq(bank):
    sequences = []
    for problem in bank:
        sequences.append(normalise_data(problem))
    return sequences


def verify_balance(train_tagged):
    """Confirms every IN-DISTRIBUTION value of every knob appears an equal
    number of times in train, relative to other in-distribution values of
    the same knob -- 'equal representation' isn't assumed, it's checked."""
    counts = {k: defaultdict(int) for k in KNOBS}
    for p, t in train_tagged:
        for k, v in t.items():
            if isinstance(v, int):
                counts[k][v] += 1
    for k, c in counts.items():
        vals = list(c.values())
        balanced = len(set(vals)) == 1
        print(f"  {k}: counts per in-distribution value = {dict(c)}  "
              f"{'(balanced)' if balanced else '(NOT balanced!)'}")

training_set, validation_set, ood_by_knob, ood_compound, train_tagged_all = build_multi_knob_split()
valid_training_set = list(training_set)
print(f"train:{len(training_set)} problems")
print(f"interp_val: {len(validation_set)} problems")
for knob, probs in ood_by_knob.items():
    print(f"ood[{knob}]:  {len(probs)} problems (only '{knob}' is novel)")
print(f"ood_compound: {len(ood_compound)} problems (2+ knobs novel at once)\n")
verify_balance(train_tagged_all)


INCLUDE_INCORRECT = False
INCLUDE_TRUNCATIONS = True
TRUNCATION_MAX = None  # set to an int to cap how many legal prefixes are added
INCORRECT_SEED = 0

if INCLUDE_INCORRECT:
    raise ValueError(
        "Hard incorrect corruptions are intentionally excluded from normal LM training. "
        "Use INCLUDE_TRUNCATIONS for legal incomplete prefixes."
    )

if INCLUDE_TRUNCATIONS:
    rng = random.Random(INCORRECT_SEED)
    truncations = list(INCORRECT_TRUNCATION_BANK)
    rng.shuffle(truncations)
    if TRUNCATION_MAX is not None:
        truncations = truncations[:TRUNCATION_MAX]
    training_set = list(training_set) + truncations
    print(f"Added {len(truncations)} legal incomplete prefixes to training_set -> total {len(training_set)}")


def make_cursor_problem(problem, cursor_pos):
    flat = flatten(problem)
    blocks = flat[:cursor_pos] + ["<CURSOR>"] + flat[cursor_pos:]
    return make_problem(blocks)


def build_cursor_training_examples(problems, k=3, seed=0):
    """
    Builds insertion-style LM examples from valid problems. A sequence like
    SELECT <CURSOR> COLUMN FROM TABLE teaches COLUMN as the target after
    <CURSOR>, while preserving the normal left-to-right loss everywhere else.
    """
    rng = random.Random(seed)
    examples = []
    for problem in problems:
        flat = flatten(problem)
        if not flat:
            continue
        positions = {0, len(flat)}
        interior = list(range(1, len(flat)))
        rng.shuffle(interior)
        positions.update(interior[: max(0, k - len(positions))])
        for cursor_pos in sorted(positions):
            examples.append(make_cursor_problem(problem, cursor_pos))
    return examples


cursor_training_set = build_cursor_training_examples(valid_training_set, k=3, seed=0)
print(f"Added {len(cursor_training_set)} cursor insertion examples")


training_tensors = to_token_seq(training_set) + to_token_seq(cursor_training_set)
validation_tensors = to_token_seq(validation_set)

# ---------------------------------------------------------------------------
# Extra evaluation sets for autocomplete robustness.
# ---------------------------------------------------------------------------

def build_prefix_next_token_pairs(problems, n_pairs=2000, seed=0, min_prefix_len=1):
    """
    Returns (prefix_inputs, next_token_ids) where each prefix input is a 1D
    LongTensor beginning with SOS and ending at some cutoff inside the
    flattened sequence; next_token_ids is a list of the true next token id.

    This matches the autocomplete setting: given an incomplete prefix, score
    whether the model ranks the correct next token highly.
    """
    rng = random.Random(seed)
    prefix_inputs = []
    next_token_ids = []

    for _ in range(n_pairs):
        problem = rng.choice(problems)
        flat = flatten(problem)
        if len(flat) <= 1:
            continue
        cut = rng.randrange(max(1, min_prefix_len), len(flat))
        prefix = flat[:cut]
        next_tok = flat[cut]
        prefix_ids = torch.tensor([SOS_ID] + [TOKEN_TO_ID[t] for t in prefix], dtype=torch.long)
        prefix_inputs.append(prefix_ids)
        next_token_ids.append(TOKEN_TO_ID[next_tok])

    return prefix_inputs, next_token_ids


# Prefix eval: use truncations (incomplete-but-sql-ish) for realism.
incorrect_prefix_inputs, incorrect_prefix_next_ids = build_prefix_next_token_pairs(
    INCORRECT_TRUNCATION_BANK,
    n_pairs=2000,
    seed=0,
    min_prefix_len=1,
)


def build_cursor_next_token_pairs(problems, n_pairs=2000, seed=1):
    """
    Returns (cursor_inputs, next_token_ids) where each input ends with
    <CURSOR>, and next_token_ids is the token that was originally at that
    cursor position.
    """
    rng = random.Random(seed)
    cursor_inputs = []
    next_token_ids = []

    for _ in range(n_pairs):
        problem = rng.choice(problems)
        flat = flatten(problem)
        if not flat:
            continue
        cursor_pos = rng.randrange(0, len(flat))
        prefix = flat[:cursor_pos] + ["<CURSOR>"]
        next_tok = flat[cursor_pos]
        prefix_ids = torch.tensor([SOS_ID] + [TOKEN_TO_ID[t] for t in prefix], dtype=torch.long)
        cursor_inputs.append(prefix_ids)
        next_token_ids.append(TOKEN_TO_ID[next_tok])

    return cursor_inputs, next_token_ids


cursor_prefix_inputs, cursor_prefix_next_ids = build_cursor_next_token_pairs(
    validation_set,
    n_pairs=2000,
    seed=1,
)

# Corruption eval: perplexity over the corrupted sequences themselves (distribution-shift stress test).
incorrect_corruption_tensors = to_token_seq(INCORRECT_CORRUPTION_BANK)

