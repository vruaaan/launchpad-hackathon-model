"""
The SQL grammar (a hand-rolled finite-state machine over the block
vocabulary) and the combinatorial problem bank built from it, generated
and validated together in this file rather than inside the RL env.
 
Why the grammar lives here too, not in sql_block_env_v3.py
-------------------------------------------------------------
The problem bank is generated and then immediately validated against the
grammar (see `validate_problem` at the bottom) -- every one of the 5,020
generated problems has every prefix checked for legality and every ending
checked for completeness at import time. That validation IS the grammar's
main consumer, so it belongs next to the bank it's validating. The RL env
(sql_block_env_v3.py) still needs the grammar too (for its wrong-but-legal
vs wrong-and-illegal reward split), so it imports `legal_next_blocks` and
friends from here rather than the bank importing them from the env --
that direction would be circular, since the env also needs the bank.
 
    SELECT [DISTINCT] (AGG_FUNC)? (COLUMN|STAR)
    FROM TABLE
    [JOIN TABLE ON COLUMN OPERATOR COLUMN]
    [WHERE <cond> (AND|OR <cond>)*]
    [GROUP_BY COLUMN [HAVING <cond> (AND|OR <cond>)*]]
    [ORDER_BY COLUMN [ASC|DESC]]
    [LIMIT [OFFSET]]
 
where <cond> is one of:
    COLUMN OPERATOR VALUE
    COLUMN IS_NULL | COLUMN IS_NOT_NULL
    [NOT] COLUMN LIKE VALUE
    COLUMN IN SUBQUERY_START SUBQUERY_END   (the nested query itself lives
                                              in a separate pushed Frame --
                                              from THIS frame's grammar
                                              perspective it's just three
                                              tokens in a row, exactly like
                                              any other atomic condition)
 
Subqueries reuse this exact same grammar recursively (each pushed Frame
starts its own FSM from "start"), so a nested query has to obey the same
rules as a top-level one -- there's no separate "subquery grammar."
"""
 
import itertools
import random

def make_problem(blocks, subqueries=None):
    return {"blocks": blocks, "subqueries": subqueries or {}}


def _deepcopy_problem(problem):
    return make_problem(list(problem["blocks"]), subqueries={k: v for k, v in problem.get("subqueries", {}).items()})
 
# ---------------------------------------------------------------------------
# 1. General SQL grammar -- a finite-state machine over block tokens.
# ---------------------------------------------------------------------------
# TRANSITIONS[state][token] = next_state. If `token` isn't a key at the
# current state, it's not legal SQL at this position under ANY query.
# COMPLETE_STATES = states where the frame could legally stop right now.
 
TRANSITIONS = {
    "start": {"SELECT": "select_open"},
 
    # --- SELECT item ---
    "select_open": {
        "DISTINCT": "select_after_distinct",
        "AGG_FUNC": "select_after_agg",
        "COLUMN": "select_item_done",
        "STAR": "select_item_done",
    },
    "select_after_distinct": {
        "AGG_FUNC": "select_after_agg",
        "COLUMN": "select_item_done",
        "STAR": "select_item_done",
    },
    "select_after_agg": {"COLUMN": "select_item_done", "STAR": "select_item_done"},

    # Optional alias for the SELECT item (e.g., COLUMN AS ALIAS)
    "select_item_done": {"AS": "select_after_as", "FROM": "after_from"},
    "select_after_as": {"ALIAS": "select_item_aliased"},
    "select_item_aliased": {"FROM": "after_from"},

    "after_from": {"TABLE": "main_table_done"},
 
    # --- After FROM TABLE: everything else is optional from here ---
    "main_table_done": {
        "AS": "from_after_as",
        "JOIN": "after_join",
        "WHERE": "where_cond_open",
        "GROUP_BY": "after_groupby",
        "ORDER_BY": "after_orderby",
        "LIMIT": "after_limit",
    },
    "from_after_as": {"ALIAS": "main_table_done"},
 
    # --- JOIN ---
    "after_join": {"TABLE": "join_table_done"},
    "join_table_done": {"AS": "join_after_as", "ON": "after_on"},
    "join_after_as": {"ALIAS": "join_table_done_aliased"},
    "join_table_done_aliased": {"ON": "after_on"},
    "after_on": {"COLUMN": "on_lhs_done"},
    "on_lhs_done": {"OPERATOR": "on_op_done"},
    "on_op_done": {"COLUMN": "join_done"},
    "join_done": {
        "WHERE": "where_cond_open",
        "GROUP_BY": "after_groupby",
        "ORDER_BY": "after_orderby",
        "LIMIT": "after_limit",
    },
 
    # --- WHERE condition automaton ---
    "where_cond_open": {"NOT": "where_after_not", "COLUMN": "where_cond_lhs_done"},
    "where_after_not": {"COLUMN": "where_cond_lhs_done"},
    "where_cond_lhs_done": {
        "OPERATOR": "where_cond_op_done",
        "IS_NULL": "where_cond_done",
        "IS_NOT_NULL": "where_cond_done",
        "LIKE": "where_cond_op_done",
        "IN": "where_cond_in",
    },
    "where_cond_op_done": {"VALUE": "where_cond_done"},
    "where_cond_in": {"SUBQUERY_START": "where_cond_subquery_open"},
    "where_cond_subquery_open": {"SUBQUERY_END": "where_cond_done"},
    "where_cond_done": {
        "AND": "where_cond_open",
        "OR": "where_cond_open",
        "GROUP_BY": "after_groupby",
        "ORDER_BY": "after_orderby",
        "LIMIT": "after_limit",
    },
 
    # --- GROUP BY / HAVING ---
    "after_groupby": {"COLUMN": "groupby_col_done"},
    "groupby_col_done": {
        "HAVING": "having_cond_open",
        "ORDER_BY": "after_orderby",
        "LIMIT": "after_limit",
    },
    "having_cond_open": {"AGG_FUNC": "having_after_agg", "COLUMN": "having_cond_lhs_done"},
    "having_after_agg": {"COLUMN": "having_cond_lhs_done"},
    "having_cond_lhs_done": {"OPERATOR": "having_cond_op_done"},
    "having_cond_op_done": {"VALUE": "having_cond_done"},
    "having_cond_done": {
        "AND": "having_cond_open",
        "OR": "having_cond_open",
        "ORDER_BY": "after_orderby",
        "LIMIT": "after_limit",
    },
 
    # --- ORDER BY ---
    "after_orderby": {"COLUMN": "orderby_col_done"},
    "orderby_col_done": {"ASC": "orderby_dir_done", "DESC": "orderby_dir_done", "LIMIT": "after_limit"},
    "orderby_dir_done": {"LIMIT": "after_limit"},
 
    # --- LIMIT / OFFSET ---
    "after_limit": {"OFFSET": "after_offset"},
    "after_offset": {},
}
 
COMPLETE_STATES = {
    "main_table_done", "join_done", "where_cond_done", "groupby_col_done",
    "having_cond_done", "orderby_col_done", "orderby_dir_done",
    "after_limit", "after_offset",
}

def replay(built_seq):
    state = "start"
    for tok in built_seq:
        state = TRANSITIONS[state][tok]
    return state
 
def legal_next_blocks(built_seq):
    state = replay(built_seq)
    return set(TRANSITIONS.get(state, {}).keys())
 
 
def can_stop_here(built_seq):
    return replay(built_seq) in COMPLETE_STATES
 
 
# ---------------------------------------------------------------------------
# 2. Scaled-up problem bank, generated combinatorially from the grammar's
#    own building blocks (so every generated problem is legal by
#    construction -- validated at the bottom of this file as a sanity check).
# ---------------------------------------------------------------------------
 
SELECT_ITEMS = [
    ["COLUMN"],
    ["DISTINCT", "COLUMN"],
    ["STAR"],
    ["AGG_FUNC", "COLUMN"],
    ["AGG_FUNC", "STAR"],
]

SELECT_ALIASES = [
    None,
    ["AS", "ALIAS"],
]
 
JOINS = [
    None,
    ["JOIN", "TABLE", "ON", "COLUMN", "OPERATOR", "COLUMN"],
    ["JOIN", "TABLE", "AS", "ALIAS", "ON", "COLUMN", "OPERATOR", "COLUMN"],
]

FROM_ALIASES = [
    None,
    ["AS", "ALIAS"],
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
 
 
def assemble(select_item, select_alias, from_alias, join, where, groupby, orderby, limit):
    blocks = ["SELECT"] + select_item
    if select_alias:
        blocks += select_alias
    blocks += ["FROM", "TABLE"]
    if from_alias:
        blocks += from_alias
    for part in (join, where, groupby, orderby, limit):
        if part:
            blocks += part
    return blocks
 

# --- Validate every generated problem against the grammar (build-time sanity check -- catches FSM bugs and generator bugs together) ---
def validate_problem(problem, path="root"):
    blocks = problem["blocks"]
    for i in range(1, len(blocks) + 1):
        prefix = blocks[:i]
        try:
            state = replay(prefix)
        except KeyError:
            raise AssertionError(f"[{path}] illegal prefix under grammar: {prefix}")
    final_state = replay(blocks)
    if final_state not in COMPLETE_STATES:
        raise AssertionError(f"[{path}] target ends in a non-terminal state: {blocks}")
    for idx, sub in problem.get("subqueries", {}).items():
        validate_problem(sub, path=f"{path}->sub@{idx}")


def validate_prefixes_only(problem, path="root"):
    """
    Sanity-check a *partial* or intentionally incomplete sequence.

    Ensures every prefix is legal under the grammar, but does NOT require the
    final state to be terminal/complete. This is useful for autocomplete
    training where the model must handle in-progress queries.
    """
    blocks = problem["blocks"]
    for i in range(1, len(blocks) + 1):
        prefix = blocks[:i]
        try:
            replay(prefix)
        except KeyError:
            raise AssertionError(f"[{path}] illegal prefix under grammar: {prefix}")
    for idx, sub in problem.get("subqueries", {}).items():
        validate_prefixes_only(sub, path=f"{path}->sub@{idx}")


def _truncate_problem(problem, rng, min_len=1):
    blocks = problem["blocks"]
    if len(blocks) <= min_len:
        return None
    cut = rng.randrange(min_len, len(blocks))
    truncated = make_problem(blocks[:cut], subqueries={})
    # If we cut before/inside a subquery marker, drop subqueries entirely for the partial.
    return truncated


def _corrupt_blocks(blocks, rng):
    """
    Produce an *incorrect* sequence that is still SQL-ish.

    The output may be:
      - incomplete-but-legal (e.g. truncation handled elsewhere),
      - illegal (e.g. SELECT FROM),
      - or structurally odd (e.g. duplicate WHERE).
    """
    if not blocks:
        return blocks

    patterns = []

    # 1) Delete a "slot" token if present (common missing-block failure mode).
    slot_tokens = {"COLUMN", "TABLE", "VALUE", "OPERATOR", "STAR"}
    deletable = [i for i, t in enumerate(blocks) if t in slot_tokens]
    if deletable:
        i = rng.choice(deletable)
        patterns.append(blocks[:i] + blocks[i + 1 :])

    # 2) Swap a nearby pair (horizontal mis-ordering).
    if len(blocks) >= 2:
        i = rng.randrange(0, len(blocks) - 1)
        swapped = list(blocks)
        swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
        patterns.append(swapped)

    # 3) Duplicate a clause keyword (stray extra block).
    keywords = [i for i, t in enumerate(blocks) if t in {"WHERE", "JOIN", "GROUP_BY", "ORDER_BY", "LIMIT"}]
    if keywords:
        i = rng.choice(keywords)
        patterns.append(blocks[: i + 1] + [blocks[i]] + blocks[i + 1 :])

    # 4) Force the exact bad forms the user mentioned, when possible.
    if "SELECT" in blocks and "FROM" in blocks:
        patterns.append(["SELECT", "FROM"])
        patterns.append(["SELECT", "FROM", "WHERE"])

    return rng.choice(patterns) if patterns else blocks


def generate_incorrect_problem_bank(
    valid_bank,
    *,
    n_truncations=None,
    n_corruptions=None,
    seed=0,
):
    """
    Build a bank of intentionally incorrect / incomplete queries for autocomplete.

    - Truncations are always legal prefixes of valid queries (incomplete SQL).
    - Corruptions introduce missing/extra/mis-ordered tokens (may be illegal).

    Returns a list of problems in the same shape as PROBLEM_BANK entries.
    """
    rng = random.Random(seed)
    if n_truncations is None:
        n_truncations = len(valid_bank) // 2
    if n_corruptions is None:
        n_corruptions = len(valid_bank) - n_truncations

    truncations = []
    corruptions = []

    # Truncated (incomplete but grammar-legal prefixes)
    for _ in range(n_truncations):
        base = rng.choice(valid_bank)
        truncated = _truncate_problem(base, rng, min_len=1)
        if truncated is not None:
            # Prefix legality should hold by construction; still cheap to assert.
            validate_prefixes_only(truncated, path="incorrect_trunc")
            truncations.append(truncated)

    # Corrupted (missing blocks / mis-order / duplicates; may be illegal)
    for _ in range(n_corruptions):
        base = rng.choice(valid_bank)
        corrupted_blocks = _corrupt_blocks(base["blocks"], rng)
        corruptions.append(make_problem(corrupted_blocks, subqueries={}))

    return truncations, corruptions


def generate_problem_bank():
    final = []
    subquery_pool = []
    for select_item, select_alias, from_alias, join, where, groupby, orderby, limit in itertools.product( #Main combinatorial sweep, no subqueries yet
    SELECT_ITEMS, SELECT_ALIASES, FROM_ALIASES, JOINS, WHERE_VARIANTS, GROUPBY_VARIANTS, ORDERBY_VARIANTS, LIMIT_VARIANTS):
        blocks = assemble(select_item, select_alias, from_alias, join, where, groupby, orderby, limit)
        final.append(make_problem(blocks))
    for select_item, select_alias, from_alias, where in itertools.product( # skip the AND/OR-chained variants, keep it simple for subquery
    SELECT_ITEMS, SELECT_ALIASES, FROM_ALIASES, WHERE_VARIANTS[:6]):
        blocks = assemble(select_item, select_alias, from_alias, None, where, None, None, None)
        subquery_pool.append(make_problem(blocks))
    random.seed(0)  # deterministic bank generation
    for select_item, select_alias, from_alias, join, groupby, orderby, limit in itertools.product( #level 1 nesting of subqueries
    SELECT_ITEMS, SELECT_ALIASES, FROM_ALIASES, JOINS, GROUPBY_VARIANTS, ORDERBY_VARIANTS[:2], LIMIT_VARIANTS[:2]):
        where = ["WHERE", "COLUMN", "IN", "SUBQUERY_START", "SUBQUERY_END"]
        blocks = assemble(select_item, select_alias, from_alias, join, where, groupby, orderby, limit)
        subquery_pos = blocks.index("SUBQUERY_START")
        nested = random.choice(subquery_pool)
        final.append(make_problem(blocks, subqueries={subquery_pos: nested}))
    for _ in range(20): #level 2 nesting of subqueries
        inner = random.choice(subquery_pool)
        mid_blocks = ["SELECT", "COLUMN", "FROM", "TABLE", "WHERE", "COLUMN", "IN", "SUBQUERY_START", "SUBQUERY_END"]
        mid = make_problem(mid_blocks, subqueries={7: inner})
        outer_blocks = ["SELECT", "COLUMN", "FROM", "TABLE", "WHERE", "COLUMN", "IN", "SUBQUERY_START", "SUBQUERY_END"]
        final.append(make_problem(outer_blocks, subqueries={7: mid}))
    for _i, _p in enumerate(final): #validation check
        validate_problem(_p, path=f"problem[{_i}]")
    return final, subquery_pool


PROBLEM_BANK, SIMPLE_SUBQUERY_POOL = generate_problem_bank()

# Intentionally incorrect / incomplete SQL sequences for autocomplete training.
# Kept separate so the original bank can remain "all complete + grammar-valid".
INCORRECT_TRUNCATION_BANK, INCORRECT_CORRUPTION_BANK = generate_incorrect_problem_bank(PROBLEM_BANK, seed=0)
INCORRECT_PROBLEM_BANK = INCORRECT_TRUNCATION_BANK + INCORRECT_CORRUPTION_BANK
