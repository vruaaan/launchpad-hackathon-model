import itertools
import random

VERTICAL_BLOCKS = [
    "SELECT",
    "FROM",
    "WHERE",
    "GROUP_BY",
    "HAVING",
    "ORDER_BY",
    "LIMIT",
    "JOIN",
]

HORIZONTAL_BLOCKS = [
    "DISTINCT",
    "COLUMN",           # <column name>, optionally alias-qualified
    "STAR",              # (*)
    "TABLE",              # <table name>, optionally aliased
    "IS_NULL",
    "IS_NOT_NULL",
    "OPERATOR",           # dropdown: =, >=, <=, <, >, !=, <>
    "LIKE",
    "IN",
    "NOT",
    "AND",
    "OR",
    "VALUE",
    "AGG_FUNC",           # dropdown: COUNT, SUM, AVG, MIN, MAX -- wraps COLUMN/STAR
    "ASC",
    "DESC",
    "OFFSET",
    "ON",
    "SUBQUERY_START",     # opens a nested query -- pushes a new frame
    "SUBQUERY_END",       # closes it -- placed after the nested frame auto-pops
]


BLOCK_TYPES = VERTICAL_BLOCKS + HORIZONTAL_BLOCKS
BLOCK_TO_IDX = {b: i for i, b in enumerate(BLOCK_TYPES)}
IDX_TO_BLOCK = {i: b for b, i in BLOCK_TO_IDX.items()}
N_BLOCKS = len(BLOCK_TYPES)

MAX_SEQUENCE_LEN = 14   # padded length of the CURRENT (innermost) frame's built_seq
MAX_TOTAL_STEPS = 40    # episode-wide step budget across all frames combined
MAX_DEPTH = 3            # cap on subquery nesting depth (0 = top level)




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
    "select_item_done": {"FROM": "after_from"},
    "after_from": {"TABLE": "main_table_done"},

    # --- After FROM TABLE: everything else is optional from here ---
    "main_table_done": {
        "JOIN": "after_join",
        "WHERE": "where_cond_open",
        "GROUP_BY": "after_groupby",
        "ORDER_BY": "after_orderby",
        "LIMIT": "after_limit",
    },

    # --- JOIN ---
    "after_join": {"TABLE": "join_table_done"},
    "join_table_done": {"ON": "after_on"},
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
    stage = "start"
    for tok in built_seq:
        stage = TRANSITIONS[stage][tok]
    return stage

def legal_next_blocks(built_seq):
    stage = replay(built_seq)
    return set(TRANSITIONS.get(stage, {}).keys())

def valid_stop(built_seq):
    return replay(built_seq) in COMPLETE_STATES

SELECT_ITEMS = [ # things that can come after ITEMS
    ["COLUMN"],
    ["DISTINCT", "COLUMN"],
    ["STAR"],
    ["AGG_FUNC", "COLUMN"],
    ["AGG_FUNC", "STAR"],
]

JOINS = [ # things that can come after JOIN
    None,
    ["JOIN", "TABLE", "ON", "COLUMN", "OPERATOR", "COLUMN"],
]

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


def generate_problem_bank(max_depth):
    final = []
    subquery_pool = []
    for select_item, join, where, groupby, orderby, limit in itertools.product(
    SELECT_ITEMS, JOINS, WHERE_VARIANTS, GROUPBY_VARIANTS, ORDERBY_VARIANTS, LIMIT_VARIANTS
):
        blocks = assemble(select_item, join, where, groupby, orderby, limit)
        final.append(make_problem(blocks))
    for select_item, where in itertools.product(
        SELECT_ITEMS, WHERE_VARIANTS[:6]  # skip the AND/OR-chained variants, keep it simple
    ):
        blocks = assemble(select_item, None, where, None, None, None)
        subquery_pool.append(make_problem(blocks))
    random.seed(0)  # deterministic bank generation
    for select_item, join, groupby, orderby, limit in itertools.product(
        SELECT_ITEMS, JOINS, GROUPBY_VARIANTS, ORDERBY_VARIANTS[:2], LIMIT_VARIANTS[:2]
    ):
        where = ["WHERE", "COLUMN", "IN", "SUBQUERY_START", "SUBQUERY_END"]
        blocks = assemble(select_item, join, where, groupby, orderby, limit)
        subquery_pos = blocks.index("SUBQUERY_START")
        nested = random.choice(subquery_pool)
        final.append(make_problem(blocks, subqueries={subquery_pos: nested}))
    for _ in range(20):
        inner = random.choice(subquery_pool)
        mid_blocks = ["SELECT", "COLUMN", "FROM", "TABLE", "WHERE", "COLUMN", "IN", "SUBQUERY_START", "SUBQUERY_END"]
        mid = make_problem(mid_blocks, subqueries={7: inner})

        outer_blocks = ["SELECT", "COLUMN", "FROM", "TABLE", "WHERE", "COLUMN", "IN", "SUBQUERY_START", "SUBQUERY_END"]
        final.append(make_problem(outer_blocks, subqueries={7: mid}))
    return final, subquery_pool

PROBLEM_BANK, SIMPLE_SUBQUERY_POOL = generate_problem_bank(MAX_DEPTH)
