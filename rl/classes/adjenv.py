import numpy as np
import gymnasium as gym
from gymnasium import spaces
import itertools
import numpy as np
import random
from dataclasses import dataclass, field


#BLOCKS
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


@dataclass
class Frame:
    target: dict # {"blocks": [...], "subqueries": {...}}
    built_seq: list = field(default_factory=list)


@dataclass
class EpisodeState:
    stack: list = field(default_factory=list)  # stack[-1] is the active (innermost) frame
    total_steps: int = 0  # steps across ALL frames, for truncation

#REWARD SHAPING
REWARD = {
    "block_reward": { #bonus based on the block placed, checked according to pattern
        "correct": {
            "SUBQUERY_START": {"CORRECT_NESTING": 1.0, "WRONG_NESTING": -1.0},
            "others": 1.0,
        },
        "wrong": {"legal_wrong": -0.3, "illegal_wrong": -2.0},
    },
    "stop_reward": { #bonus based on the stop flag given 
        "correct": 1.0,
        "wrong": -1.0,
    },
    "completion_bonus": { # bonuses on top of block_reward when a frame/episode actually ends
        "subquery_pop": {"true_done": 2.0, "false_done": -2.0},
        "final": {"true_done": 5.0, "false_done": -3.0},
    }
}



def legal_mask_from_obs_batch(obs_batch):
    """Reconstruct a (B, N_BLOCKS) boolean legal-action mask directly from a
    batch of encoded observations, without needing a live env instance.

    get_obs() encodes the current frame's built_seq as the first
    MAX_SEQUENCE_LEN entries of the observation, using (BLOCK_TO_IDX[b] + 1)
    per token and 0 for padding. That's enough to replay the grammar and
    recover legal_next_blocks() for each row, which is what we need to mask
    the target network's next-state Q-values during (Double) DQN updates.
    """
    obs_np = obs_batch.detach().cpu().numpy() if hasattr(obs_batch, "detach") else np.asarray(obs_batch)
    masks = np.zeros((obs_np.shape[0], N_BLOCKS), dtype=bool)
    for i in range(obs_np.shape[0]):
        tokens = obs_np[i, :MAX_SEQUENCE_LEN].astype(int)
        built_seq = [IDX_TO_BLOCK[t - 1] for t in tokens if t > 0]
        legal = legal_next_blocks(built_seq)
        for b in legal:
            masks[i, BLOCK_TO_IDX[b]] = True
    return masks


class SQLEnvAdvanced(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        obs_dim = MAX_SEQUENCE_LEN + 1 + N_BLOCKS + 1
        self.observation_space = spaces.Box(
            low=0, high=max(N_BLOCKS, MAX_SEQUENCE_LEN),
            shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(N_BLOCKS * 2)
        self.state: EpisodeState | None = None

    @property
    def current_frame(self) -> Frame:
        return self.state.stack[-1]

    def legal_action_mask(self):
        """Boolean mask of shape (N_BLOCKS,), True for block choices that are
        syntactically legal in the current frame's grammar state. Used to mask
        the block head's Q-values before argmax/exploration so the agent
        never wastes steps on illegal moves."""
        legal = legal_next_blocks(self.current_frame.built_seq)
        mask = np.zeros(N_BLOCKS, dtype=bool)
        for b in legal:
            mask[BLOCK_TO_IDX[b]] = True
        return mask

    def required_blocks_vector(self):
        frame = self.current_frame
        target_list = frame.target["blocks"]
        remaining = target_list[len(frame.built_seq):]
        vec = np.zeros(N_BLOCKS, dtype=np.float32)
        for b in remaining:
            vec[BLOCK_TO_IDX[b]] = 1.0
        return vec

    def get_obs(self):
        frame = self.current_frame
        built_idx = [BLOCK_TO_IDX[b] + 1 for b in frame.built_seq]
        built_idx = built_idx[:MAX_SEQUENCE_LEN]
        built_idx += [0] * (MAX_SEQUENCE_LEN - len(built_idx))
        step_count = np.array([len(frame.built_seq) / MAX_SEQUENCE_LEN], dtype=np.float32)
        required = self.required_blocks_vector()
        depth = np.array([(len(self.state.stack) - 1) / MAX_DEPTH], dtype=np.float32)
        return np.concatenate([
            np.array(built_idx, dtype=np.float32),
            step_count,
            required,
            depth,
        ])

    def get_info(self):
        return {
            "depth": len(self.state.stack) - 1,
            "current_built_seq": list(self.current_frame.built_seq),
            "current_target_seq": list(self.current_frame.target["blocks"]),
            "total_steps": self.state.total_steps,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        problem = random.choice(PROBLEM_BANK)
        self.state = EpisodeState(stack=[Frame(target=problem)], total_steps=0)
        return self.get_obs(), self.get_info()

    def step(self, action):
        block_idx, stop_flag = divmod(int(action), 2)
        block = self.idx_to_block(block_idx)
        self.state.total_steps += 1
        frame = self.current_frame
        target_list = frame.target["blocks"]
        pos = len(frame.built_seq)
        expected_block = target_list[pos] if pos < len(target_list) else None
        terminated = False
        truncated = False
        block_reward = 0.0
        stop_reward = 0.0
        block_correct = expected_block is not None and block == expected_block
        syntax_ok = block in legal_next_blocks(frame.built_seq)
        if block_correct:
            assert syntax_ok, f"target block {block!r} was flagged illegal -- grammar/bank bug" #debugging
            frame.built_seq.append(block)
            true_done = len(frame.built_seq) == len(target_list)
            if block == "SUBQUERY_START": #if supposed to start subquery,
                stop_reward = (
                    REWARD["stop_reward"]["correct"] if stop_flag == 0
                    else REWARD["stop_reward"]["wrong"]
                )
                if len(self.state.stack) - 1 < MAX_DEPTH:
                    nested_problem = frame.target["subqueries"][pos]
                    self.state.stack.append(Frame(target=nested_problem))
                    block_reward = REWARD["block_reward"]["correct"]["SUBQUERY_START"]["CORRECT_NESTING"]
                else:
                    frame.built_seq.pop()
                    block_reward = REWARD["block_reward"]["correct"]["SUBQUERY_START"]["WRONG_NESTING"]
                    stop_reward = 0.0
            else:
                block_reward = REWARD["block_reward"]["correct"]["others"]
                stop_reward = (
                    REWARD["stop_reward"]["correct"] if stop_flag == int(true_done)
                    else REWARD["stop_reward"]["wrong"]
                )

                if stop_flag == 1:
                    if len(self.state.stack) > 1:
                        self.state.stack.pop()
                        block_reward += (
                            REWARD["completion_bonus"]["subquery_pop"]["true_done"] if true_done
                            else REWARD["completion_bonus"]["subquery_pop"]["false_done"]
                        )
                    else:
                        terminated = True
                        block_reward += (
                            REWARD["completion_bonus"]["final"]["true_done"] if true_done
                            else REWARD["completion_bonus"]["final"]["false_done"]
                        )
        else:
            block_reward = (
                REWARD["block_reward"]["wrong"]["legal_wrong"] if syntax_ok
                else REWARD["block_reward"]["wrong"]["illegal_wrong"]
            )
            stop_reward = REWARD["stop_reward"]["wrong"] if stop_flag == 1 else 0.0
        total_reward = block_reward + stop_reward
        if self.state.total_steps >= MAX_TOTAL_STEPS:
            truncated = True
        return self.get_obs(), total_reward, terminated, truncated, self.get_info()

    def render(self):
        if self.render_mode == "human":
            for depth, frame in enumerate(self.state.stack):
                indent = "  " * depth
                print(f"{indent}depth {depth} built:  {frame.built_seq}")
                print(f"{indent}depth {depth} target: {frame.target['blocks']}")

    @staticmethod
    def idx_to_block(idx):
        return IDX_TO_BLOCK[idx]