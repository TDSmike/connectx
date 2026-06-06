"""Negamax ConnectX opponents.

``NegamaxPolicy`` is a fast alpha-beta opponent used as a cheap curriculum
opponent. ``KaggleNegamaxPolicy`` is a local, faithful port of Kaggle's built-in
``negamax`` agent, used when the target is beating Kaggle's baseline.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .env import Policy, ROWS, COLS, INAROW

WIN = 100_000
SIZE = ROWS * COLS


def _drop_row(board, col):
    for r in range(ROWS - 1, -1, -1):
        if board[r, col] == 0:
            return r
    return -1


def _is_win(board, r, c, p):
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        cnt = 1
        for s in (1, -1):
            rr, cc = r + dr * s, c + dc * s
            while 0 <= rr < ROWS and 0 <= cc < COLS and board[rr, cc] == p:
                cnt += 1
                rr += dr * s
                cc += dc * s
        if cnt >= INAROW:
            return True
    return False


def _heuristic(board, player):
    """Cheap leaf evaluation: weighted count of open 2/3-in-a-row windows."""
    opp = 3 - player
    score = 0
    # all length-4 windows
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                er, ec = r + 3 * dr, c + 3 * dc
                if not (0 <= er < ROWS and 0 <= ec < COLS):
                    continue
                w = [board[r + i * dr, c + i * dc] for i in range(INAROW)]
                me, ot, em = w.count(player), w.count(opp), w.count(0)
                if me and ot:
                    continue
                if me == 3 and em == 1:
                    score += 50
                elif me == 2 and em == 2:
                    score += 10
                elif ot == 3 and em == 1:
                    score -= 80
                elif ot == 2 and em == 2:
                    score -= 10
    # center preference
    score += 3 * int((board[:, COLS // 2] == player).sum())
    return score


# move ordering: center-out
_ORDER = sorted(range(COLS), key=lambda c: abs(c - COLS // 2))


def negamax(board, player, depth, alpha, beta):
    legal = [c for c in _ORDER if board[0, c] == 0]
    if not legal:
        return 0, -1
    best_col = legal[0]
    best = -1e18
    for col in legal:
        r = _drop_row(board, col)
        board[r, col] = player
        if _is_win(board, r, col, player):
            board[r, col] = 0
            return WIN - (10 - depth), col          # prefer faster wins
        if depth <= 1:
            val = _heuristic(board, player)
        else:
            val = -negamax(board, 3 - player, depth - 1, -beta, -alpha)[0]
        board[r, col] = 0
        if val > best:
            best, best_col = val, col
        alpha = max(alpha, val)
        if alpha >= beta:
            break
    return best, best_col


def _root_values(board, player, depth):
    """Exact negamax value of every legal root move (no cross-root pruning).

    Returns {col: value} using the *same* per-move scoring as ``negamax`` so
    randomized tie-breaking picks uniformly among genuinely equal-best moves.
    """
    vals = {}
    for col in _ORDER:
        if board[0, col] != 0:
            continue
        r = _drop_row(board, col)
        board[r, col] = player
        if _is_win(board, r, col, player):
            v = WIN - (10 - depth)
        elif depth <= 1:
            v = _heuristic(board, player)
        else:
            v = -negamax(board, 3 - player, depth - 1, -1e18, 1e18)[0]
        board[r, col] = 0
        vals[col] = v
    return vals


class NegamaxPolicy(Policy):
    """Depth-limited negamax opponent.

    ``randomize`` breaks ties uniformly among equal-best moves (as Kaggle's own
    negamax does).  This matters for *evaluation*: against a deterministic
    greedy policy a deterministic negamax replays the same two games (one per
    seat), collapsing the win-rate curve to a meaningless 0.0 / 0.5; randomized
    tie-breaks recover an informative win-rate distribution.
    """

    def __init__(self, depth: int = 3, seed: int = 0, randomize: bool = False):
        self.depth = depth
        self.randomize = randomize
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        # reconstruct canonical board: to-move = 1, opponent = 2
        board = (obs[0] * 1 + obs[1] * 2).astype(np.int8)
        if self.randomize:
            vals = _root_values(board, 1, self.depth)
            if vals:
                best = max(vals.values())
                cols = [c for c, v in vals.items() if v >= best - 1e-6]
                col = int(self.rng.choice(cols))
            else:
                col = -1
        else:
            _, col = negamax(board, 1, self.depth, -1e18, 1e18)
        if col < 0 or not mask[col]:
            col = int(self.rng.choice(np.where(mask)[0]))
        return int(col)


# ---------------------------------------------------------------------------
# Faithful Kaggle built-in negamax clone
# ---------------------------------------------------------------------------
def _flat_tuple_from_obs(obs: np.ndarray) -> tuple[int, ...]:
    """Tuple version for cached Kaggle-style search."""
    board = (obs[0] * 1 + obs[1] * 2).astype(np.int8)
    return tuple(int(x) for x in board.reshape(-1))


def _kg_drop_row(board: tuple[int, ...], column: int) -> int:
    for r in range(ROWS - 1, -1, -1):
        if board[column + (r * COLS)] == 0:
            return r
    return -1


def _kg_play_tuple(board: tuple[int, ...], column: int, mark: int) -> tuple[int, ...]:
    row = _kg_drop_row(board, column)
    if row < 0:
        return board
    idx = column + (row * COLS)
    return board[:idx] + (mark,) + board[idx + 1:]


def _kg_is_win(board: tuple[int, ...], column: int, mark: int, has_played: bool = True) -> bool:
    inarow = INAROW - 1
    if has_played:
        row = min(r for r in range(ROWS) if board[column + (r * COLS)] == mark)
    else:
        row = max(r for r in range(ROWS) if board[column + (r * COLS)] == 0)

    def count(offset_row: int, offset_column: int) -> int:
        for i in range(1, inarow + 1):
            r = row + offset_row * i
            c = column + offset_column * i
            if r < 0 or r >= ROWS or c < 0 or c >= COLS or board[c + (r * COLS)] != mark:
                return i - 1
        return inarow

    return (
        count(1, 0) >= inarow
        or (count(0, 1) + count(0, -1)) >= inarow
        or (count(-1, -1) + count(1, 1)) >= inarow
        or (count(-1, 1) + count(1, -1)) >= inarow
    )


def _kg_leaf_score(board: tuple[int, ...], column: int, mark: int, moves: int) -> float:
    row = _kg_drop_row(board, column)
    score = (SIZE + 1 - moves) / 2
    if column > 0 and board[row * COLS + column - 1] == mark:
        score += 1
    if column < COLS - 1 and board[row * COLS + column + 1] == mark:
        score += 1
    if row > 0 and board[(row - 1) * COLS + column] == mark:
        score += 1
    if row < ROWS - 2 and board[(row + 1) * COLS + column] == mark:
        score += 1
    return score


@lru_cache(maxsize=500_000)
def _kg_score(board: tuple[int, ...], mark: int, depth: int) -> float:
    """Cached Kaggle negamax score.

    Kaggle's recursive tie-break randomness changes only which equal-scoring
    column is returned, not the score propagated to the parent. Caching the
    score is therefore faithful for callers that still perform root tie-breaks.
    """
    moves = sum(cell != 0 for cell in board)
    if moves == SIZE:
        return 0.0

    for column in range(COLS):
        if board[column] == 0 and _kg_is_win(board, column, mark, False):
            return (SIZE + 1 - moves) / 2

    best_score = -SIZE
    for column in range(COLS):
        if board[column] != 0:
            continue
        if depth <= 0:
            score = _kg_leaf_score(board, column, mark, moves)
        else:
            next_board = _kg_play_tuple(board, column, mark)
            score = -_kg_score(next_board, 1 if mark == 2 else 2, depth - 1)
        if score > best_score:
            best_score = score
    return float(best_score)


def kaggle_negamax(
    board: list[int] | tuple[int, ...],
    mark: int,
    depth: int,
    rng: np.random.Generator,
):
    """Port of kaggle_environments.envs.connectx.connectx.negamax_agent.

    Kaggle scans columns left-to-right and breaks equal-score ties randomly.
    Keeping those details matters: the current learner was overfitting to a
    deterministic center-out alpha-beta opponent that is not the Kaggle agent.
    """
    board_t = tuple(board)
    moves = sum(cell != 0 for cell in board_t)
    if moves == SIZE:
        return 0, None

    for column in range(COLS):
        if board_t[column] == 0 and _kg_is_win(board_t, column, mark, False):
            return (SIZE + 1 - moves) / 2, column

    best_score = -SIZE
    best_column = None
    for column in range(COLS):
        if board_t[column] != 0:
            continue
        if depth <= 0:
            score = _kg_leaf_score(board_t, column, mark, moves)
        else:
            next_board = _kg_play_tuple(board_t, column, mark)
            score = -_kg_score(next_board, 1 if mark == 2 else 2, depth - 1)
        if score > best_score or (score == best_score and bool(rng.integers(0, 2))):
            best_score = score
            best_column = column
    return best_score, best_column


class KaggleNegamaxPolicy(Policy):
    """Local clone of Kaggle's built-in ``negamax`` ConnectX agent.

    The policy consumes canonical observations, so the side to move is always
    represented as mark 1 and the opponent as mark 2. That relabeling is
    equivalent for Kaggle's symmetric negamax code.
    """

    def __init__(self, depth: int = 4, seed: int = 0):
        self.depth = depth
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        _, col = kaggle_negamax(_flat_tuple_from_obs(obs), 1, self.depth, self.rng)
        if col is None or not mask[int(col)]:
            col = int(self.rng.choice(np.where(mask)[0]))
        return int(col)

    @staticmethod
    def cache_info():
        return _kg_score.cache_info()

    @staticmethod
    def clear_cache():
        _kg_score.cache_clear()
