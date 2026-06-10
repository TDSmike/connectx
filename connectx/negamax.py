"""Local clone of Kaggle's built-in ConnectX ``negamax`` agent.

A faithful port of ``kaggle_environments.envs.connectx.connectx.negamax_agent``:
it scans columns left-to-right and breaks equal-score ties randomly. Keeping
those details matters — it is the exact opponent we ultimately want to beat, and
its random tie-breaking exercises many lines so a greedy policy can't memorize a
single deterministic game. Used both as a self-play curriculum opponent and as
the evaluation benchmark.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .env import Policy, ROWS, COLS, INAROW

SIZE = ROWS * COLS


def _flat_board(obs: np.ndarray) -> tuple[int, ...]:
    """Canonical obs -> flat row-major board tuple (to-move = 1, opponent = 2)."""
    board = (obs[0] * 1 + obs[1] * 2).astype(np.int8)
    return tuple(int(x) for x in board.reshape(-1))


def _drop_row(board: tuple[int, ...], column: int) -> int:
    for r in range(ROWS - 1, -1, -1):
        if board[column + r * COLS] == 0:
            return r
    return -1


def _play(board: tuple[int, ...], column: int, mark: int) -> tuple[int, ...]:
    row = _drop_row(board, column)
    if row < 0:
        return board
    idx = column + row * COLS
    return board[:idx] + (mark,) + board[idx + 1:]


def _is_win(board: tuple[int, ...], column: int, mark: int, has_played: bool = True) -> bool:
    inarow = INAROW - 1
    if has_played:
        row = min(r for r in range(ROWS) if board[column + r * COLS] == mark)
    else:
        row = max(r for r in range(ROWS) if board[column + r * COLS] == 0)

    def count(dr: int, dc: int) -> int:
        for i in range(1, inarow + 1):
            r, c = row + dr * i, column + dc * i
            if r < 0 or r >= ROWS or c < 0 or c >= COLS or board[c + r * COLS] != mark:
                return i - 1
        return inarow

    return (count(1, 0) >= inarow
            or count(0, 1) + count(0, -1) >= inarow
            or count(-1, -1) + count(1, 1) >= inarow
            or count(-1, 1) + count(1, -1) >= inarow)


def _leaf_score(board: tuple[int, ...], column: int, mark: int, moves: int) -> float:
    row = _drop_row(board, column)
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
def _score(board: tuple[int, ...], mark: int, depth: int) -> float:
    """Cached negamax score. Tie-break randomness only changes which equal-best
    column is returned at the root, not the score propagated up — so caching the
    score is faithful for callers that still tie-break at the root."""
    moves = sum(cell != 0 for cell in board)
    if moves == SIZE:
        return 0.0
    for column in range(COLS):
        if board[column] == 0 and _is_win(board, column, mark, False):
            return (SIZE + 1 - moves) / 2
    best = -SIZE
    for column in range(COLS):
        if board[column] != 0:
            continue
        if depth <= 0:
            s = _leaf_score(board, column, mark, moves)
        else:
            s = -_score(_play(board, column, mark), 3 - mark, depth - 1)
        best = max(best, s)
    return float(best)


class KaggleNegamaxPolicy(Policy):
    """Kaggle's built-in ``negamax`` agent, over canonical observations."""

    def __init__(self, depth: int = 4, seed: int = 0):
        self.depth = depth
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        col = self._best_column(_flat_board(obs))
        if col is None or not mask[int(col)]:
            col = int(self.rng.choice(np.where(mask)[0]))
        return int(col)

    def _best_column(self, board: tuple[int, ...]):
        moves = sum(cell != 0 for cell in board)
        if moves == SIZE:
            return None
        # 1) take any immediate win — first in the left-to-right scan, no rng
        for column in range(COLS):
            if board[column] == 0 and _is_win(board, column, 1, False):
                return column
        # 2) otherwise the best-scoring column, with Kaggle's random tie-break
        best, best_col = -SIZE, None
        for column in range(COLS):
            if board[column] != 0:
                continue
            if self.depth <= 0:
                s = _leaf_score(board, column, 1, moves)
            else:
                s = -_score(_play(board, column, 1), 2, self.depth - 1)
            if s > best or (s == best and bool(self.rng.integers(0, 2))):
                best, best_col = s, column
        return best_col
