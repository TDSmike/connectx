"""A fast depth-limited negamax (alpha-beta) ConnectX opponent.

Used as the strong benchmark for win-rate curves *during* training (Kaggle's
own `negamax` agent is faithful but too slow to call thousands of times).  It
operates on the canonical board recovered from a perspective-encoded obs, where
the player to move is always mark 1.
"""
from __future__ import annotations

import numpy as np

from .env import Policy, ROWS, COLS, INAROW

WIN = 100_000


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


class NegamaxPolicy(Policy):
    def __init__(self, depth: int = 3, seed: int = 0):
        self.depth = depth
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        # reconstruct canonical board: to-move = 1, opponent = 2
        board = (obs[0] * 1 + obs[1] * 2).astype(np.int8)
        _, col = negamax(board, 1, self.depth, -1e18, 1e18)
        if col < 0 or not mask[col]:
            col = int(self.rng.choice(np.where(mask)[0]))
        return int(col)
