"""ConnectX (Connect-4) self-play environment.

A lightweight, fast engine that matches Kaggle ConnectX conventions
(7 columns x 6 rows, 4-in-a-row, marks 1/2, flat row-major board), plus

  * a canonical multi-channel state encoding (状态表示增强),
  * legal-action masks (action masking),
  * a single-agent `SelfPlayEnv` view that folds the opponent into the
    environment dynamics so DQN / PPO can train with a standard
    (obs, reward, done) loop while still learning by self-play.

Everything downstream (ppo / grpo) depends only on this file.
"""
from __future__ import annotations

import numpy as np

ROWS, COLS, INAROW = 6, 7, 4
N_ACTIONS = COLS
# state encoding: 3 channels (own pieces / opponent pieces / empty cells)
OBS_SHAPE = (3, ROWS, COLS)


class ConnectFour:
    """Connect-4 game engine.

    board: (ROWS, COLS) int8, value 0 empty / 1 player-1 / 2 player-2.
    Row 0 is the TOP row (matching Kaggle's row-major flat board), so a
    dropped piece falls to the largest empty row index in its column.
    """

    __slots__ = ("board", "player", "winner", "done", "last")

    def __init__(self):
        self.reset()

    def reset(self) -> "ConnectFour":
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.player = 1            # player to move (1 or 2)
        self.winner = 0           # 0 = ongoing/none, 1/2 = winner, -1 = draw
        self.done = False
        self.last = None          # (row, col) of last move
        return self

    # --- queries -------------------------------------------------------
    def legal_mask(self) -> np.ndarray:
        """Boolean (COLS,) mask, True where a piece can be dropped."""
        return self.board[0] == 0

    def legal_moves(self) -> np.ndarray:
        return np.where(self.board[0] == 0)[0]

    def _drop_row(self, col: int) -> int:
        for r in range(ROWS - 1, -1, -1):
            if self.board[r, col] == 0:
                return r
        return -1

    # --- transition ----------------------------------------------------
    def step(self, col: int) -> "ConnectFour":
        """Play `col` for the current player and advance the game."""
        assert not self.done, "step() called on a finished game"
        r = self._drop_row(col)
        if r < 0:                                  # illegal: instant loss
            self.winner = 3 - self.player
            self.done = True
            return self
        self.board[r, col] = self.player
        self.last = (r, col)
        if self._is_win(r, col, self.player):
            self.winner = self.player
            self.done = True
        elif not (self.board[0] == 0).any():
            self.winner = -1                        # board full -> draw
            self.done = True
        else:
            self.player = 3 - self.player           # swap turn
        return self

    def _is_win(self, r: int, c: int, p: int) -> bool:
        b = self.board
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            cnt = 1
            for s in (1, -1):
                rr, cc = r + dr * s, c + dc * s
                while 0 <= rr < ROWS and 0 <= cc < COLS and b[rr, cc] == p:
                    cnt += 1
                    rr += dr * s
                    cc += dc * s
            if cnt >= INAROW:
                return True
        return False

    def copy(self) -> "ConnectFour":
        g = ConnectFour.__new__(ConnectFour)
        g.board = self.board.copy()
        g.player = self.player
        g.winner = self.winner
        g.done = self.done
        g.last = self.last
        return g


# ---------------------------------------------------------------------------
# Encodings
# ---------------------------------------------------------------------------
def encode(board: np.ndarray, player: int) -> np.ndarray:
    """Canonical (3, ROWS, COLS) float32 tensor from `player`'s perspective.

    ch0 = my pieces, ch1 = opponent pieces, ch2 = empty cells.  Because the
    board is always encoded relative to the player to move, ONE network can
    play both seats.
    """
    own = (board == player)
    opp = (board == (3 - player))
    empty = (board == 0)
    return np.stack([own, opp, empty], axis=0).astype(np.float32)


def state_key(board: np.ndarray, player: int) -> bytes:
    """Hashable canonical key for tabular methods.

    Relabels the board so the player-to-move is always '1', giving a single
    shared Q-table across both seats.
    """
    if player == 1:
        rel = board
    else:
        rel = board.copy()
        rel[board == 1] = 2
        rel[board == 2] = 1
    return rel.tobytes()


# ---------------------------------------------------------------------------
# Policy interface + simple baselines
# ---------------------------------------------------------------------------
class Policy:
    """Anything that maps (canonical obs, legal mask) -> action column."""

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:  # pragma: no cover
        raise NotImplementedError


class RandomPolicy(Policy):
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        legal = np.where(mask)[0]
        return int(self.rng.choice(legal))


# ---------------------------------------------------------------------------
# Single-agent self-play view
# ---------------------------------------------------------------------------
class SelfPlayEnv:
    """Single-agent MDP view of self-play Connect-4.

    The learner controls one seat; an `opponent` Policy controls the other and
    is folded into env dynamics.  Observations are always returned from the
    learner's perspective, so the learner faces a standard
    (obs, reward, done) loop while genuinely training by self-play.

    Reward: +1 learner wins, -1 learner loses (incl. opponent win on its
    reply), 0 draw / non-terminal.  The learner's seat is randomised each
    episode for balance.
    """

    def __init__(self, seed: int = 0, tactical_coef: float = 0.0):
        self.game = ConnectFour()
        self.rng = np.random.default_rng(seed)
        self.opponent: Policy | None = None
        self.learner_seat = 1
        # dense tactical shaping: penalize the two fatal Connect-4 blunders a
        # terminal-only reward teaches too slowly to a model-free policy —
        # (a) not taking an available immediate win, and (b) leaving the
        # opponent an immediate win (a missed block or a self-made threat).
        # 0.0 disables it (pure +1/-1/0).  Kept << 1 so the game outcome still
        # dominates; this only sharpens credit assignment onto the blunder ply.
        self.tactical_coef = float(tactical_coef)

    def _has_immediate_win(self, player: int) -> bool:
        g = self.game
        for c in range(COLS):
            if g.board[0, c] != 0:
                continue
            r = g._drop_row(c)
            g.board[r, c] = player
            won = g._is_win(r, c, player)
            g.board[r, c] = 0
            if won:
                return True
        return False

    def reset(self, opponent: Policy, learner_seat: int | None = None):
        self.opponent = opponent
        self.game.reset()
        self.learner_seat = (
            int(self.rng.integers(1, 3)) if learner_seat is None else learner_seat
        )
        if self.game.player != self.learner_seat:   # opponent moves first
            self._opponent_move()
        return self._obs()

    # internal helpers --------------------------------------------------
    def _obs(self):
        g = self.game
        return encode(g.board, self.learner_seat), g.legal_mask()

    def _opponent_move(self):
        g = self.game
        obs = encode(g.board, g.player)
        a = self.opponent.act(obs, g.legal_mask())
        g.step(int(a))

    def _terminal(self, info):
        g = self.game
        if g.winner == self.learner_seat:
            r = 1.0
        elif g.winner == -1:
            r = 0.0
        else:
            r = -1.0
        info["winner"] = g.winner
        return self._obs(), r, True, info

    # gym-like API ------------------------------------------------------
    def step(self, action: int):
        g = self.game
        # tactical context BEFORE the learner moves (it is the learner's turn)
        had_win = self.tactical_coef and self._has_immediate_win(self.learner_seat)
        g.step(int(action))
        if g.done:                                   # learner's move ended it
            return self._terminal({})
        shaped = 0.0
        if self.tactical_coef:
            if had_win:                              # had a win, didn't take it
                shaped -= self.tactical_coef
            if self._has_immediate_win(g.player):    # left opponent an immediate win
                shaped -= self.tactical_coef
        self._opponent_move()
        if g.done:
            obs, r, done, info = self._terminal({})
            return obs, r + shaped, done, info
        obs, mask = self._obs()
        return (obs, mask), shaped, False, {}
