"""S0: the reference engine.

A correct, shippable negamax engine built on python-chess. It is slow - the
library's move generation dominates its node cost - and that is accepted here.
This engine has three jobs, in order:

1.  Insurance. There is always a validated submission that plays real chess.
2.  Oracle. Every faster implementation is differential-tested against it.
3.  Baseline. It is the first opponent any candidate has to actually beat.

Design commitments that exist for reliability rather than strength:

*   The best move from the last *completed* iterative-deepening pass is always
    retained. An aborted pass is discarded entirely, never partially trusted.
*   The clock is checked inside the search, not only between iterations.
*   Repetition is scored as a draw using the real game history, so a won
    position is never shuffled into a threefold claim by the referee.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import chess

from deepblue.constants import DRAW_SCORE, INFINITY, MATE_SCORE, MATE_THRESHOLD
from deepblue.evaluation import evaluate
from deepblue.time_manager import (
    DEFAULT_ITERATION_GROWTH,
    MAX_ITERATION_GROWTH,
    MIN_ITERATION_GROWTH,
)

# Transposition table bound types.
EXACT = 0
LOWER = 1  # fail-high: the true score is at least this
UPPER = 2  # fail-low: the true score is at most this

MAX_PLY = 128

# The table is a plain dict. At roughly 200 bytes per entry this cap is a few
# hundred MB, comfortably inside the 2 GB the platform allows, and clearing on
# overflow is preferred to an unbounded dict that could be OOM-killed mid-game.
TT_MAX_ENTRIES = 1_000_000

# How often the clock is consulted, in nodes. Small enough that a single check
# interval cannot overrun the hard limit, large enough not to cost measurably.
CLOCK_CHECK_INTERVAL = 1024

# Move ordering bands, widely separated so no two categories can interleave.
ORDER_TT = 1_000_000
ORDER_PROMOTION = 900_000
ORDER_CAPTURE = 800_000
ORDER_KILLER_1 = 700_000
ORDER_KILLER_2 = 690_000
HISTORY_CEILING = 600_000

# Most Valuable Victim / Least Valuable Attacker, in the usual piece order.
MVV_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 20}


class TimeUp(Exception):
    """Raised inside the search when the hard deadline passes."""


@dataclass
class SearchResult:
    move: chess.Move | None
    score: int
    depth: int
    nodes: int
    elapsed_ms: float
    principal_variation: list[chess.Move] = field(default_factory=list)


class Engine:
    """One engine instance. One per game; state persists across moves."""

    def __init__(self) -> None:
        self.transposition: dict[object, tuple[int, int, int, chess.Move | None]] = {}
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY)]
        self.history: dict[tuple[int, int, int], int] = {}
        # Position keys that have actually occurred in the game, with counts.
        self.game_keys: dict[object, int] = {}
        self.nodes = 0
        self.deadline = 0.0
        self._checks = 0

    # -- game bookkeeping ------------------------------------------------

    def record_position(self, board: chess.Board) -> None:
        key = board._transposition_key()
        self.game_keys[key] = self.game_keys.get(key, 0) + 1

    # -- entry point -----------------------------------------------------

    def search(
        self,
        board: chess.Board,
        soft_ms: float,
        hard_ms: float,
        max_depth: int = 64,
    ) -> SearchResult:
        started = time.monotonic()
        self.deadline = started + hard_ms / 1000.0
        self.nodes = 0
        self._checks = 0

        legal = list(board.legal_moves)
        if not legal:
            return SearchResult(None, 0, 0, 0, 0.0)

        best_move = legal[0]
        best_score = 0
        completed_depth = 0
        principal_variation: list[chess.Move] = []
        previous_iteration_ms = 0.0
        growth = DEFAULT_ITERATION_GROWTH

        for depth in range(1, max_depth + 1):
            iteration_started = time.monotonic()
            try:
                score, move, line = self._search_root(board, depth, best_move)
            except TimeUp:
                break
            best_score, best_move, principal_variation = score, move, line
            completed_depth = depth

            now = time.monotonic()
            elapsed = (now - started) * 1000.0
            iteration_ms = (now - iteration_started) * 1000.0

            if elapsed >= soft_ms:
                break
            # A forced mate is found; deeper passes cannot improve on it.
            if abs(score) > MATE_THRESHOLD:
                break
            # Learn how fast passes are growing in this position, then refuse
            # to start one the estimate says cannot finish inside the budget.
            if previous_iteration_ms > 1.0:
                measured = iteration_ms / previous_iteration_ms
                growth = min(MAX_ITERATION_GROWTH, max(MIN_ITERATION_GROWTH, measured))
            previous_iteration_ms = iteration_ms
            if elapsed + iteration_ms * growth > soft_ms:
                break

        elapsed = (time.monotonic() - started) * 1000.0
        return SearchResult(best_move, best_score, completed_depth, self.nodes, elapsed, principal_variation)

    def _search_root(
        self, board: chess.Board, depth: int, previous_best: chess.Move
    ) -> tuple[int, chess.Move, list[chess.Move]]:
        alpha, beta = -INFINITY, INFINITY
        best_move = previous_best
        best_score = -INFINITY
        best_line: list[chess.Move] = []

        moves = self._ordered_moves(board, 0, previous_best)
        for move in moves:
            board.push(move)
            self.game_keys_push(board)
            try:
                score = -self._negamax(board, depth - 1, -beta, -alpha, 1)
            finally:
                self.game_keys_pop(board)
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move
                best_line = [move] + self._collect_pv(board, move, depth - 1)
                if score > alpha:
                    alpha = score
        return best_score, best_move, best_line

    # -- search ----------------------------------------------------------

    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        self._checks += 1
        if self._checks >= CLOCK_CHECK_INTERVAL:
            self._checks = 0
            if time.monotonic() >= self.deadline:
                raise TimeUp

        # A position repeated inside the search, or already seen in the game, is
        # a draw. Checked before anything else so a won position is never
        # shuffled into a referee-claimed threefold.
        #
        # The caller has already counted this node's key, so its own occurrence
        # is included in the count. Two or more means the position has genuinely
        # occurred before, either earlier in the game or higher up this line.
        key = board._transposition_key()
        if ply > 0 and self.game_keys.get(key, 0) >= 2:
            return DRAW_SCORE
        if board.halfmove_clock >= 100:
            return DRAW_SCORE

        original_alpha = alpha
        entry = self.transposition.get(key)
        tt_move: chess.Move | None = None
        if entry is not None:
            stored_depth, stored_score, bound, tt_move = entry
            if stored_depth >= depth and ply > 0:
                if bound == EXACT:
                    return stored_score
                if bound == LOWER and stored_score > alpha:
                    alpha = stored_score
                elif bound == UPPER and stored_score < beta:
                    beta = stored_score
                if alpha >= beta:
                    return stored_score

        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply)

        moves = self._ordered_moves(board, ply, tt_move)
        if not moves:
            # No legal moves: checkmate is scored by distance so the engine
            # prefers the fastest mate and the slowest loss.
            return -MATE_SCORE + ply if board.is_check() else DRAW_SCORE

        best_score = -INFINITY
        best_move: chess.Move | None = None
        for move in moves:
            board.push(move)
            self.game_keys_push(board)
            try:
                score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                self.game_keys_pop(board)
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            if best_score > alpha:
                alpha = best_score
            if alpha >= beta:
                self._record_cutoff(board, move, depth, ply)
                break

        self._store(key, depth, best_score, original_alpha, beta, best_move)
        return best_score

    def _quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        self._checks += 1
        if self._checks >= CLOCK_CHECK_INTERVAL:
            self._checks = 0
            if time.monotonic() >= self.deadline:
                raise TimeUp
        if ply >= MAX_PLY - 1:
            return evaluate(board)

        in_check = board.is_check()
        if in_check:
            # In check every move is a candidate, otherwise the search would
            # stand pat on a position where the king is about to be taken.
            moves = self._ordered_moves(board, ply, None)
            if not moves:
                return -MATE_SCORE + ply
        else:
            stand_pat = evaluate(board)
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            best = stand_pat
            moves = self._ordered_captures(board)

        if in_check:
            best = -INFINITY

        for move in moves:
            board.push(move)
            try:
                score = -self._quiescence(board, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    # -- move ordering ---------------------------------------------------

    def _ordered_moves(
        self, board: chess.Board, ply: int, tt_move: chess.Move | None
    ) -> list[chess.Move]:
        killer_1, killer_2 = self.killers[ply] if ply < MAX_PLY else (None, None)
        history = self.history
        turn = board.turn
        scored: list[tuple[int, chess.Move]] = []
        for move in board.legal_moves:
            if move == tt_move:
                scored.append((ORDER_TT, move))
                continue
            score = 0
            if move.promotion:
                score = ORDER_PROMOTION + MVV_VALUE.get(move.promotion, 0)
            elif board.is_capture(move):
                victim = board.piece_type_at(move.to_square)
                victim_value = MVV_VALUE[victim] if victim else MVV_VALUE[chess.PAWN]
                attacker = board.piece_type_at(move.from_square)
                attacker_value = MVV_VALUE[attacker] if attacker else 0
                score = ORDER_CAPTURE + victim_value * 16 - attacker_value
            elif move == killer_1:
                score = ORDER_KILLER_1
            elif move == killer_2:
                score = ORDER_KILLER_2
            else:
                piece = board.piece_type_at(move.from_square)
                value = history.get((turn, piece or 0, move.to_square), 0)
                score = value if value < HISTORY_CEILING else HISTORY_CEILING
            scored.append((score, move))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [move for _, move in scored]

    def _ordered_captures(self, board: chess.Board) -> list[chess.Move]:
        scored: list[tuple[int, chess.Move]] = []
        for move in board.generate_legal_captures():
            victim = board.piece_type_at(move.to_square)
            victim_value = MVV_VALUE[victim] if victim else MVV_VALUE[chess.PAWN]
            attacker = board.piece_type_at(move.from_square)
            attacker_value = MVV_VALUE[attacker] if attacker else 0
            bonus = MVV_VALUE.get(move.promotion, 0) * 16 if move.promotion else 0
            scored.append((victim_value * 16 - attacker_value + bonus, move))
        # Quiet promotions are forcing and belong in quiescence, but scanning
        # every legal move to find them would double the cost of this function.
        # Only look when a pawn is actually one rank from promoting.
        promotion_rank = chess.BB_RANK_7 if board.turn == chess.WHITE else chess.BB_RANK_2
        if board.pawns & board.occupied_co[board.turn] & promotion_rank:
            for move in board.generate_legal_moves():
                if move.promotion == chess.QUEEN and not board.is_capture(move):
                    scored.append((MVV_VALUE[chess.QUEEN] * 16, move))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [move for _, move in scored]

    def _record_cutoff(self, board: chess.Board, move: chess.Move, depth: int, ply: int) -> None:
        if board.is_capture(move) or move.promotion:
            return
        if ply < MAX_PLY:
            slot = self.killers[ply]
            if slot[0] != move:
                slot[1] = slot[0]
                slot[0] = move
        piece = board.piece_type_at(move.from_square)
        key = (board.turn, piece or 0, move.to_square)
        self.history[key] = self.history.get(key, 0) + depth * depth

    # -- transposition table ---------------------------------------------

    def _store(
        self,
        key: object,
        depth: int,
        score: int,
        original_alpha: int,
        beta: int,
        move: chess.Move | None,
    ) -> None:
        if len(self.transposition) >= TT_MAX_ENTRIES:
            self.transposition.clear()
        if score <= original_alpha:
            bound = UPPER
        elif score >= beta:
            bound = LOWER
        else:
            bound = EXACT
        existing = self.transposition.get(key)
        if existing is None or existing[0] <= depth:
            self.transposition[key] = (depth, score, bound, move)

    # -- repetition bookkeeping during search -----------------------------

    def game_keys_push(self, board: chess.Board) -> None:
        key = board._transposition_key()
        self.game_keys[key] = self.game_keys.get(key, 0) + 1

    def game_keys_pop(self, board: chess.Board) -> None:
        key = board._transposition_key()
        count = self.game_keys.get(key, 0)
        if count <= 1:
            self.game_keys.pop(key, None)
        else:
            self.game_keys[key] = count - 1

    def _collect_pv(self, board: chess.Board, first: chess.Move, depth: int) -> list[chess.Move]:
        """Walk the transposition table to recover the principal variation."""
        line: list[chess.Move] = []
        pushed = 0
        board.push(first)
        pushed += 1
        try:
            for _ in range(depth):
                entry = self.transposition.get(board._transposition_key())
                if entry is None or entry[3] is None:
                    break
                move = entry[3]
                if move not in board.legal_moves:
                    break
                line.append(move)
                board.push(move)
                pushed += 1
        finally:
            for _ in range(pushed):
                board.pop()
        return line
