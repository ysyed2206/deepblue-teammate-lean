"""S0: the reference engine.

A correct negamax engine built on python-chess. It is slow - the library's move
generation dominates its node cost - and that is accepted. It has three jobs:

1.  Insurance. There is always a validated submission that plays real chess.
2.  Oracle. Every faster implementation is differential-tested against it.
3.  Baseline. It is the first opponent any candidate has to actually beat.

Terminal handling follows the referee. The referee calls
``board.outcome(claim_draw=True)``, whose precedence is checkmate, then
insufficient material, then stalemate, then the claimable draws. Scoring a
checkmate as a fifty-move draw, or a stalemate as a material advantage, are
both half-point failures in a thirteen-game Swiss, so precedence here is
matched deliberately rather than approximately.
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

EXACT = 0
LOWER = 1  # fail-high: the true score is at least this
UPPER = 2  # fail-low: the true score is at most this

MAX_PLY = 128
TT_MAX_ENTRIES = 1_000_000

# The clock is consulted every this many nodes. It is deliberately small when
# the remaining budget is small: a 30 ms panic budget checked every 60 ms is
# not a budget at all.
CLOCK_CHECK_INTERVAL = 1024
CLOCK_CHECK_INTERVAL_LOW = 64
LOW_BUDGET_MS = 250.0

# An actual FIDE threefold is three occurrences of the same position.
REPETITION_RULE_THRESHOLD = 3

# Most engines additionally treat a position repeated *inside the search line*
# as a draw: a line that returns to a position it already visited has gained
# nothing, and searching the cycle again is wasted work.
#
# This applies to the SEARCH PATH ONLY. Applying it to a single prior
# occurrence in the actual game is wrong and was measured as such: once
# agent.py began correctly recording both our turn and the position our move
# creates, the game history doubled, every line touching any earlier game
# position scored as a draw, and the engine lost 13-2 to its own predecessor.
# One prior occurrence plus the current one is a twofold, and a twofold is not
# a draw.
TWOFOLD_SEARCH_HEURISTIC = True

# is_insufficient_material() costs a scan, so it is gated behind a popcount.
# Four pieces is the largest board on which the rule can possibly apply
# (K+B vs K+B and K+N vs K are the interesting cases).
INSUFFICIENT_MATERIAL_MAX_PIECES = 4

# Detecting stalemate at a quiescence leaf needs a legal-move probe, measured
# at 9.0 us against a ~60 us node - too expensive to run at every quiet leaf.
# It is gated on a piece count instead. The case that actually matters is
# capturing into stalemate while winning, which by construction happens in
# sparse positions; a stalemate with more than this many pieces on the board is
# not something engine play reaches from a won position. This is a heuristic
# gate on an exact test, and the residual risk is named rather than hidden.
STALEMATE_PROBE_MAX_PIECES = 6

# Near the fifty-move boundary a position's value depends on the halfmove
# clock, which the transposition key does not encode. Rather than widen the
# key, the table is simply not used in that region. It is rare enough that the
# cost is negligible and the failure it prevents is a lost half point.
TT_HALFMOVE_SAFE_LIMIT = 80

ORDER_TT = 1_000_000
ORDER_PROMOTION = 900_000
ORDER_CAPTURE = 800_000
ORDER_KILLER_1 = 700_000
ORDER_KILLER_2 = 690_000
HISTORY_CEILING = 600_000

MVV_VALUE = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 20,
}

popcount = chess.popcount


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

        # Two separate structures, deliberately. `game_counts` holds positions
        # that have ACTUALLY OCCURRED in the game and is permanent for the
        # game. `path_counts` holds positions on the line currently being
        # searched and is pushed and popped as the search walks. Conflating
        # them, as an earlier version did, made the engine simultaneously miss
        # real repetitions and invent hypothetical ones.
        self.game_counts: dict[object, int] = {}
        self.path_counts: dict[object, int] = {}

        self.nodes = 0
        self.deadline = 0.0
        self.check_interval = CLOCK_CHECK_INTERVAL
        self._checks = 0

    # -- actual game history ----------------------------------------------

    def record_game_position(self, board: chess.Board) -> None:
        """Record a position that has genuinely occurred in the game.

        Called for the position we are asked to move in, and again for the
        position our own move creates. The referee can claim a threefold on
        either, and we are only ever shown one of the two.
        """
        key = board._transposition_key()
        self.game_counts[key] = self.game_counts.get(key, 0) + 1

    def occurrences(self, key: object) -> int:
        return self.game_counts.get(key, 0) + self.path_counts.get(key, 0)

    def _path_push(self, key: object) -> None:
        self.path_counts[key] = self.path_counts.get(key, 0) + 1

    def _path_pop(self, key: object) -> None:
        remaining = self.path_counts.get(key, 0) - 1
        if remaining <= 0:
            self.path_counts.pop(key, None)
        else:
            self.path_counts[key] = remaining

    # -- entry point -------------------------------------------------------

    def search(
        self, board: chess.Board, soft_ms: float, hard_ms: float, max_depth: int = 64
    ) -> SearchResult:
        started = time.monotonic()
        self.deadline = started + hard_ms / 1000.0
        self.check_interval = (
            CLOCK_CHECK_INTERVAL_LOW if hard_ms <= LOW_BUDGET_MS else CLOCK_CHECK_INTERVAL
        )
        self.nodes = 0
        self._checks = 0
        self.path_counts.clear()

        legal = list(board.legal_moves)
        if not legal:
            return SearchResult(None, 0, 0, 0, 0.0)

        best_move = legal[0]
        best_score = 0
        completed_depth = 0
        principal_variation: list[chess.Move] = []
        previous_iteration_ms = 0.0
        growth = DEFAULT_ITERATION_GROWTH

        root_key = board._transposition_key()
        self._path_push(root_key)
        try:
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
                if abs(score) > MATE_THRESHOLD:
                    break
                if previous_iteration_ms > 1.0:
                    measured = iteration_ms / previous_iteration_ms
                    growth = min(MAX_ITERATION_GROWTH, max(MIN_ITERATION_GROWTH, measured))
                previous_iteration_ms = iteration_ms
                if elapsed + iteration_ms * growth > soft_ms:
                    break
        finally:
            self._path_pop(root_key)

        # The referee claims the fifty-move draw before asking us to move, so
        # a position at the limit is drawn whatever material we can still win.
        # Checkmate outranks it and is never overwritten. Same rule bug, and
        # same fix, as the one found in S1: a correctness fix, not a tuning
        # change to a frozen engine.
        if board.halfmove_clock >= 100 and abs(best_score) <= MATE_THRESHOLD:
            best_score = DRAW_SCORE

        elapsed = (time.monotonic() - started) * 1000.0
        return SearchResult(
            best_move, best_score, completed_depth, self.nodes, elapsed, principal_variation
        )

    def _search_root(
        self, board: chess.Board, depth: int, previous_best: chess.Move
    ) -> tuple[int, chess.Move, list[chess.Move]]:
        alpha, beta = -INFINITY, INFINITY
        best_move = previous_best
        best_score = -INFINITY
        best_line: list[chess.Move] = []

        for move in self._ordered_moves(board, 0, previous_best):
            board.push(move)
            key = board._transposition_key()
            self._path_push(key)
            try:
                score = -self._negamax(board, depth - 1, -beta, -alpha, 1, key)
            finally:
                self._path_pop(key)
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move
                best_line = [move] + self._collect_pv(board, move, depth - 1)
                if score > alpha:
                    alpha = score
        return best_score, best_move, best_line

    # -- terminal conditions ------------------------------------------------

    def _rule_draw(self, board: chess.Board, key: object, ply: int) -> int | None:
        """Draws that hold regardless of whose move it is and cannot be mate.

        Returns DRAW_SCORE if the position is drawn, otherwise None. The
        fifty-move rule is deliberately *not* handled here, because checkmate
        outranks it and detecting mate needs move generation.
        """
        if ply > 0:
            # The rule: three occurrences in total, counting the game history
            # and this search line together.
            if self.occurrences(key) >= REPETITION_RULE_THRESHOLD:
                return DRAW_SCORE
            # The heuristic: this line has already visited this position.
            if TWOFOLD_SEARCH_HEURISTIC and self.path_counts.get(key, 0) >= 2:
                return DRAW_SCORE
        # is_insufficient_material() is true only when NEITHER side can mate,
        # so it can safely pre-empt move generation.
        if popcount(board.occupied) <= INSUFFICIENT_MATERIAL_MAX_PIECES:
            if board.is_insufficient_material():
                return DRAW_SCORE
        return None

    def _fifty_move_draw(self, board: chess.Board, ply: int) -> int | None:
        """The fifty-move draw, with checkmate taking precedence over it.

        A position can be both checkmate and past the halfmove limit; the
        referee scores that as checkmate. Move generation only happens in the
        rare case where the clock is actually at the limit.
        """
        if board.halfmove_clock < 100:
            return None
        if board.is_checkmate():
            return -MATE_SCORE + ply
        return DRAW_SCORE

    # -- search -------------------------------------------------------------

    def _tick(self) -> None:
        self._checks += 1
        if self._checks >= self.check_interval:
            self._checks = 0
            if time.monotonic() >= self.deadline:
                raise TimeUp

    def _negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        key: object | None = None,
    ) -> int:
        self.nodes += 1
        self._tick()

        # The caller already computed this key to push it onto the search
        # path; recomputing it here cost 1.4 us on every node.
        if key is None:
            key = board._transposition_key()
        drawn = self._rule_draw(board, key, ply)
        if drawn is not None:
            return drawn
        fifty = self._fifty_move_draw(board, ply)
        if fifty is not None:
            return fifty

        original_alpha = alpha
        tt_usable = board.halfmove_clock < TT_HALFMOVE_SAFE_LIMIT
        tt_move: chess.Move | None = None
        if tt_usable:
            entry = self.transposition.get(key)
            if entry is not None:
                stored_depth, stored_score, bound, tt_move = entry
                stored_score = _score_from_tt(stored_score, ply)
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
            return -MATE_SCORE + ply if board.is_check() else DRAW_SCORE

        best_score = -INFINITY
        best_move: chess.Move | None = None
        for move in moves:
            board.push(move)
            child_key = board._transposition_key()
            self._path_push(child_key)
            try:
                score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1, child_key)
            finally:
                self._path_pop(child_key)
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            if best_score > alpha:
                alpha = best_score
            if alpha >= beta:
                self._record_cutoff(board, move, depth, ply)
                break

        if tt_usable:
            self._store(key, depth, best_score, original_alpha, beta, best_move, ply)
        return best_score

    def _quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        self._tick()
        if ply >= MAX_PLY - 1:
            return evaluate(board)

        # Quiescence must not be able to disagree with the referee about
        # whether a position is over. It does not do repetition bookkeeping,
        # because every move it makes is a capture or a promotion and neither
        # is reversible - a capture-only line cannot repeat a position.
        if popcount(board.occupied) <= INSUFFICIENT_MATERIAL_MAX_PIECES:
            if board.is_insufficient_material():
                return DRAW_SCORE
        fifty = self._fifty_move_draw(board, ply)
        if fifty is not None:
            return fifty

        if board.is_check():
            moves = self._ordered_moves(board, ply, None)
            if not moves:
                return -MATE_SCORE + ply
            best = -INFINITY
        else:
            moves = self._ordered_captures(board)
            if (
                not moves
                and popcount(board.occupied) <= STALEMATE_PROBE_MAX_PIECES
                and not any(board.generate_legal_moves())
            ):
                # No tactical move and no quiet move either: stalemate. Worth
                # exactly zero, not the material on the board.
                return DRAW_SCORE
            stand_pat = evaluate(board)
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            best = stand_pat

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

    # -- move ordering ------------------------------------------------------

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
            if move.promotion:
                score = ORDER_PROMOTION + MVV_VALUE.get(move.promotion, 0)
            elif board.is_capture(move):
                victim = board.piece_type_at(move.to_square)
                victim_value = MVV_VALUE[victim] if victim else MVV_VALUE[chess.PAWN]
                attacker = board.piece_type_at(move.from_square)
                score = ORDER_CAPTURE + victim_value * 16 - (MVV_VALUE[attacker] if attacker else 0)
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
            bonus = MVV_VALUE.get(move.promotion, 0) * 16 if move.promotion else 0
            scored.append(
                (victim_value * 16 - (MVV_VALUE[attacker] if attacker else 0) + bonus, move)
            )
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

    # -- transposition table ------------------------------------------------

    def _store(
        self,
        key: object,
        depth: int,
        score: int,
        original_alpha: int,
        beta: int,
        move: chess.Move | None,
        ply: int,
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
            self.transposition[key] = (depth, _score_to_tt(score, ply), bound, move)

    def _collect_pv(self, board: chess.Board, first: chess.Move, depth: int) -> list[chess.Move]:
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


def _score_to_tt(score: int, ply: int) -> int:
    """Make a mate score independent of where in the tree it was found.

    A mate score means "mate in N from here". Stored unadjusted, the same
    position reached at a different ply reads back the wrong distance.
    """
    if score > MATE_THRESHOLD:
        return score + ply
    if score < -MATE_THRESHOLD:
        return score - ply
    return score


def _score_from_tt(score: int, ply: int) -> int:
    if score > MATE_THRESHOLD:
        return score - ply
    if score < -MATE_THRESHOLD:
        return score + ply
    return score
