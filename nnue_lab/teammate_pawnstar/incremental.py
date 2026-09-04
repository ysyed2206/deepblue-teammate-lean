"""Incremental (and optionally lazy) accumulator maintenance across make/unmake.

Mirrors the donor's ``Network::Update`` design (``src/nnue.h``): on an ordinary
transition, only the squares whose (colour, piece_type) differ between the
before/after position are touched -- captures, en passant, castling,
promotion and capture-promotion all "fall out" of the same generic diff, with
no special-casing, because they are all just "this square's occupant
changed." A king move that crosses a king-bucket boundary is the one case
that needs a full rebuild of that one perspective (the other perspective is
still diffed normally).

This module works on ``chess.Board`` objects (a before/after pair) rather
than Deep Blue's own board representation, matching the mission's own
"standalone prototype, not a live-engine edit" scope; ``INTEGRATION_CONTRACT.md``
describes the small adapter Deep Blue's own make/unmake would need to drive
this with its own delta instead of two full boards.

Priority 7 (lazy/deferred accumulators): the donor defers ``Update`` until an
evaluation actually reads the accumulator, so a node that cuts off before
evaluating pays nothing. ``NNUEState`` supports the same pattern: pass
``lazy=True`` to ``make_move``/``unmake_move`` to only *record* the pending
transition, and ``evaluate()`` will collapse any queued transitions (in
order) before reading the accumulator. This is transparent to the caller --
the returned score is identical either way -- it only changes *when* the
update work happens.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

import features as feat
from reference_eval import evaluate_from_accumulators, refresh_accumulator
from weights import NNUEWeights


def _piece_set(board: chess.Board) -> dict[int, tuple[int, int]]:
    return {square: (colour, piece_type) for square, colour, piece_type in feat.board_pieces(board)}


def _diff_side(weights: NNUEWeights, acc: np.ndarray, board_from: chess.Board, board_to: chess.Board, perspective: int, king_square: int) -> np.ndarray:
    """Apply the (colour, piece_type)-per-square diff between two positions to
    one perspective's accumulator, assuming that perspective's king bucket is
    unchanged (checked by the caller). Every move type -- quiet, capture, en
    passant, castling, promotion, capture-promotion -- reduces to some set of
    squares whose occupant differs, so no move-type branching is needed here."""
    before = _piece_set(board_from)
    after = _piece_set(board_to)
    removed = [(sq, *before[sq]) for sq in before.keys() - after.keys()]
    same_square_changed = [(sq, *after[sq]) for sq in (before.keys() & after.keys()) if before[sq] != after[sq]]
    removed += [(sq, *before[sq]) for sq, _, _ in same_square_changed]
    added = [(sq, *after[sq]) for sq in after.keys() - before.keys()] + same_square_changed

    for square, colour, piece_type in removed:
        row = feat.feature_row(colour, piece_type, square, king_square, perspective)
        acc = (acc - weights.feature_weights[row]).astype(np.int16)
    for square, colour, piece_type in added:
        row = feat.feature_row(colour, piece_type, square, king_square, perspective)
        acc = (acc + weights.feature_weights[row]).astype(np.int16)
    return acc


@dataclass
class TransitionStats:
    """Move-category counters, incremented by ``NNUEState`` as it classifies
    each transition it is asked to apply (used by the correctness gate)."""

    quiet: int = 0
    capture: int = 0
    en_passant: int = 0
    king_same_bucket: int = 0
    king_bucket_change: int = 0
    castle_kingside: int = 0
    castle_queenside: int = 0
    promotion: int = 0
    capture_promotion: int = 0
    underpromotion: int = 0

    def classify(self, board_before: chess.Board, move: chess.Move) -> None:
        if board_before.is_en_passant(move):
            self.en_passant += 1
        elif board_before.is_castling(move):
            if board_before.is_kingside_castling(move):
                self.castle_kingside += 1
            else:
                self.castle_queenside += 1
        elif move.promotion is not None:
            is_capture = board_before.is_capture(move)
            if is_capture:
                self.capture_promotion += 1
            else:
                self.promotion += 1
            if move.promotion != chess.QUEEN:
                self.underpromotion += 1
        elif board_before.is_capture(move):
            self.capture += 1
        else:
            self.quiet += 1

        moved_piece = board_before.piece_at(move.from_square)
        if moved_piece is not None and moved_piece.piece_type == chess.KING:
            perspective = feat.WHITE if moved_piece.color == chess.WHITE else feat.BLACK
            before_bucket = feat.king_bucket(move.from_square, perspective)
            after_bucket = feat.king_bucket(move.to_square, perspective)
            if before_bucket == after_bucket:
                self.king_same_bucket += 1
            else:
                self.king_bucket_change += 1


@dataclass
class _PendingTransition:
    board_from: chess.Board
    board_to: chess.Board


@dataclass
class NNUEState:
    """Incrementally-maintained accumulator pair for one search line.

    ``white_acc``/``black_acc`` are always *current* unless ``lazy`` updates
    are pending (tracked in ``_pending``), in which case ``evaluate()``
    collapses them first. ``_history`` is a stack of applied (or pending)
    transitions, enabling exact, order-correct ``unmake_move()``.
    """

    weights: NNUEWeights
    white_acc: np.ndarray
    black_acc: np.ndarray
    white_bucket: int
    black_bucket: int
    _history: list[_PendingTransition] = field(default_factory=list)
    _pending: list[_PendingTransition] = field(default_factory=list)

    @classmethod
    def from_position(cls, weights: NNUEWeights, board: chess.Board) -> "NNUEState":
        white_acc = refresh_accumulator(weights, board, feat.WHITE)
        black_acc = refresh_accumulator(weights, board, feat.BLACK)
        wb, bb = feat.king_buckets_for(board)
        return cls(weights, white_acc, black_acc, wb, bb)

    def _apply(self, board_from: chess.Board, board_to: chess.Board) -> None:
        white_king, black_king = feat.king_squares(board_to)
        new_wb, new_bb = feat.king_buckets_for(board_to)

        if new_wb == self.white_bucket:
            self.white_acc = _diff_side(self.weights, self.white_acc, board_from, board_to, feat.WHITE, white_king)
        else:
            self.white_acc = refresh_accumulator(self.weights, board_to, feat.WHITE)
        self.white_bucket = new_wb

        if new_bb == self.black_bucket:
            self.black_acc = _diff_side(self.weights, self.black_acc, board_from, board_to, feat.BLACK, black_king)
        else:
            self.black_acc = refresh_accumulator(self.weights, board_to, feat.BLACK)
        self.black_bucket = new_bb

    def make_move(self, board_before: chess.Board, board_after: chess.Board, *, lazy: bool = False) -> None:
        """Advance the state from ``board_before`` to ``board_after``. With
        ``lazy=True`` the update is only queued (donor-style deferred
        accumulator maintenance); it is applied automatically the next time
        ``evaluate()`` is called, or immediately by a following
        non-lazy ``make_move``/``unmake_move``.

        Both boards are snapshotted (``.copy()``) on entry, not held by
        reference: a lazy caller is expected to keep mutating its own board
        across several further moves before the deferred update is ever
        settled, so holding a live reference would silently read whatever
        position the caller's board has drifted to by settle time, not the
        position at the moment of this call."""
        transition = _PendingTransition(board_before.copy(), board_after.copy())
        self._history.append(transition)
        if lazy:
            self._pending.append(transition)
        else:
            self._settle()
            self._apply(board_before, board_after)

    def unmake_move(self, *, lazy: bool = False) -> None:
        """Undo the most recent ``make_move``, by applying the exact inverse
        transition (swap from/to) -- not by restoring a cached snapshot, so
        this genuinely re-exercises the diff logic in reverse (the same
        property the donor's own incremental test checks)."""
        if not self._history:
            raise RuntimeError("unmake_move() with no matching make_move")
        transition = self._history.pop()
        if self._pending and self._pending[-1] is transition:
            # Never actually applied -- just drop the pending entry.
            self._pending.pop()
            return
        if lazy:
            self._pending.append(_PendingTransition(transition.board_to, transition.board_from))
        else:
            self._settle()
            self._apply(transition.board_to, transition.board_from)

    def _settle(self) -> None:
        """Apply any queued lazy transitions, in order, then clear the queue."""
        for transition in self._pending:
            self._apply(transition.board_from, transition.board_to)
        self._pending.clear()

    def evaluate(self, side_to_move: bool) -> int:
        self._settle()
        return evaluate_from_accumulators(self.weights, self.white_acc, self.black_acc, side_to_move)
