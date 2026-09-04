"""Donor-compatible feature mapper for the Pawnstar-v12 NNUE format.

Standalone: this module has no dependency on the rest of ``nnue_lab`` and does
not import or modify it. See ``PAWNSTAR_FORMAT.md`` in this directory for the
verified source citations behind every constant and formula below.

Architecture (verified from github.com/jonny-reckless/pawnstar `src/nnue.h`
and `nnue/README.md`, commit history through the shipped v12 net):

    768 inputs per (perspective, king bucket)  ->  1024-wide feature transformer
    8 king buckets (file-pair x board-half), selected per perspective by that
    perspective's own king square, oriented to it.

Feature row (donor `FeatureRow`, `src/nnue.h`):

    white perspective: bucket*768 + colour*384     + (piece_type)*64 + square
    black perspective: bucket*768 + (1-colour)*384 + (piece_type)*64 + (square ^ 0x38)

``colour`` is absolute (0=white, 1=black); ``piece_type`` is 0=pawn .. 5=king
(donor's `kPawn`=0 in `constants.h`, so `piece - kPawn` is already 0..5 despite
the README's 1..6 "pt" prose numbering -- verified directly against the enum).
``0x38`` (=56) is the donor's `kRankFlip`, a vertical mirror.

King bucket (donor `kKingBucketMap`, `src/nnue.h`): index by the perspective's
own king square oriented to that perspective (white: as-is; black: square ^
0x38), giving `file // 2` (a/b->0, c/d->1, e/f->2, g/h->3) plus 4 when the
oriented square is on the perspective's advanced half (oriented rank >= 5,
i.e. square index >= 32).
"""

from __future__ import annotations

from collections.abc import Iterable

import chess
import numpy as np

WHITE = 0
BLACK = 1
NUM_PERSPECTIVES = 2

NUM_KING_BUCKETS = 8
INPUT_SIZE = 768  # 2 colours * 6 piece types * 64 squares, per (perspective, bucket)
FEATURE_ROWS = INPUT_SIZE * NUM_KING_BUCKETS  # 6144
HIDDEN_SIZE = 1024

RANK_FLIP = 0x38  # donor kRankFlip: vertical mirror (square ^ 56)

MAX_PIECES = 32
PADDING_FEATURE = FEATURE_ROWS  # sentinel for an unused slot in a padded batch


def orient_square(square: int, perspective: int) -> int:
    """Orient a square into ``perspective``'s own frame (donor: identity for
    white, ``square ^ kRankFlip`` for black)."""
    if not 0 <= square < 64:
        raise ValueError(f"square out of range: {square}")
    if perspective not in (WHITE, BLACK):
        raise ValueError(f"perspective out of range: {perspective}")
    return square if perspective == WHITE else square ^ RANK_FLIP


def king_bucket(king_square: int, perspective: int) -> int:
    """Donor ``kKingBucketMap``: file-pair (0-3) plus 4 on the perspective's
    advanced half, indexed by the king square oriented to that perspective."""
    oriented = orient_square(king_square, perspective)
    file_pair = (oriented & 7) >> 1
    advanced_half = 4 if (oriented >> 3) >= 4 else 0
    return file_pair + advanced_half


def feature_row(colour: int, piece_type: int, square: int, king_square: int, perspective: int) -> int:
    """Donor ``FeatureRow``: the sparse-input row for one piece, in one
    perspective's king-bucket bank.

    ``colour``: 0=white, 1=black (absolute, not relative to ``perspective``).
    ``piece_type``: 0=pawn, 1=knight, 2=bishop, 3=rook, 4=queen, 5=king.
    ``square``: the piece's actual board square (a1=0 .. h8=63), unoriented.
    ``king_square``: ``perspective``'s own king's actual board square.
    """
    if colour not in (WHITE, BLACK):
        raise ValueError(f"colour out of range: {colour}")
    if not 0 <= piece_type < 6:
        raise ValueError(f"piece_type out of range: {piece_type}")
    if not 0 <= square < 64:
        raise ValueError(f"square out of range: {square}")
    bucket = king_bucket(king_square, perspective)
    if perspective == WHITE:
        slot_colour = colour
        oriented_square = square
    else:
        slot_colour = 1 - colour
        oriented_square = square ^ RANK_FLIP
    return bucket * INPUT_SIZE + slot_colour * 384 + piece_type * 64 + oriented_square


def _piece_colour_type(piece: chess.Piece) -> tuple[int, int]:
    colour = WHITE if piece.color == chess.WHITE else BLACK
    piece_type = piece.piece_type - 1  # chess.PAWN(1)..chess.KING(6) -> 0..5
    return colour, piece_type


def board_pieces(board: chess.Board) -> Iterable[tuple[int, int, int]]:
    """Yield ``(square, colour, piece_type)`` for every piece on the board, in
    stable square order."""
    for square in chess.scan_forward(board.occupied):
        piece = board.piece_at(square)
        if piece is None:  # pragma: no cover - guarded by the occupied bitboard
            raise AssertionError("occupied square has no piece")
        colour, piece_type = _piece_colour_type(piece)
        yield square, colour, piece_type


def king_squares(board: chess.Board) -> tuple[int, int]:
    """Return ``(white_king_square, black_king_square)``, validating exactly
    one king per side (a donor/engine invariant; python-chess does not
    enforce it on an arbitrary constructed board)."""
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    if (
        white_king is None
        or black_king is None
        or chess.popcount(board.kings & board.occupied_co[chess.WHITE]) != 1
        or chess.popcount(board.kings & board.occupied_co[chess.BLACK]) != 1
    ):
        raise ValueError("position must contain exactly one king per side")
    return white_king, black_king


def active_features(board: chess.Board, perspective: int) -> list[int]:
    """Return the sorted list of active feature-row indices for one
    perspective of ``board`` (one row per piece on the board, including both
    kings -- kings have their own feature rows same as any other piece; the
    king *bucket* is a separate, per-perspective selector)."""
    white_king, black_king = king_squares(board)
    king_square = white_king if perspective == WHITE else black_king
    return [
        feature_row(colour, piece_type, square, king_square, perspective)
        for square, colour, piece_type in board_pieces(board)
    ]


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode ``board`` as ``uint16[2, MAX_PIECES]`` (white row, black row),
    padded with ``PADDING_FEATURE``. Perspective order is fixed: index 0 is
    always White, index 1 is always Black (side-to-move ordering happens only
    at the evaluator's output-concatenation step, not here)."""
    white_king, black_king = king_squares(board)
    pieces = list(board_pieces(board))
    if len(pieces) > MAX_PIECES:
        raise ValueError(f"position has {len(pieces)} pieces; maximum is {MAX_PIECES}")
    encoded = np.full((NUM_PERSPECTIVES, MAX_PIECES), PADDING_FEATURE, dtype=np.uint16)
    for slot, (square, colour, piece_type) in enumerate(pieces):
        encoded[WHITE, slot] = feature_row(colour, piece_type, square, white_king, WHITE)
        encoded[BLACK, slot] = feature_row(colour, piece_type, square, black_king, BLACK)
    return encoded


def king_buckets_for(board: chess.Board) -> tuple[int, int]:
    """Return ``(white_bucket, black_bucket)`` for the current position."""
    white_king, black_king = king_squares(board)
    return king_bucket(white_king, WHITE), king_bucket(black_king, BLACK)
