"""Perspective Chess768 features with no king-dependent buckets.

Deep Blue numbers squares from ``a1=0`` and pieces as white P..K ``0..5``
then black P..K ``6..11``.  Both fixed accumulators share one 768-row table.
The black perspective flips ranks and swaps relative piece colours, so every
perspective sees its own pieces as colour zero advancing towards rank eight.
"""

from __future__ import annotations

from collections.abc import Iterator

import chess
import numpy as np

WHITE = 0
BLACK = 1
NUM_PERSPECTIVES = 2
PIECE_TYPES = 6
SQUARES = 64
COLOUR_STRIDE = PIECE_TYPES * SQUARES
NUM_FEATURES = 2 * COLOUR_STRIDE
PADDING_FEATURE = NUM_FEATURES
MAX_PIECES = 32


def orient_square(square: int, perspective: int) -> int:
    """Return ``square`` in a perspective whose home rank is rank one."""
    if not 0 <= square < SQUARES:
        raise ValueError(f"square out of range: {square}")
    if perspective not in (WHITE, BLACK):
        raise ValueError(f"perspective out of range: {perspective}")
    return square if perspective == WHITE else square ^ 56


def feature_index(piece: int, square: int, perspective: int) -> int:
    """Map one Deep Blue piece-square pair to a shared Chess768 row."""
    if not 0 <= piece < 12:
        raise ValueError(f"piece out of range: {piece}")
    relative_colour = (piece // PIECE_TYPES) ^ perspective
    piece_type = piece % PIECE_TYPES
    return (
        relative_colour * COLOUR_STRIDE
        + piece_type * SQUARES
        + orient_square(square, perspective)
    )


def _piece_codes(board: chess.Board) -> Iterator[tuple[int, int]]:
    for square in chess.scan_forward(board.occupied):
        piece = board.piece_at(square)
        if piece is None:  # pragma: no cover - occupied guarantees a piece
            raise AssertionError("occupied square has no piece")
        colour_offset = 0 if piece.color == chess.WHITE else PIECE_TYPES
        yield square, colour_offset + piece.piece_type - 1


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a python-chess board as padded ``uint16[2, 32]`` rows."""
    pieces = list(_piece_codes(board))
    if len(pieces) > MAX_PIECES:
        raise ValueError(f"position has {len(pieces)} pieces; maximum is {MAX_PIECES}")
    encoded = np.full(
        (NUM_PERSPECTIVES, MAX_PIECES), PADDING_FEATURE, dtype=np.uint16
    )
    for slot, (square, piece) in enumerate(pieces):
        encoded[WHITE, slot] = feature_index(piece, square, WHITE)
        encoded[BLACK, slot] = feature_index(piece, square, BLACK)
    return encoded


def encode_mailbox(mail: np.ndarray) -> np.ndarray:
    """Encode a Deep Blue mailbox as padded ``uint16[2, 32]`` rows."""
    if mail.shape != (SQUARES,):
        raise ValueError(f"mailbox must have shape (64,), got {mail.shape}")
    invalid = (mail != 15) & ((mail < 0) | (mail >= 12))
    if np.any(invalid):
        square = int(np.flatnonzero(invalid)[0])
        raise ValueError(f"invalid piece code {int(mail[square])} on square {square}")
    occupied = np.flatnonzero(mail < 12)
    if occupied.size > MAX_PIECES:
        raise ValueError(f"position has {occupied.size} pieces; maximum is {MAX_PIECES}")
    encoded = np.full(
        (NUM_PERSPECTIVES, MAX_PIECES), PADDING_FEATURE, dtype=np.uint16
    )
    for slot, square_value in enumerate(occupied):
        square = int(square_value)
        piece = int(mail[square])
        encoded[WHITE, slot] = feature_index(piece, square, WHITE)
        encoded[BLACK, slot] = feature_index(piece, square, BLACK)
    return encoded


def active_feature_rows(encoded: np.ndarray, perspective: int) -> np.ndarray:
    """Return sorted active rows for symmetry tests and diagnostics."""
    if encoded.shape != (NUM_PERSPECTIVES, MAX_PIECES):
        raise ValueError(f"encoded features must have shape (2, 32), got {encoded.shape}")
    if perspective not in (WHITE, BLACK):
        raise ValueError(f"perspective out of range: {perspective}")
    rows = encoded[perspective]
    return np.sort(rows[rows < NUM_FEATURES])


def mirror_colours_and_ranks(board: chess.Board) -> chess.Board:
    """Colour-swap and vertically mirror a board for invariance checks."""
    mirrored = chess.Board(None)
    for square, piece in board.piece_map().items():
        mirrored.set_piece_at(square ^ 56, chess.Piece(piece.piece_type, not piece.color))
    mirrored.turn = not board.turn
    rights = 0
    for square in chess.scan_forward(board.castling_rights):
        rights |= chess.BB_SQUARES[square ^ 56]
    mirrored.castling_rights = rights
    mirrored.ep_square = None if board.ep_square is None else board.ep_square ^ 56
    mirrored.halfmove_clock = board.halfmove_clock
    mirrored.fullmove_number = board.fullmove_number
    return mirrored
