"""Independent ChessBuckets feature encoder used by training and inference.

The two fixed accumulators are always WHITE-perspective then BLACK-perspective.
Only the scalar tail reorders them to side-to-move first.  Squares use the
Deep Blue/python-chess convention a1=0, h8=63.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import chess
import numpy as np

WHITE = 0
BLACK = 1
NUM_PERSPECTIVES = 2
NUM_BUCKETS = 8
PIECES_PER_BUCKET = 2 * 6 * 64
NUM_FEATURES = NUM_BUCKETS * PIECES_PER_BUCKET
MAX_PIECES = 32
PADDING_FEATURE = NUM_FEATURES


def orient_square(square: int, perspective: int) -> int:
    """Orient a square so the perspective's home rank is ranks 1-4."""
    if not 0 <= square < 64:
        raise ValueError(f"square out of range: {square}")
    if perspective not in (WHITE, BLACK):
        raise ValueError(f"perspective out of range: {perspective}")
    return square if perspective == WHITE else square ^ 56


def king_bucket(king_square: int, perspective: int) -> int:
    """Return file-pair x board-half bucket after perspective orientation."""
    oriented = orient_square(king_square, perspective)
    file_pair = (oriented & 7) >> 1
    far_half = 4 if (oriented >> 3) >= 4 else 0
    return file_pair + far_half


def feature_index(piece: int, square: int, king_square: int, perspective: int) -> int:
    """Map a Deep Blue piece code and square to one of 6144 sparse rows."""
    if not 0 <= piece < 12:
        raise ValueError(f"piece out of range: {piece}")
    absolute_colour = piece // 6
    piece_type = piece % 6
    relative_colour = absolute_colour ^ perspective
    oriented_square = orient_square(square, perspective)
    return (
        king_bucket(king_square, perspective) * PIECES_PER_BUCKET
        + relative_colour * 384
        + piece_type * 64
        + oriented_square
    )


def normalize_fen(fen: str) -> str:
    """Accept the dataset's four-field FEN and return a six-field FEN."""
    fields = fen.strip().split()
    if len(fields) == 4:
        fields.extend(("0", "1"))
    if len(fields) != 6:
        raise ValueError(f"expected four or six FEN fields, got {len(fields)}")
    return " ".join(fields)


def board_piece_codes(board: chess.Board) -> Iterable[tuple[int, int]]:
    """Yield ``(square, Deep Blue piece code)`` in stable square order."""
    for square in chess.scan_forward(board.occupied):
        piece = board.piece_at(square)
        if piece is None:  # pragma: no cover - guarded by occupied bitboard
            raise AssertionError("occupied square has no piece")
        colour_offset = 0 if piece.color == chess.WHITE else 6
        yield square, colour_offset + piece.piece_type - 1


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a board as uint16[2,32], padded with ``PADDING_FEATURE``."""
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    if (
        white_king is None
        or black_king is None
        or chess.popcount(board.kings & board.occupied_co[chess.WHITE]) != 1
        or chess.popcount(board.kings & board.occupied_co[chess.BLACK]) != 1
    ):
        raise ValueError("position must contain exactly one king per side")
    pieces = list(board_piece_codes(board))
    if len(pieces) > MAX_PIECES:
        raise ValueError(f"position has {len(pieces)} pieces; maximum is {MAX_PIECES}")
    encoded = np.full((NUM_PERSPECTIVES, MAX_PIECES), PADDING_FEATURE, dtype=np.uint16)
    for slot, (square, piece) in enumerate(pieces):
        encoded[WHITE, slot] = feature_index(piece, square, white_king, WHITE)
        encoded[BLACK, slot] = feature_index(piece, square, black_king, BLACK)
    return encoded


def encode_mailbox(mail: np.ndarray) -> np.ndarray:
    """Encode a Deep Blue mailbox without constructing a python-chess board."""
    if mail.shape != (64,):
        raise ValueError(f"mailbox must have shape (64,), got {mail.shape}")
    white_king_squares = np.flatnonzero(mail == 5)
    black_king_squares = np.flatnonzero(mail == 11)
    if white_king_squares.size != 1 or black_king_squares.size != 1:
        raise ValueError("mailbox must contain exactly one king per side")
    occupied = np.flatnonzero(mail < 12)
    if occupied.size > MAX_PIECES:
        raise ValueError(f"position has {occupied.size} pieces; maximum is {MAX_PIECES}")
    encoded = np.full((NUM_PERSPECTIVES, MAX_PIECES), PADDING_FEATURE, dtype=np.uint16)
    kings = (int(white_king_squares[0]), int(black_king_squares[0]))
    for slot, square_value in enumerate(occupied):
        square = int(square_value)
        piece = int(mail[square])
        for perspective in (WHITE, BLACK):
            encoded[perspective, slot] = feature_index(
                piece, square, kings[perspective], perspective
            )
    return encoded


def stable_hash64(text: str, seed: int = 0) -> int:
    """Stable keyed 64-bit hash used for leakage-safe split and sampling."""
    key = seed.to_bytes(8, "little", signed=False)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8, key=key).digest()
    return int.from_bytes(digest, "little", signed=False)


def split_for_fen(fen: str, seed: int = 20260901) -> str:
    """Assign 95% train, 2.5% validation, 2.5% test by FEN hash."""
    bucket = stable_hash64(fen, seed) % 2000
    if bucket < 50:
        return "validation"
    if bucket < 100:
        return "test"
    return "train"


def stm_target(white_relative_cp: int, board: chess.Board, clip_cp: int = 2000) -> int:
    """Clip a White-relative teacher score, then orient it to side to move."""
    clipped = max(-clip_cp, min(clip_cp, int(white_relative_cp)))
    return clipped if board.turn == chess.WHITE else -clipped


def mirrored_colour_board(board: chess.Board) -> chess.Board:
    """Vertical flip plus colour swap, used only by feature symmetry tests."""
    mirrored = chess.Board(None)
    for square, piece in board.piece_map().items():
        mirrored.set_piece_at(square ^ 56, chess.Piece(piece.piece_type, not piece.color))
    mirrored.turn = not board.turn
    rights = 0
    for rook_square in chess.scan_forward(board.castling_rights):
        rights |= chess.BB_SQUARES[rook_square ^ 56]
    mirrored.castling_rights = rights
    mirrored.ep_square = None if board.ep_square is None else board.ep_square ^ 56
    mirrored.halfmove_clock = board.halfmove_clock
    mirrored.fullmove_number = board.fullmove_number
    return mirrored
