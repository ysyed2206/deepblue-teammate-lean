"""Zobrist hashing for the S1 core.

The hash is maintained *outside* fastcore, computed from the move's own bits
and the state before and after the move. That keeps the verified fast core
untouched - perft, the differential tests and the invariant suite all still
exercise exactly the code they were green against.

What is hashed: piece-square occupancy, side to move, castling rights, and the
en-passant file when an en-passant capture is actually available. Hashing the
en-passant square unconditionally would make two positions differ that a
player cannot distinguish, which pollutes both the transposition table and
repetition detection.
"""

from __future__ import annotations

import numpy as np
from numba import njit, uint64, int64

from deepblue.fastcore import (
    BK,
    BN,
    BP,
    BQ,
    BR,
    BB_,
    CAPTURE_SHIFT,
    FLAG_CASTLE,
    FLAG_EP,
    FROM_SHIFT,
    KING_ATTACKS,
    KNIGHT_ATTACKS,
    NO_PIECE,
    PAWN_ATTACKS,
    PIECE_SHIFT,
    PROMOTION_SHIFT,
    TO_SHIFT,
    WHITE,
    WK,
    WN,
    WP,
    WQ,
    WR,
    WB,
    bishop_attacks,
    lsb,
    rook_attacks,
)


def _tables() -> tuple[np.ndarray, np.uint64, np.ndarray, np.ndarray]:
    """Fixed random tables. The seed is pinned so a hash is reproducible
    across processes, which matters because a mismatch would otherwise look
    like a heisenbug."""
    rng = np.random.default_rng(0xC0FFEE)
    piece = rng.integers(0, 2**64, size=(12, 64), dtype=np.uint64)
    side = np.uint64(rng.integers(0, 2**64, dtype=np.uint64))
    castle = rng.integers(0, 2**64, size=16, dtype=np.uint64)
    ep_file = rng.integers(0, 2**64, size=8, dtype=np.uint64)
    return piece, side, castle, ep_file


PIECE_KEYS, SIDE_KEY, CASTLE_KEYS, EP_FILE_KEYS = _tables()
ZERO = np.uint64(0)
ONE = np.uint64(1)


@njit(int64(uint64[:], uint64[:], int64[:]), cache=True, nogil=True, inline="always")
def canonical_ep_file(bb, occ, st):
    """Return the EP file only when an en-passant capture is actually legal.

    Repetition identity follows the *available moves*, not raw FEN text.  A
    double pawn push may leave a nominal EP square even when every adjacent pawn
    is pinned and therefore cannot capture.  python-chess omits such an EP square
    from its transposition key; doing the same avoids missing genuine repeats.

    At most two candidate pawns exist.  We test king safety on the hypothetical
    EP occupancy directly instead of making/unmaking a move or generating a
    complete move list, so the common ``st[2] < 0`` path is essentially free.
    """
    ep = st[2]
    if ep < 0:
        return -1

    side = st[0]
    enemy = 1 - side
    pawn = WP if side == WHITE else BP
    enemy_pawn = BP if side == WHITE else WP
    candidates = PAWN_ATTACKS[1 - side][ep] & bb[pawn]
    if candidates == ZERO:
        return -1

    captured_square = ep - 8 if side == WHITE else ep + 8
    captured_bit = ONE << np.uint64(captured_square)
    if (bb[enemy_pawn] & captured_bit) == ZERO:
        return -1

    king_bits = bb[WK] if side == WHITE else bb[BK]
    if king_bits == ZERO:
        return -1
    king_square = lsb(king_bits)

    enemy_knights = bb[WN if enemy == WHITE else BN]
    enemy_bishops = bb[WB if enemy == WHITE else BB_]
    enemy_rooks = bb[WR if enemy == WHITE else BR]
    enemy_queens = bb[WQ if enemy == WHITE else BQ]
    enemy_king = bb[WK if enemy == WHITE else BK]
    enemy_pawns_after = bb[enemy_pawn] & ~captured_bit
    ep_bit = ONE << np.uint64(ep)

    while candidates:
        from_square = lsb(candidates)
        candidates &= candidates - ONE
        from_bit = ONE << np.uint64(from_square)
        occupied_after = (occ[2] & ~from_bit & ~captured_bit) | ep_bit

        attacked = False
        if KNIGHT_ATTACKS[king_square] & enemy_knights:
            attacked = True
        elif KING_ATTACKS[king_square] & enemy_king:
            attacked = True
        elif PAWN_ATTACKS[1 - enemy][king_square] & enemy_pawns_after:
            attacked = True
        elif rook_attacks(king_square, occupied_after) & (enemy_rooks | enemy_queens):
            attacked = True
        elif bishop_attacks(king_square, occupied_after) & (enemy_bishops | enemy_queens):
            attacked = True

        if not attacked:
            return ep & 7
    return -1


@njit(uint64(uint64[:], uint64[:], int64[:], uint64[:, :], uint64, uint64[:], uint64[:]),
      cache=True, nogil=True)
def full_hash(bb, occ, st, piece_keys, side_key, castle_keys, ep_file_keys):
    """Recompute the hash from scratch. The incremental hash must always
    equal this; that equality is a permanent invariant."""
    value = ZERO
    for piece in range(12):
        bits = bb[piece]
        while bits:
            square = lsb(bits)
            bits &= bits - ONE
            value ^= piece_keys[piece, square]
    if st[0] != WHITE:
        value ^= side_key
    value ^= castle_keys[st[1] & 15]
    ep_file = canonical_ep_file(bb, occ, st)
    if ep_file >= 0:
        value ^= ep_file_keys[ep_file]
    return value


@njit(uint64(uint64, uint64, int64, int64, int64, int64, int64,
             uint64[:, :], uint64, uint64[:], uint64[:]),
      cache=True, nogil=True)
def apply_move(value, move, side, old_castle, old_ep_file, new_castle, new_ep_file,
               piece_keys, side_key, castle_keys, ep_file_keys):
    """Update the hash for a move, from the move's bits and the state either
    side of it. Every term is its own XOR, so unmake is the identical call
    with the old and new state swapped - which is why the make/unmake hash
    invariant is exact rather than approximate."""
    from_square = (move >> np.uint64(FROM_SHIFT)) & np.uint64(63)
    to_square = (move >> np.uint64(TO_SHIFT)) & np.uint64(63)
    piece = (move >> np.uint64(PIECE_SHIFT)) & np.uint64(15)
    captured = (move >> np.uint64(CAPTURE_SHIFT)) & np.uint64(15)
    promotion = (move >> np.uint64(PROMOTION_SHIFT)) & np.uint64(15)

    value ^= piece_keys[piece, from_square]
    landed = promotion if promotion != np.uint64(NO_PIECE) else piece
    value ^= piece_keys[landed, to_square]

    if captured != np.uint64(NO_PIECE):
        if move & np.uint64(FLAG_EP):
            capture_square = to_square - np.uint64(8) if side == WHITE else to_square + np.uint64(8)
        else:
            capture_square = to_square
        value ^= piece_keys[captured, capture_square]

    if move & np.uint64(FLAG_CASTLE):
        if to_square == np.uint64(6):
            rook, rook_from, rook_to = 3, 7, 5
        elif to_square == np.uint64(2):
            rook, rook_from, rook_to = 3, 0, 3
        elif to_square == np.uint64(62):
            rook, rook_from, rook_to = 9, 63, 61
        else:
            rook, rook_from, rook_to = 9, 56, 59
        value ^= piece_keys[rook, rook_from]
        value ^= piece_keys[rook, rook_to]

    if old_castle != new_castle:
        value ^= castle_keys[old_castle & 15]
        value ^= castle_keys[new_castle & 15]
    if old_ep_file >= 0:
        value ^= ep_file_keys[old_ep_file]
    if new_ep_file >= 0:
        value ^= ep_file_keys[new_ep_file]
    value ^= side_key
    return value
