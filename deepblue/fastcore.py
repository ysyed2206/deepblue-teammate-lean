"""S1: our own board representation, built to stay inside Numba nopython mode.

No Python object crosses into the hot path. A position is a set of flat NumPy
arrays; a move is a packed uint32; move lists are preallocated arrays. Every
function that runs during a search is @njit.

Layout
------
``bb``   uint64[12]  piece bitboards, indexed by the PIECE constants below
``occ``  uint64[3]   [white, black, both], maintained incrementally
``st``   int64[5]    [side to move, castling mask, en-passant square,
                      halfmove clock, fullmove number]

Squares are 0..63 with a1 = 0 and h8 = 63, matching python-chess, so positions
can be compared against the oracle without translation.

Sliding attacks use the classical ray method: a precomputed ray per direction
per square, masked by occupancy, with the ray beyond the first blocker removed.
It is chosen because it is obviously correct on inspection. Faster schemes are
an experiment to run *after* the correctness gate passes, not before.
"""

from __future__ import annotations

import numpy as np
from numba import njit, uint64, int64, uint32, boolean, int32

# --------------------------------------------------------------------------
# Piece indexing
# --------------------------------------------------------------------------
WP, WN, WB, WR, WQ, WK = 0, 1, 2, 3, 4, 5
BP, BN, BB_, BR, BQ, BK = 6, 7, 8, 9, 10, 11
NO_PIECE = 15

WHITE, BLACK = 0, 1

# Castling rights bit mask.
CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ = 1, 2, 4, 8

NO_SQUARE = -1

# --------------------------------------------------------------------------
# Move encoding: one uint32.
#   bits  0-5   from square
#   bits  6-11  to square
#   bits 12-15  moving piece   (0..11)
#   bits 16-19  captured piece (0..11, or NO_PIECE)
#   bits 20-23  promotion piece(0..11, or NO_PIECE)
#   bit  24     en passant capture
#   bit  25     castling
#   bit  26     double pawn push
# --------------------------------------------------------------------------
FROM_SHIFT, TO_SHIFT, PIECE_SHIFT = 0, 6, 12
CAPTURE_SHIFT, PROMOTION_SHIFT = 16, 20
FLAG_EP = np.uint32(1 << 24)
FLAG_CASTLE = np.uint32(1 << 25)
FLAG_DOUBLE = np.uint32(1 << 26)

# The largest known legal move count in a chess position is 218, and
# tools/movebuffer_fuzz.py measures 218 pseudo-legal moves as the worst case
# over 30,000 probes. 256 left only 15% headroom on a buffer that lives inside
# nopython code, where an overflow is silent memory corruption rather than an
# exception. The extra slots cost a few hundred kilobytes.
MAX_MOVES = 320
MAX_PLY = 256

# --------------------------------------------------------------------------
# Precomputed tables, built once at import in plain Python integers and handed
# to the jitted code as arrays.
# --------------------------------------------------------------------------

# Direction indices. 0-3 are rook directions, 4-7 are bishop directions.
NORTH, EAST, SOUTH, WEST, NORTH_EAST, SOUTH_EAST, SOUTH_WEST, NORTH_WEST = range(8)
_DELTAS = ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1))
# A ray runs toward higher square indices for these directions, so the nearest
# blocker along them is the least significant set bit.
_POSITIVE = (NORTH, EAST, NORTH_EAST, NORTH_WEST)


def _build_rays() -> np.ndarray:
    rays = np.zeros((8, 64), dtype=np.uint64)
    for direction, (df, dr) in enumerate(_DELTAS):
        for square in range(64):
            file, rank = square & 7, square >> 3
            bits = 0
            f, r = file + df, rank + dr
            while 0 <= f < 8 and 0 <= r < 8:
                bits |= 1 << (r * 8 + f)
                f += df
                r += dr
            rays[direction][square] = np.uint64(bits)
    return rays


def _build_knight_attacks() -> np.ndarray:
    table = np.zeros(64, dtype=np.uint64)
    jumps = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
    for square in range(64):
        file, rank = square & 7, square >> 3
        bits = 0
        for df, dr in jumps:
            f, r = file + df, rank + dr
            if 0 <= f < 8 and 0 <= r < 8:
                bits |= 1 << (r * 8 + f)
        table[square] = np.uint64(bits)
    return table


def _build_king_attacks() -> np.ndarray:
    table = np.zeros(64, dtype=np.uint64)
    for square in range(64):
        file, rank = square & 7, square >> 3
        bits = 0
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if df == 0 and dr == 0:
                    continue
                f, r = file + df, rank + dr
                if 0 <= f < 8 and 0 <= r < 8:
                    bits |= 1 << (r * 8 + f)
        table[square] = np.uint64(bits)
    return table


def _build_pawn_attacks() -> np.ndarray:
    table = np.zeros((2, 64), dtype=np.uint64)
    for square in range(64):
        file, rank = square & 7, square >> 3
        white = 0
        black = 0
        for df in (-1, 1):
            f = file + df
            if 0 <= f < 8:
                if rank + 1 < 8:
                    white |= 1 << ((rank + 1) * 8 + f)
                if rank - 1 >= 0:
                    black |= 1 << ((rank - 1) * 8 + f)
        table[WHITE][square] = np.uint64(white)
        table[BLACK][square] = np.uint64(black)
    return table


RAYS = _build_rays()
KNIGHT_ATTACKS = _build_knight_attacks()
KING_ATTACKS = _build_king_attacks()
PAWN_ATTACKS = _build_pawn_attacks()
IS_POSITIVE_RAY = np.zeros(8, dtype=np.int64)
for _d in _POSITIVE:
    IS_POSITIVE_RAY[_d] = 1

_DEBRUIJN = np.uint64(0x03F79D71B4CB0A89)
_DEBRUIJN_INDEX = np.zeros(64, dtype=np.int64)
for _i in range(64):
    _DEBRUIJN_INDEX[int((( (1 << _i) * int(_DEBRUIJN)) & 0xFFFFFFFFFFFFFFFF) >> 58)] = _i

ONE = np.uint64(1)
ZERO = np.uint64(0)
FULL = np.uint64(0xFFFFFFFFFFFFFFFF)

# --------------------------------------------------------------------------
# Bit helpers
# --------------------------------------------------------------------------


@njit(int64(uint64), cache=True, nogil=True, inline="always")
def lsb(bits):
    """Index of the least significant set bit. Undefined for zero."""
    isolated = bits & (~bits + ONE)
    return _DEBRUIJN_INDEX[(isolated * _DEBRUIJN) >> np.uint64(58)]


@njit(int64(uint64), cache=True, nogil=True)
def msb(bits):
    """Index of the most significant set bit. Undefined for zero."""
    index = 0
    value = bits
    if value >> np.uint64(32):
        value >>= np.uint64(32)
        index += 32
    if value >> np.uint64(16):
        value >>= np.uint64(16)
        index += 16
    if value >> np.uint64(8):
        value >>= np.uint64(8)
        index += 8
    if value >> np.uint64(4):
        value >>= np.uint64(4)
        index += 4
    if value >> np.uint64(2):
        value >>= np.uint64(2)
        index += 2
    if value >> np.uint64(1):
        index += 1
    return index


@njit(int64(uint64), cache=True, nogil=True)
def popcount(bits):
    count = 0
    value = bits
    while value:
        value &= value - ONE
        count += 1
    return count


# --------------------------------------------------------------------------
# Attack generation
# --------------------------------------------------------------------------


@njit(uint64(int64, int64, uint64), cache=True, nogil=True)
def ray_attacks(direction, square, occupied):
    """Squares attacked along one direction, stopping on the first blocker.

    The blocker's own square is included, because it is either capturable or
    defended - both of which the caller needs to know about.
    """
    attacks = RAYS[direction][square]
    blockers = attacks & occupied
    if blockers:
        if IS_POSITIVE_RAY[direction]:
            nearest = lsb(blockers)
        else:
            nearest = msb(blockers)
        attacks ^= RAYS[direction][nearest]
    return attacks


@njit(uint64(int64, uint64), cache=True, nogil=True)
def rook_attacks(square, occupied):
    return (
        ray_attacks(NORTH, square, occupied)
        | ray_attacks(EAST, square, occupied)
        | ray_attacks(SOUTH, square, occupied)
        | ray_attacks(WEST, square, occupied)
    )


@njit(uint64(int64, uint64), cache=True, nogil=True)
def bishop_attacks(square, occupied):
    return (
        ray_attacks(NORTH_EAST, square, occupied)
        | ray_attacks(SOUTH_EAST, square, occupied)
        | ray_attacks(SOUTH_WEST, square, occupied)
        | ray_attacks(NORTH_WEST, square, occupied)
    )


@njit(uint64(int64, uint64), cache=True, nogil=True)
def queen_attacks(square, occupied):
    return rook_attacks(square, occupied) | bishop_attacks(square, occupied)


@njit(boolean(uint64[:], uint64[:], int64, int64), cache=True, nogil=True)
def is_attacked(bb, occ, square, by_side):
    """Is `square` attacked by any piece of `by_side`?

    Written as a set of reverse lookups from the target square: a knight
    attacks `square` exactly when `square`'s knight-attack set contains an
    enemy knight, and so on. That is one table read per piece type instead of
    a scan over every enemy piece.
    """
    offset = 0 if by_side == WHITE else 6
    occupied = occ[2]

    if KNIGHT_ATTACKS[square] & bb[offset + WN]:
        return True
    if KING_ATTACKS[square] & bb[offset + WK]:
        return True
    # A white pawn attacks `square` if `square` is in the black-pawn attack set
    # seen from `square` - the reverse-lookup trick, so the colour index flips.
    if PAWN_ATTACKS[1 - by_side][square] & bb[offset + WP]:
        return True
    queens = bb[offset + WQ]
    if rook_attacks(square, occupied) & (bb[offset + WR] | queens):
        return True
    if bishop_attacks(square, occupied) & (bb[offset + WB] | queens):
        return True
    return False


@njit(boolean(uint64[:], uint64[:], int64[:], int64), cache=True, nogil=True)
def in_check(bb, occ, st, side):
    king = bb[WK] if side == WHITE else bb[BK]
    if king == ZERO:
        return False
    return is_attacked(bb, occ, lsb(king), 1 - side)


# --------------------------------------------------------------------------
# Position state
# --------------------------------------------------------------------------
# A position is four arrays. `mail` is a 64-entry mailbox holding the piece
# index on each square, or NO_PIECE. It is redundant with `bb` but makes
# capture lookup a single read instead of a scan over twelve bitboards, and
# makes make/unmake straightforward to verify.
#
# `st` is [side to move, castling mask, en-passant square, halfmove clock,
# fullmove number].
#
# Undo records are three int64 per ply: castling mask, en-passant square and
# halfmove clock as they were before the move. The captured piece travels
# inside the move itself.

UNDO_STRIDE = 3

# Squares that, when vacated or captured on, remove a castling right.
A1, E1, H1 = 0, 4, 7
A8, E8, H8 = 56, 60, 63


@njit(int64(uint64[:], uint64[:], int64[:], int64[:], uint32[:]), cache=True, nogil=True)
def generate_pseudo_legal(bb, occ, mail, st, moves):
    """Fill `moves` with pseudo-legal moves; return how many.

    Pseudo-legal means every rule is applied except that the mover's king may
    be left in check. Legality is decided by making the move and testing, which
    is slower than pin masks and very much easier to get right. Correctness
    first; the faster scheme is an experiment for after the perft gate.
    """
    side = st[0]
    count = 0
    own = occ[side]
    enemy = occ[1 - side]
    occupied = occ[2]
    empty = ~occupied
    offset = 0 if side == WHITE else 6
    ep_square = st[2]

    # ---- pawns ----------------------------------------------------------
    pawns = bb[offset + WP]
    promotion_rank = 7 if side == WHITE else 0
    start_rank = 1 if side == WHITE else 6
    forward = 8 if side == WHITE else -8
    while pawns:
        from_square = lsb(pawns)
        pawns &= pawns - ONE
        base = (np.uint32(from_square) << np.uint32(FROM_SHIFT)) | (
            np.uint32(offset + WP) << np.uint32(PIECE_SHIFT)
        )

        to_square = from_square + forward
        if 0 <= to_square < 64 and (empty >> np.uint64(to_square)) & ONE:
            if (to_square >> 3) == promotion_rank:
                for promotion in (WQ, WR, WB, WN):
                    moves[count] = (
                        base
                        | (np.uint32(to_square) << np.uint32(TO_SHIFT))
                        | (np.uint32(NO_PIECE) << np.uint32(CAPTURE_SHIFT))
                        | (np.uint32(offset + promotion) << np.uint32(PROMOTION_SHIFT))
                    )
                    count += 1
            else:
                moves[count] = (
                    base
                    | (np.uint32(to_square) << np.uint32(TO_SHIFT))
                    | (np.uint32(NO_PIECE) << np.uint32(CAPTURE_SHIFT))
                    | (np.uint32(NO_PIECE) << np.uint32(PROMOTION_SHIFT))
                )
                count += 1
                double_square = from_square + 2 * forward
                if (from_square >> 3) == start_rank and (empty >> np.uint64(double_square)) & ONE:
                    moves[count] = (
                        base
                        | (np.uint32(double_square) << np.uint32(TO_SHIFT))
                        | (np.uint32(NO_PIECE) << np.uint32(CAPTURE_SHIFT))
                        | (np.uint32(NO_PIECE) << np.uint32(PROMOTION_SHIFT))
                        | FLAG_DOUBLE
                    )
                    count += 1

        targets = PAWN_ATTACKS[side][from_square] & enemy
        while targets:
            to_square = lsb(targets)
            targets &= targets - ONE
            captured = mail[to_square]
            if (to_square >> 3) == promotion_rank:
                for promotion in (WQ, WR, WB, WN):
                    moves[count] = (
                        base
                        | (np.uint32(to_square) << np.uint32(TO_SHIFT))
                        | (np.uint32(captured) << np.uint32(CAPTURE_SHIFT))
                        | (np.uint32(offset + promotion) << np.uint32(PROMOTION_SHIFT))
                    )
                    count += 1
            else:
                moves[count] = (
                    base
                    | (np.uint32(to_square) << np.uint32(TO_SHIFT))
                    | (np.uint32(captured) << np.uint32(CAPTURE_SHIFT))
                    | (np.uint32(NO_PIECE) << np.uint32(PROMOTION_SHIFT))
                )
                count += 1

        if ep_square >= 0 and (PAWN_ATTACKS[side][from_square] >> np.uint64(ep_square)) & ONE:
            captured_piece = (BP if side == WHITE else WP)
            moves[count] = (
                base
                | (np.uint32(ep_square) << np.uint32(TO_SHIFT))
                | (np.uint32(captured_piece) << np.uint32(CAPTURE_SHIFT))
                | (np.uint32(NO_PIECE) << np.uint32(PROMOTION_SHIFT))
                | FLAG_EP
            )
            count += 1

    # ---- knights, king, sliders -----------------------------------------
    for piece_kind in range(1, 6):
        pieces = bb[offset + piece_kind]
        while pieces:
            from_square = lsb(pieces)
            pieces &= pieces - ONE
            if piece_kind == WN:
                attacks = KNIGHT_ATTACKS[from_square]
            elif piece_kind == WB:
                attacks = bishop_attacks(from_square, occupied)
            elif piece_kind == WR:
                attacks = rook_attacks(from_square, occupied)
            elif piece_kind == WQ:
                attacks = queen_attacks(from_square, occupied)
            else:
                attacks = KING_ATTACKS[from_square]
            attacks &= ~own
            base = (np.uint32(from_square) << np.uint32(FROM_SHIFT)) | (
                np.uint32(offset + piece_kind) << np.uint32(PIECE_SHIFT)
            )
            while attacks:
                to_square = lsb(attacks)
                attacks &= attacks - ONE
                moves[count] = (
                    base
                    | (np.uint32(to_square) << np.uint32(TO_SHIFT))
                    | (np.uint32(mail[to_square]) << np.uint32(CAPTURE_SHIFT))
                    | (np.uint32(NO_PIECE) << np.uint32(PROMOTION_SHIFT))
                )
                count += 1

    # ---- castling --------------------------------------------------------
    castling = st[1]
    if side == WHITE:
        king_square = E1
        if (castling & CASTLE_WK) and mail[5] == NO_PIECE and mail[6] == NO_PIECE:
            if (
                not is_attacked(bb, occ, 4, BLACK)
                and not is_attacked(bb, occ, 5, BLACK)
                and not is_attacked(bb, occ, 6, BLACK)
            ):
                moves[count] = (
                    (np.uint32(4) << np.uint32(FROM_SHIFT))
                    | (np.uint32(6) << np.uint32(TO_SHIFT))
                    | (np.uint32(WK) << np.uint32(PIECE_SHIFT))
                    | (np.uint32(NO_PIECE) << np.uint32(CAPTURE_SHIFT))
                    | (np.uint32(NO_PIECE) << np.uint32(PROMOTION_SHIFT))
                    | FLAG_CASTLE
                )
                count += 1
        if (
            (castling & CASTLE_WQ)
            and mail[1] == NO_PIECE
            and mail[2] == NO_PIECE
            and mail[3] == NO_PIECE
        ):
            if (
                not is_attacked(bb, occ, 4, BLACK)
                and not is_attacked(bb, occ, 3, BLACK)
                and not is_attacked(bb, occ, 2, BLACK)
            ):
                moves[count] = (
                    (np.uint32(4) << np.uint32(FROM_SHIFT))
                    | (np.uint32(2) << np.uint32(TO_SHIFT))
                    | (np.uint32(WK) << np.uint32(PIECE_SHIFT))
                    | (np.uint32(NO_PIECE) << np.uint32(CAPTURE_SHIFT))
                    | (np.uint32(NO_PIECE) << np.uint32(PROMOTION_SHIFT))
                    | FLAG_CASTLE
                )
                count += 1
    else:
        if (castling & CASTLE_BK) and mail[61] == NO_PIECE and mail[62] == NO_PIECE:
            if (
                not is_attacked(bb, occ, 60, WHITE)
                and not is_attacked(bb, occ, 61, WHITE)
                and not is_attacked(bb, occ, 62, WHITE)
            ):
                moves[count] = (
                    (np.uint32(60) << np.uint32(FROM_SHIFT))
                    | (np.uint32(62) << np.uint32(TO_SHIFT))
                    | (np.uint32(BK) << np.uint32(PIECE_SHIFT))
                    | (np.uint32(NO_PIECE) << np.uint32(CAPTURE_SHIFT))
                    | (np.uint32(NO_PIECE) << np.uint32(PROMOTION_SHIFT))
                    | FLAG_CASTLE
                )
                count += 1
        if (
            (castling & CASTLE_BQ)
            and mail[57] == NO_PIECE
            and mail[58] == NO_PIECE
            and mail[59] == NO_PIECE
        ):
            if (
                not is_attacked(bb, occ, 60, WHITE)
                and not is_attacked(bb, occ, 59, WHITE)
                and not is_attacked(bb, occ, 58, WHITE)
            ):
                moves[count] = (
                    (np.uint32(60) << np.uint32(FROM_SHIFT))
                    | (np.uint32(58) << np.uint32(TO_SHIFT))
                    | (np.uint32(BK) << np.uint32(PIECE_SHIFT))
                    | (np.uint32(NO_PIECE) << np.uint32(CAPTURE_SHIFT))
                    | (np.uint32(NO_PIECE) << np.uint32(PROMOTION_SHIFT))
                    | FLAG_CASTLE
                )
                count += 1

    return count


@njit(int64(uint64[:], uint64[:], int64[:], int64[:], uint32[:]), cache=True, nogil=True)
def generate_pseudo_tactical(bb, occ, mail, st, moves):
    """Fill ``moves`` with pseudo-legal captures and promotions only.

    Quiescence does not need ordinary quiet moves.  Generating the complete
    pseudo-legal list and discarding the quiet majority was measured as one of
    search1's dominant costs.  This routine deliberately shares the same packed
    move format and legality contract as :func:`generate_pseudo_legal`: king
    safety is still checked by the caller after make_move().

    Included:
      * every capture by every piece,
      * en-passant captures,
      * capture promotions,
      * quiet promotions (promotion is tactical even without a capture).

    Excluded: ordinary pawn pushes, quiet piece moves and castling.
    """
    side = st[0]
    count = 0
    enemy = occ[1 - side]
    occupied = occ[2]
    offset = 0 if side == WHITE else 6
    ep_square = st[2]

    # ---- pawns: promotions, captures, en-passant ------------------------
    pawns = bb[offset + WP]
    promotion_rank = 7 if side == WHITE else 0
    forward = 8 if side == WHITE else -8
    while pawns:
        from_square = lsb(pawns)
        pawns &= pawns - ONE
        base = (np.uint32(from_square) << np.uint32(FROM_SHIFT)) | (
            np.uint32(offset + WP) << np.uint32(PIECE_SHIFT)
        )

        # Quiet promotions still belong in qsearch.
        to_square = from_square + forward
        if (
            0 <= to_square < 64
            and (to_square >> 3) == promotion_rank
            and ((occupied >> np.uint64(to_square)) & ONE) == ZERO
        ):
            for promotion in (WQ, WR, WB, WN):
                moves[count] = (
                    base
                    | (np.uint32(to_square) << np.uint32(TO_SHIFT))
                    | (np.uint32(NO_PIECE) << np.uint32(CAPTURE_SHIFT))
                    | (np.uint32(offset + promotion) << np.uint32(PROMOTION_SHIFT))
                )
                count += 1

        targets = PAWN_ATTACKS[side][from_square] & enemy
        while targets:
            to_square = lsb(targets)
            targets &= targets - ONE
            captured = mail[to_square]
            if (to_square >> 3) == promotion_rank:
                for promotion in (WQ, WR, WB, WN):
                    moves[count] = (
                        base
                        | (np.uint32(to_square) << np.uint32(TO_SHIFT))
                        | (np.uint32(captured) << np.uint32(CAPTURE_SHIFT))
                        | (np.uint32(offset + promotion) << np.uint32(PROMOTION_SHIFT))
                    )
                    count += 1
            else:
                moves[count] = (
                    base
                    | (np.uint32(to_square) << np.uint32(TO_SHIFT))
                    | (np.uint32(captured) << np.uint32(CAPTURE_SHIFT))
                    | (np.uint32(NO_PIECE) << np.uint32(PROMOTION_SHIFT))
                )
                count += 1

        if ep_square >= 0 and (PAWN_ATTACKS[side][from_square] >> np.uint64(ep_square)) & ONE:
            captured_piece = BP if side == WHITE else WP
            moves[count] = (
                base
                | (np.uint32(ep_square) << np.uint32(TO_SHIFT))
                | (np.uint32(captured_piece) << np.uint32(CAPTURE_SHIFT))
                | (np.uint32(NO_PIECE) << np.uint32(PROMOTION_SHIFT))
                | FLAG_EP
            )
            count += 1

    # ---- non-pawns: captures only ---------------------------------------
    for piece_kind in range(1, 6):
        pieces = bb[offset + piece_kind]
        while pieces:
            from_square = lsb(pieces)
            pieces &= pieces - ONE
            if piece_kind == WN:
                attacks = KNIGHT_ATTACKS[from_square]
            elif piece_kind == WB:
                attacks = bishop_attacks(from_square, occupied)
            elif piece_kind == WR:
                attacks = rook_attacks(from_square, occupied)
            elif piece_kind == WQ:
                attacks = queen_attacks(from_square, occupied)
            else:
                attacks = KING_ATTACKS[from_square]
            attacks &= enemy
            base = (np.uint32(from_square) << np.uint32(FROM_SHIFT)) | (
                np.uint32(offset + piece_kind) << np.uint32(PIECE_SHIFT)
            )
            while attacks:
                to_square = lsb(attacks)
                attacks &= attacks - ONE
                moves[count] = (
                    base
                    | (np.uint32(to_square) << np.uint32(TO_SHIFT))
                    | (np.uint32(mail[to_square]) << np.uint32(CAPTURE_SHIFT))
                    | (np.uint32(NO_PIECE) << np.uint32(PROMOTION_SHIFT))
                )
                count += 1

    return count


# --------------------------------------------------------------------------
# Make / unmake
# --------------------------------------------------------------------------
# Castling rights are cleared by a lookup: moving from or capturing on any of
# the six relevant squares removes exactly the rights that square carries.
CASTLE_MASK = np.full(64, 15, dtype=np.int64)
CASTLE_MASK[A1] = 15 & ~CASTLE_WQ
CASTLE_MASK[H1] = 15 & ~CASTLE_WK
CASTLE_MASK[E1] = 15 & ~(CASTLE_WK | CASTLE_WQ)
CASTLE_MASK[A8] = 15 & ~CASTLE_BQ
CASTLE_MASK[H8] = 15 & ~CASTLE_BK
CASTLE_MASK[E8] = 15 & ~(CASTLE_BK | CASTLE_BQ)


@njit(cache=True, nogil=True)
def make_move(bb, occ, mail, st, move, undo, ply):
    """Apply `move`. Every field needed to reverse it is saved into `undo`."""
    base = ply * UNDO_STRIDE
    undo[base + 0] = st[1]
    undo[base + 1] = st[2]
    undo[base + 2] = st[3]

    from_square = (move >> np.uint32(FROM_SHIFT)) & np.uint32(63)
    to_square = (move >> np.uint32(TO_SHIFT)) & np.uint32(63)
    piece = (move >> np.uint32(PIECE_SHIFT)) & np.uint32(15)
    captured = (move >> np.uint32(CAPTURE_SHIFT)) & np.uint32(15)
    promotion = (move >> np.uint32(PROMOTION_SHIFT)) & np.uint32(15)
    is_ep = (move & FLAG_EP) != np.uint32(0)
    is_castle = (move & FLAG_CASTLE) != np.uint32(0)
    is_double = (move & FLAG_DOUBLE) != np.uint32(0)

    side = st[0]
    them = 1 - side
    from_bit = ONE << np.uint64(from_square)
    to_bit = ONE << np.uint64(to_square)

    # -- remove the captured piece ----------------------------------------
    if captured != np.uint32(NO_PIECE):
        if is_ep:
            capture_square = to_square - np.uint32(8) if side == WHITE else to_square + np.uint32(8)
        else:
            capture_square = to_square
        capture_bit = ONE << np.uint64(capture_square)
        bb[captured] ^= capture_bit
        occ[them] ^= capture_bit
        occ[2] ^= capture_bit
        mail[capture_square] = NO_PIECE

    # -- move the piece ----------------------------------------------------
    bb[piece] ^= from_bit
    occ[side] ^= from_bit
    occ[2] ^= from_bit
    mail[from_square] = NO_PIECE

    landed = promotion if promotion != np.uint32(NO_PIECE) else piece
    bb[landed] |= to_bit
    occ[side] |= to_bit
    occ[2] |= to_bit
    mail[to_square] = landed

    # -- castling moves the rook too ---------------------------------------
    if is_castle:
        if to_square == np.uint32(6):
            rook_from, rook_to, rook = 7, 5, WR
        elif to_square == np.uint32(2):
            rook_from, rook_to, rook = 0, 3, WR
        elif to_square == np.uint32(62):
            rook_from, rook_to, rook = 63, 61, BR
        else:
            rook_from, rook_to, rook = 56, 59, BR
        rook_bits = (ONE << np.uint64(rook_from)) | (ONE << np.uint64(rook_to))
        bb[rook] ^= rook_bits
        occ[side] ^= rook_bits
        occ[2] ^= rook_bits
        mail[rook_from] = NO_PIECE
        mail[rook_to] = rook

    # -- rights, en passant, clocks ----------------------------------------
    st[1] = st[1] & CASTLE_MASK[from_square] & CASTLE_MASK[to_square]
    if is_double:
        st[2] = from_square + 8 if side == WHITE else from_square - 8
    else:
        st[2] = NO_SQUARE
    if piece == np.uint32(WP) or piece == np.uint32(BP) or captured != np.uint32(NO_PIECE):
        st[3] = 0
    else:
        st[3] = st[3] + 1
    if side == BLACK:
        st[4] = st[4] + 1
    st[0] = them


@njit(cache=True, nogil=True)
def unmake_move(bb, occ, mail, st, move, undo, ply):
    """Reverse `make_move`, restoring every field exactly."""
    base = ply * UNDO_STRIDE

    from_square = (move >> np.uint32(FROM_SHIFT)) & np.uint32(63)
    to_square = (move >> np.uint32(TO_SHIFT)) & np.uint32(63)
    piece = (move >> np.uint32(PIECE_SHIFT)) & np.uint32(15)
    captured = (move >> np.uint32(CAPTURE_SHIFT)) & np.uint32(15)
    promotion = (move >> np.uint32(PROMOTION_SHIFT)) & np.uint32(15)
    is_ep = (move & FLAG_EP) != np.uint32(0)
    is_castle = (move & FLAG_CASTLE) != np.uint32(0)

    side = 1 - st[0]
    them = st[0]
    from_bit = ONE << np.uint64(from_square)
    to_bit = ONE << np.uint64(to_square)

    if is_castle:
        if to_square == np.uint32(6):
            rook_from, rook_to, rook = 7, 5, WR
        elif to_square == np.uint32(2):
            rook_from, rook_to, rook = 0, 3, WR
        elif to_square == np.uint32(62):
            rook_from, rook_to, rook = 63, 61, BR
        else:
            rook_from, rook_to, rook = 56, 59, BR
        rook_bits = (ONE << np.uint64(rook_from)) | (ONE << np.uint64(rook_to))
        bb[rook] ^= rook_bits
        occ[side] ^= rook_bits
        occ[2] ^= rook_bits
        mail[rook_to] = NO_PIECE
        mail[rook_from] = rook

    landed = promotion if promotion != np.uint32(NO_PIECE) else piece
    bb[landed] ^= to_bit
    occ[side] ^= to_bit
    occ[2] ^= to_bit
    mail[to_square] = NO_PIECE

    bb[piece] |= from_bit
    occ[side] |= from_bit
    occ[2] |= from_bit
    mail[from_square] = piece

    if captured != np.uint32(NO_PIECE):
        if is_ep:
            capture_square = to_square - np.uint32(8) if side == WHITE else to_square + np.uint32(8)
        else:
            capture_square = to_square
        capture_bit = ONE << np.uint64(capture_square)
        bb[captured] |= capture_bit
        occ[them] |= capture_bit
        occ[2] |= capture_bit
        mail[capture_square] = captured

    st[1] = undo[base + 0]
    st[2] = undo[base + 1]
    st[3] = undo[base + 2]
    if side == BLACK:
        st[4] = st[4] - 1
    st[0] = side


@njit(cache=True, nogil=True)
def generate_legal(bb, occ, mail, st, stack, pseudo_stack, undo, ply):
    """Filter pseudo-legal moves by making each and testing the mover's king.

    Slower than pin and check-evasion masks, and far easier to be sure of.
    The correctness gate comes first; the faster scheme is EXP-006.

    Both move buffers are rows of caller-owned 2-D arrays indexed by ply. An
    earlier version allocated a fresh array inside this function and inside
    perft, which put two heap allocations on every node of the tree; removing
    them is EXP-007.
    """
    count = generate_pseudo_legal_row(bb, occ, mail, st, pseudo_stack, ply)
    legal = 0
    side = st[0]
    for index in range(count):
        move = pseudo_stack[ply, index]
        make_move(bb, occ, mail, st, move, undo, ply)
        if not in_check(bb, occ, st, side):
            stack[ply, legal] = move
            legal += 1
        unmake_move(bb, occ, mail, st, move, undo, ply)
    return legal


@njit(cache=True, nogil=True)
def generate_pseudo_legal_row(bb, occ, mail, st, pseudo_stack, ply):
    """generate_pseudo_legal writing into one row of a preallocated stack."""
    return generate_pseudo_legal(bb, occ, mail, st, pseudo_stack[ply])


@njit(cache=False, nogil=True)
def perft(bb, occ, mail, st, depth, stack, pseudo_stack, undo, ply):
    """Count leaf nodes. cache=False deliberately: a recursive njit function
    compiled with cache=True loads from a stale cache and dies with an LLVM
    unresolved-symbol error, which on the platform is an instant loss."""
    if depth == 0:
        return 1
    count = generate_legal(bb, occ, mail, st, stack, pseudo_stack, undo, ply)
    if depth == 1:
        return count
    total = 0
    for index in range(count):
        move = stack[ply, index]
        make_move(bb, occ, mail, st, move, undo, ply)
        total += perft(bb, occ, mail, st, depth - 1, stack, pseudo_stack, undo, ply + 1)
        unmake_move(bb, occ, mail, st, move, undo, ply)
    return total


def new_search_buffers():
    """Move stacks, pseudo-legal stacks and the undo record, allocated once."""
    return (
        np.zeros((MAX_PLY, MAX_MOVES), dtype=np.uint32),
        np.zeros((MAX_PLY, MAX_MOVES), dtype=np.uint32),
        np.zeros(MAX_PLY * UNDO_STRIDE, dtype=np.int64),
    )


# --------------------------------------------------------------------------
# Boundary: FEN in, UCI out. Plain Python - this runs once per position at the
# edge of the engine, never inside the search.
# --------------------------------------------------------------------------

FEN_PIECES = {
    "P": WP, "N": WN, "B": WB, "R": WR, "Q": WQ, "K": WK,
    "p": BP, "n": BN, "b": BB_, "r": BR, "q": BQ, "k": BK,
}
PIECE_LETTERS = "PNBRQKpnbrqk"
PROMOTION_LETTER = {WQ: "q", WR: "r", WB: "b", WN: "n", BQ: "q", BR: "r", BB_: "b", BN: "n"}


def new_position() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros(12, dtype=np.uint64),
        np.zeros(3, dtype=np.uint64),
        np.full(64, NO_PIECE, dtype=np.int64),
        np.zeros(5, dtype=np.int64),
    )


def from_fen(fen: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a FEN into the four position arrays."""
    bb, occ, mail, st = new_position()
    parts = fen.split()
    placement = parts[0]
    square = 56
    for character in placement:
        if character == "/":
            square -= 16
        elif character.isdigit():
            square += int(character)
        else:
            piece = FEN_PIECES[character]
            bb[piece] |= np.uint64(1) << np.uint64(square)
            mail[square] = piece
            square += 1

    for piece in range(12):
        if piece < 6:
            occ[WHITE] |= bb[piece]
        else:
            occ[BLACK] |= bb[piece]
    occ[2] = occ[WHITE] | occ[BLACK]

    st[0] = WHITE if parts[1] == "w" else BLACK
    rights = parts[2] if len(parts) > 2 else "-"
    mask = 0
    if "K" in rights:
        mask |= CASTLE_WK
    if "Q" in rights:
        mask |= CASTLE_WQ
    if "k" in rights:
        mask |= CASTLE_BK
    if "q" in rights:
        mask |= CASTLE_BQ
    st[1] = mask
    ep = parts[3] if len(parts) > 3 else "-"
    st[2] = NO_SQUARE if ep == "-" else (ord(ep[0]) - 97) + (int(ep[1]) - 1) * 8
    st[3] = int(parts[4]) if len(parts) > 4 else 0
    st[4] = int(parts[5]) if len(parts) > 5 else 1
    return bb, occ, mail, st


def decode(move: int) -> tuple[int, int, int, int, int, bool, bool, bool]:
    move = int(move)
    return (
        (move >> FROM_SHIFT) & 63,
        (move >> TO_SHIFT) & 63,
        (move >> PIECE_SHIFT) & 15,
        (move >> CAPTURE_SHIFT) & 15,
        (move >> PROMOTION_SHIFT) & 15,
        bool(move & int(FLAG_EP)),
        bool(move & int(FLAG_CASTLE)),
        bool(move & int(FLAG_DOUBLE)),
    )


def move_to_uci(move: int) -> str:
    from_square, to_square, _, _, promotion, _, _, _ = decode(move)
    text = _square_name(from_square) + _square_name(to_square)
    if promotion != NO_PIECE:
        text += PROMOTION_LETTER[promotion]
    return text


def _square_name(square: int) -> str:
    return chr(97 + (square & 7)) + str((square >> 3) + 1)


def legal_moves(bb, occ, mail, st) -> list[int]:
    """Boundary helper: the legal moves of a position as a Python list."""
    stack, pseudo_stack, undo = new_search_buffers()
    count = generate_legal(bb, occ, mail, st, stack, pseudo_stack, undo, 0)
    return [int(stack[0, i]) for i in range(count)]


def warm_up() -> float:
    """Compile every signature the search will use. Called at import so the
    cost lands inside the platform's 60 second initialisation budget rather
    than on the clock."""
    import time as _time

    started = _time.perf_counter()
    bb, occ, mail, st = from_fen(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    )
    stack, pseudo_stack, undo = new_search_buffers()
    generate_pseudo_legal(bb, occ, mail, st, stack[0])
    generate_pseudo_legal_row(bb, occ, mail, st, pseudo_stack, 0)
    generate_legal(bb, occ, mail, st, stack, pseudo_stack, undo, 0)
    perft(bb, occ, mail, st, 2, stack, pseudo_stack, undo, 0)
    is_attacked(bb, occ, 0, WHITE)
    in_check(bb, occ, st, WHITE)
    return _time.perf_counter() - started
