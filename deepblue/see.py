"""Static Exchange Evaluation for Deep Blue.

This module deliberately has no search policy in it.  It answers a local
material question for a capture on one square, using the live bitboard
position but a virtual occupancy.  Search variants may use the result for move
ordering; pruning is a separate experiment.

The implementation recomputes attacks to the exchange square after every
virtual capture.  Candidate recaptures that expose their own king are skipped,
and a king capture is accepted only when the destination is not defended in
the resulting virtual position.  This costs a little more than an aggressively
specialised SEE, but is a good first Deep Blue version because correctness is
more valuable than shaving a handful of instructions from an ordering hint.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from deepblue.fastcore import (
    BLACK,
    BK,
    BP,
    BQ,
    BR,
    BB_,
    BN,
    FLAG_EP,
    KING_ATTACKS,
    KNIGHT_ATTACKS,
    NO_PIECE,
    PAWN_ATTACKS,
    WHITE,
    WK,
    WP,
    WQ,
    WR,
    WB,
    WN,
    bishop_attacks,
    lsb,
    rook_attacks,
)

ONE = np.uint64(1)

# Centipawn-like values used only for exchange arithmetic.  The absolute scale
# is irrelevant for ordering sign, but keeping conventional units makes the
# diagnostics easy to read.
SEE_VALUE = np.array(
    [100, 300, 300, 500, 900, 20000, 100, 300, 300, 500, 900, 20000],
    dtype=np.int32,
)


@njit(cache=True, nogil=True, inline="always")
def _active_piece_bits(bb, piece, occupied, target_bit):
    # The original piece that occupied the target has been captured.  The
    # current virtual occupant came from another source square, which has
    # already been removed from ``occupied``.  Masking target_bit therefore
    # removes the one stale original-board piece without needing to copy 12
    # bitboards for every SEE call.
    return bb[piece] & occupied & ~target_bit


@njit(cache=True, nogil=True)
def _virtual_attacked(bb, square, by_side, occupied, target_bit):
    """Attack test in a SEE virtual position.

    ``occupied`` models the current blockers.  Original piece bitboards are
    filtered by that occupancy and by ``target_bit`` so already-captured/moved
    pieces cannot become phantom attackers.
    """
    offset = 0 if by_side == WHITE else 6
    pawns = _active_piece_bits(bb, offset + WP, occupied, target_bit)
    knights = _active_piece_bits(bb, offset + WN, occupied, target_bit)
    bishops = _active_piece_bits(bb, offset + WB, occupied, target_bit)
    rooks = _active_piece_bits(bb, offset + WR, occupied, target_bit)
    queens = _active_piece_bits(bb, offset + WQ, occupied, target_bit)
    king = _active_piece_bits(bb, offset + WK, occupied, target_bit)

    if KNIGHT_ATTACKS[square] & knights:
        return True
    if KING_ATTACKS[square] & king:
        return True
    if PAWN_ATTACKS[1 - by_side][square] & pawns:
        return True
    if bishop_attacks(square, occupied) & (bishops | queens):
        return True
    if rook_attacks(square, occupied) & (rooks | queens):
        return True
    return False


@njit(cache=True, nogil=True)
def _attackers_of_piece(bb, side, piece_type, target, occupied, target_bit):
    """Virtual attackers of one piece type (0 pawn .. 5 king)."""
    offset = 0 if side == WHITE else 6
    piece = offset + piece_type
    bits = _active_piece_bits(bb, piece, occupied, target_bit)
    if piece_type == 0:
        return PAWN_ATTACKS[1 - side][target] & bits
    if piece_type == 1:
        return KNIGHT_ATTACKS[target] & bits
    if piece_type == 2:
        return bishop_attacks(target, occupied) & bits
    if piece_type == 3:
        return rook_attacks(target, occupied) & bits
    if piece_type == 4:
        return (bishop_attacks(target, occupied) | rook_attacks(target, occupied)) & bits
    return KING_ATTACKS[target] & bits


@njit(cache=True, nogil=True)
def _legal_lva(bb, side, target, occupied, target_bit):
    """Return (piece index, source square) of the least valuable legal attacker.

    SEE is an ordering primitive, but filtering self-pinned attackers avoids the
    most damaging false positives.  We test virtual king safety after vacating
    each candidate source.  At most a handful of captures reach this helper in
    a normal node.
    """
    king_piece = WK if side == WHITE else BK
    king_bits = _active_piece_bits(bb, king_piece, occupied, target_bit)
    king_square = lsb(king_bits) if king_bits else -1

    for piece_type in range(6):
        candidates = _attackers_of_piece(bb, side, piece_type, target, occupied, target_bit)
        while candidates:
            source = lsb(candidates)
            source_bit = ONE << np.uint64(source)
            candidates &= candidates - ONE
            after = occupied & ~source_bit

            if piece_type == 5:
                # The king lands on target.  Target remains occupied, but the
                # piece being captured there must not count as an attacker;
                # _virtual_attacked's target mask handles that.
                if not _virtual_attacked(bb, target, 1 - side, after, target_bit):
                    return side * 6 + piece_type, source
            else:
                if king_square < 0 or not _virtual_attacked(
                    bb, king_square, 1 - side, after, target_bit
                ):
                    return side * 6 + piece_type, source
    return -1, -1


@njit(cache=True, nogil=True)
def see_value(bb, occ, mail, st, move):
    """Return material SEE value for a non-promotion capture.

    Positive means the initiating side can keep a material gain if both sides
    may stop exchanging at any point.  Promotions are handled by a cheap
    immediate material term because Deep Blue always ranks promotions in the
    tactical-good stage; pruning must not use this function for promotions in
    this first version.
    """
    from_square = int(move & np.uint32(63))
    to_square = int((move >> np.uint32(6)) & np.uint32(63))
    piece = int((move >> np.uint32(12)) & np.uint32(15))
    captured = int((move >> np.uint32(16)) & np.uint32(15))
    promotion = int((move >> np.uint32(20)) & np.uint32(15))

    if captured == NO_PIECE:
        if promotion != NO_PIECE:
            # Quiet promotion: immediate material gain over the pawn.  Search
            # does not use SEE to demote promotions, but diagnostics get a
            # sensible value.
            return int(SEE_VALUE[promotion] - SEE_VALUE[piece])
        return 0

    gain = np.empty(32, dtype=np.int32)
    immediate = int(SEE_VALUE[captured])
    landed = piece
    if promotion != NO_PIECE:
        immediate += int(SEE_VALUE[promotion] - SEE_VALUE[piece])
        landed = promotion
    gain[0] = immediate
    depth = 0

    target_bit = ONE << np.uint64(to_square)
    occupied = occ[2] & ~(ONE << np.uint64(from_square))

    # En-passant removes a pawn from behind the destination and then places the
    # capturing pawn onto what was an empty target square.
    if move & FLAG_EP:
        side = int(st[0])
        capture_square = to_square - 8 if side == WHITE else to_square + 8
        occupied &= ~(ONE << np.uint64(capture_square))
        occupied |= target_bit

    side = 1 - int(st[0])
    landed_value = int(SEE_VALUE[landed])

    while depth < 30:
        attacker, source = _legal_lva(bb, side, to_square, occupied, target_bit)
        if attacker < 0:
            break

        depth += 1

        # A pawn recapturing onto its promotion rank must promote as part of
        # the exchange.  Account for the promotion material immediately and
        # treat the new virtual occupant as a queen for any following
        # recapture.  (If it is immediately recaptured, the promotion piece
        # value cancels with the promotion bonus; if the exchange stops here,
        # queen promotion is the material-maximising choice.)
        promotes = (attacker == WP and to_square >= 56) or (attacker == BP and to_square < 8)
        if promotes:
            gain[depth] = landed_value + int(SEE_VALUE[WQ] - SEE_VALUE[WP])
        else:
            gain[depth] = landed_value

        occupied &= ~(ONE << np.uint64(source))
        if promotes:
            landed_value = int(SEE_VALUE[WQ])
        else:
            landed_value = int(SEE_VALUE[attacker])

        # A legal king capture must land on an undefended target.  Therefore no
        # subsequent legal recapture exists in an ordinary exchange sequence.
        if attacker == WK or attacker == BK:
            break
        side = 1 - side

    result = int(gain[depth])
    index = depth - 1
    while index >= 0:
        if result > 0:
            result = int(gain[index]) - result
        else:
            result = int(gain[index])
        index -= 1
    return result


@njit(cache=True, nogil=True)
def see_ge_zero(bb, occ, mail, st, move):
    return see_value(bb, occ, mail, st, move) >= 0
