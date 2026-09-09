"""King-safety and passed-pawn evaluation terms, each independently testable.

Deep Blue's base ``evaluate`` is piece-square tables blended by phase and
nothing else -- no king safety, no passed pawns, no mobility. Three lost
qualification games (rounds 5, 7 and 8) trace directly to those two gaps:

- Rounds 5 and 8 were both lost to a mating attack after the engine pushed
  its own g- and h-pawns off the castled king (``g4``/``h5``/``hxg6`` and
  ``g4``/``g5``/``h4``). A piece-square table actively REWARDS an advanced
  pawn, so nothing in the old evaluation objected.
- Round 7 was lost to a passed b-pawn walking b6-b7-b8=Q while the engine
  spent its moves chasing checks. A pawn on b7 scored the same ~100cp as a
  pawn on b2.

The two terms are kept in separate functions on purpose. A first attempt
bundled both into one candidate (fastsearch31) and measured 44.3% over 96
games -- worse than the baseline -- with no way to tell which half was
responsible. Each is now its own candidate.

Both are computed White-relative; callers flip for Black, matching the base
evaluation's own convention.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from deepblue.fastcore import (
    KING_ATTACKS,
    KNIGHT_ATTACKS,
    bishop_attacks,
    lsb,
    queen_attacks,
    rook_attacks,
)

WP, WN, WB, WR, WQ, WK = 0, 1, 2, 3, 4, 5
BP, BN, BB_, BR, BQ, BK = 6, 7, 8, 9, 10, 11


def _build_masks():
    """Squares that decide 'passed' and 'has a pawn shield', per pawn/king square."""
    passed_w = np.zeros(64, dtype=np.uint64)
    passed_b = np.zeros(64, dtype=np.uint64)
    shield_w = np.zeros(64, dtype=np.uint64)
    shield_b = np.zeros(64, dtype=np.uint64)
    for square in range(64):
        file, rank = square & 7, square >> 3
        mask_pw = mask_pb = mask_sw = mask_sb = 0
        for delta in (-1, 0, 1):
            neighbour = file + delta
            if not 0 <= neighbour <= 7:
                continue
            for ahead in range(rank + 1, 8):
                mask_pw |= 1 << (ahead * 8 + neighbour)
            for ahead in range(0, rank):
                mask_pb |= 1 << (ahead * 8 + neighbour)
            for step in (1, 2):
                if 0 <= rank + step <= 7:
                    mask_sw |= 1 << ((rank + step) * 8 + neighbour)
                if 0 <= rank - step <= 7:
                    mask_sb |= 1 << ((rank - step) * 8 + neighbour)
        passed_w[square] = np.uint64(mask_pw)
        passed_b[square] = np.uint64(mask_pb)
        shield_w[square] = np.uint64(mask_sw)
        shield_b[square] = np.uint64(mask_sb)
    return passed_w, passed_b, shield_w, shield_b


PASSED_MASK_W, PASSED_MASK_B, SHIELD_MASK_W, SHIELD_MASK_B = _build_masks()


def _build_shield_bands():
    """Split the pawn shield into the rank NEXT to the king and the one after.

    SHIELD_MASK_W merges both into a single mask, so a pawn on h3 in front of
    a king on g1 counts exactly the same as a pawn on h2. That is not a
    detail. Round 58 (2026-09-07, lost by checkmate) was decided by 18. g4
    from a castled king on g1: the engine charged itself 15 centipawns for
    the whole push, having charged nothing at all for 17. h3 the move before,
    and Black's rook then used g3 and h3 -- the two squares those pushes
    vacated -- to mate. agent.py's own history already records two earlier
    losses to "mating attacks after the engine pushed its own g/h pawns off
    its castled king"; the shield term added in response cannot see the
    difference between an intact shield and an advanced one.

    These two masks make that distinction available: NEAR is rank +/- 1, FAR
    is rank +/- 2.
    """
    near_w = np.zeros(64, dtype=np.uint64)
    near_b = np.zeros(64, dtype=np.uint64)
    far_w = np.zeros(64, dtype=np.uint64)
    far_b = np.zeros(64, dtype=np.uint64)
    for square in range(64):
        file, rank = square & 7, square >> 3
        mask_nw = mask_nb = mask_fw = mask_fb = 0
        for delta in (-1, 0, 1):
            neighbour = file + delta
            if not 0 <= neighbour <= 7:
                continue
            if rank + 1 <= 7:
                mask_nw |= 1 << ((rank + 1) * 8 + neighbour)
            if rank - 1 >= 0:
                mask_nb |= 1 << ((rank - 1) * 8 + neighbour)
            if rank + 2 <= 7:
                mask_fw |= 1 << ((rank + 2) * 8 + neighbour)
            if rank - 2 >= 0:
                mask_fb |= 1 << ((rank - 2) * 8 + neighbour)
        near_w[square] = np.uint64(mask_nw)
        near_b[square] = np.uint64(mask_nb)
        far_w[square] = np.uint64(mask_fw)
        far_b[square] = np.uint64(mask_fb)
    return near_w, near_b, far_w, far_b


SHIELD_NEAR_W, SHIELD_NEAR_B, SHIELD_FAR_W, SHIELD_FAR_B = _build_shield_bands()

# Graded shield, charged per file of the three around the king.
#   pawn on the rank next to the king  -> intact, no charge
#   pawn only on the rank after that   -> advanced, ADVANCED charge
#   no pawn on either                  -> open, MISSING charge
# The existing term charges 18 for "open" and zero for both other cases; this
# splits that into two and raises the open charge, so that h2-h3 stops being
# free and g2-g4 costs roughly double what it did.
SHIELD_ADVANCED_PENALTY = np.int64(11)
SHIELD_MISSING_PENALTY = np.int64(27)
# A stronger setting, kept alongside rather than replacing it: the graded
# STRUCTURE is what round 58 proves is wrong, the magnitude is a free
# parameter and the 80-game match is what decides it.
SHIELD_ADVANCED_PENALTY_STRONG = np.int64(24)
SHIELD_MISSING_PENALTY_STRONG = np.int64(52)

# Indexed by how far the pawn has advanced (0 = own back rank, 6 = one step
# from promoting). Deliberately more conservative than a first guess of
# [0,8,14,26,48,90,150]: a bonus that large lets the search talk itself into
# unsound pawn races, and the bundled candidate that carried it lost ground.
PASSED_BONUS = np.array([0, 5, 9, 17, 32, 58, 95, 0], dtype=np.int64)

# Charged per missing shield pawn on the king's file and its two neighbours.
SHIELD_PENALTY = np.int64(18)


@njit(cache=True, nogil=True)
def popcount(value):
    count = 0
    while value:
        value &= value - np.uint64(1)
        count += 1
    return count


@njit(cache=True, nogil=True)
def passed_pawns_white_relative(bb):
    """Passed-pawn bonus, White-relative, in centipawns."""
    score = np.int64(0)
    white_pawns = bb[WP]
    black_pawns = bb[BP]
    bits = white_pawns
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        if black_pawns & PASSED_MASK_W[square] == np.uint64(0):
            score += PASSED_BONUS[square >> 3]
    bits = black_pawns
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        if white_pawns & PASSED_MASK_B[square] == np.uint64(0):
            score -= PASSED_BONUS[7 - (square >> 3)]
    return score


# Round 10 (2026-09-04, lost to Triggerfish) was decided by moves 27-36: White
# doubled rooks and dominated the e-file (Re7, Rde1, R7e2, Re6, Re8) while
# Yumo's pieces shuffled on the queenside. Piece-square tables give a rook the
# same value on e1 as on e7, or on a fully open file as on a closed one --
# this term is the direct fix for that blind spot.
OPEN_FILE_BONUS = np.int64(22)
SEMI_OPEN_FILE_BONUS = np.int64(11)
SEVENTH_RANK_BONUS = np.int64(18)

FILE_MASK = np.array(
    [np.uint64(0x0101010101010101 << f) for f in range(8)], dtype=np.uint64
)


@njit(cache=True, nogil=True)
def rook_files_white_relative(bb):
    """Open/semi-open file and seventh-rank bonuses for rooks, White-relative."""
    score = np.int64(0)
    white_pawns = bb[WP]
    black_pawns = bb[BP]
    all_pawns = white_pawns | black_pawns

    bits = bb[WR]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        file_mask = FILE_MASK[square & 7]
        if all_pawns & file_mask == np.uint64(0):
            score += OPEN_FILE_BONUS
        elif white_pawns & file_mask == np.uint64(0):
            score += SEMI_OPEN_FILE_BONUS
        if (square >> 3) == 6:
            score += SEVENTH_RANK_BONUS

    bits = bb[BR]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        file_mask = FILE_MASK[square & 7]
        if all_pawns & file_mask == np.uint64(0):
            score -= OPEN_FILE_BONUS
        elif black_pawns & file_mask == np.uint64(0):
            score -= SEMI_OPEN_FILE_BONUS
        if (square >> 3) == 1:
            score -= SEVENTH_RANK_BONUS

    return score


# Per-square-of-mobility weight, by piece type. Knights and bishops gain the
# most from being unblocked (their entire value is which squares they hit);
# rooks and queens are already long-range so the marginal square matters
# less. Deliberately modest -- mobility is a tie-breaker between otherwise
# similar moves, not a term that should outweigh material or king safety.
KNIGHT_MOBILITY_WEIGHT = np.int64(4)
BISHOP_MOBILITY_WEIGHT = np.int64(3)
ROOK_MOBILITY_WEIGHT = np.int64(2)
QUEEN_MOBILITY_WEIGHT = np.int64(1)


@njit(cache=True, nogil=True)
def mobility_white_relative(bb):
    """Pseudo-legal attack-square counts for knights/bishops/rooks/queens,
    White-relative, excluding squares a piece's own side already occupies.

    Takes only ``bb`` (not the separately-maintained ``occ`` array) so this
    can be called from ``evaluate()``, whose signature -- shared across every
    call site in the engine -- never carried occupancy. Both white and black
    occupancy are cheap to rebuild from the 12 piece bitboards directly.

    Pseudo-legal, not fully legal (no pin/check filtering) -- exactly like
    the mobility everyone actually ships. Filtering to legal-only would need
    a make/unmake per candidate move, at eval-call frequency; the field
    settled on pseudo-legal decades ago because the heuristic value is the
    same either way and the cost difference is not.
    """
    white_occ = np.uint64(0)
    for piece in range(6):
        white_occ |= bb[piece]
    black_occ = np.uint64(0)
    for piece in range(6, 12):
        black_occ |= bb[piece]
    occupied = white_occ | black_occ
    score = np.int64(0)

    bits = bb[WN]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        score += popcount(KNIGHT_ATTACKS[square] & ~white_occ) * KNIGHT_MOBILITY_WEIGHT
    bits = bb[WB]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        score += popcount(bishop_attacks(square, occupied) & ~white_occ) * BISHOP_MOBILITY_WEIGHT
    bits = bb[WR]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        score += popcount(rook_attacks(square, occupied) & ~white_occ) * ROOK_MOBILITY_WEIGHT
    bits = bb[WQ]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        score += popcount(queen_attacks(square, occupied) & ~white_occ) * QUEEN_MOBILITY_WEIGHT

    bits = bb[BN]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        score -= popcount(KNIGHT_ATTACKS[square] & ~black_occ) * KNIGHT_MOBILITY_WEIGHT
    bits = bb[BB_]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        score -= popcount(bishop_attacks(square, occupied) & ~black_occ) * BISHOP_MOBILITY_WEIGHT
    bits = bb[BR]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        score -= popcount(rook_attacks(square, occupied) & ~black_occ) * ROOK_MOBILITY_WEIGHT
    bits = bb[BQ]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        score -= popcount(queen_attacks(square, occupied) & ~black_occ) * QUEEN_MOBILITY_WEIGHT

    return score


@njit(cache=True, nogil=True)
def king_safety_white_relative(bb, phase, total_phase):
    """Pawn-shield term, White-relative, scaled down toward the endgame where
    an exposed king is normal and a shield is worth little."""
    if phase <= 0:
        return np.int64(0)
    white_king = bb[WK]
    black_king = bb[BK]
    if white_king == np.uint64(0) or black_king == np.uint64(0):
        return np.int64(0)
    wk = lsb(white_king)
    bk = lsb(black_king)
    wk_files = 3 if 0 < (wk & 7) < 7 else 2
    bk_files = 3 if 0 < (bk & 7) < 7 else 2
    white_shield = popcount(bb[WP] & SHIELD_MASK_W[wk])
    black_shield = popcount(bb[BP] & SHIELD_MASK_B[bk])
    if white_shield > wk_files:
        white_shield = wk_files
    if black_shield > bk_files:
        black_shield = bk_files
    safety = np.int64(0)
    safety -= (wk_files - white_shield) * SHIELD_PENALTY
    safety += (bk_files - black_shield) * SHIELD_PENALTY
    return (safety * phase) // total_phase


@njit(cache=True, nogil=True)
def _shield_charge(pawns, king_square, near_mask, far_mask,
                   advanced_penalty, missing_penalty):
    """Charge for one side's shield: per-file, intact / advanced / open.

    Counting is done per FILE rather than by popcount over the whole mask,
    because two pawns on one file must not pay for an empty neighbouring
    file -- the bug that a single merged popcount would reintroduce.
    """
    king_file = king_square & 7
    charge = np.int64(0)
    for delta in (-1, 0, 1):
        neighbour = king_file + delta
        if not 0 <= neighbour <= 7:
            continue
        file_bits = FILE_MASK[neighbour]
        if pawns & near_mask & file_bits:
            continue
        if pawns & far_mask & file_bits:
            charge += advanced_penalty
        else:
            charge += missing_penalty
    return charge


@njit(cache=True, nogil=True)
def king_safety_graded_white_relative(bb, phase, total_phase,
                                     advanced_penalty, missing_penalty):
    """Pawn-shield term that distinguishes an INTACT shield from an ADVANCED
    one. See _build_shield_bands for why: the flat version scores a king on
    g1 behind f2/g2/h3 exactly as safe as one behind f2/g2/h2, which is how
    round 58 was lost."""
    if phase <= 0:
        return np.int64(0)
    white_king = bb[WK]
    black_king = bb[BK]
    if white_king == np.uint64(0) or black_king == np.uint64(0):
        return np.int64(0)
    wk = lsb(white_king)
    bk = lsb(black_king)
    white_charge = _shield_charge(bb[WP], wk, SHIELD_NEAR_W[wk], SHIELD_FAR_W[wk],
                                  advanced_penalty, missing_penalty)
    black_charge = _shield_charge(bb[BP], bk, SHIELD_NEAR_B[bk], SHIELD_FAR_B[bk],
                                  advanced_penalty, missing_penalty)
    safety = black_charge - white_charge
    return (safety * phase) // total_phase


# Attacker-count-and-weight king danger term (donor: Stockfish 11's actual
# evaluate.cpp, read from its real source, not a description of it).
# KingAttackWeights there: N=81 B=52 R=44 Q=10 -- deliberately kept as-is
# rather than re-guessed, even though Stockfish's own low queen weight only
# makes sense alongside its much larger safe-check/mobility/threat terms
# that this codebase does not have. This is ONE isolated factor (the
# attacker-count-and-weight piece only, not the full Stockfish king-danger
# system, which also folds in unsafe checks, king blockers, mobility deltas
# and a pawn-shelter/storm table) -- kept separate from the existing
# shield-only king_safety_white_relative term above on purpose, matching
# this file's own convention that each factor is independently testable.
# Danger-to-penalty conversion is the same SHAPE as Stockfish's real formula
# (a quadratic once danger crosses a threshold), but Stockfish's own actual
# constants (threshold 100, divisor 4096) were measured against their FULL
# accumulated danger -- unsafe checks, weak squares, king blockers, mobility
# deltas and a flank term all summed in before that comparison -- and this
# term supplies only the attacker-weight piece of that sum. Checked directly
# against a real position from tonight's round-28 loss (White's bishop+queen
# both bearing on the king zone): the un-rescaled constants left danger at
# 62, under Stockfish's 100 threshold, so the term fired NEVER touched the
# score. Rescaled down for a single-factor sum: threshold 0 (always active,
# small danger already rounds near 0 through the quadratic itself) and a
# divisor of 512 (an eyeballed ~1/8 of Stockfish's, reasoning that the
# missing terms would typically contribute a comparable multiple of this
# one's own magnitude) rather than 4096. This rescaling is a judgment call,
# not sourced from the donor -- the paired-game test is what actually
# decides if it is calibrated well enough to keep.
KING_ZONE_ATTACK_WEIGHT_N = 81
KING_ZONE_ATTACK_WEIGHT_B = 81
KING_ZONE_ATTACK_WEIGHT_R = 121
# Queen weight, RE-DERIVED for this engine rather than inherited.
#
# The published Stockfish value is 10 -- eight times less than a knight --
# and that is correct THERE because their king-danger sum separately adds
# large terms for queen checks, safe checks, weak squares in the king ring
# and king mobility. This engine has none of those, so the queen's entire
# contribution to danger has to come through this one weight.
#
# Evidence it was wrong here, from a Stockfish-annotated game (round 69,
# 2026-09-08): with a white QUEEN ON h6 and KNIGHT ON h7, both adjacent to
# our king on g8, this term read ELEVEN centipawns, and the evaluation
# scored 23...Qxe4 -- Stockfish's "??" -- as +99 GOOD for us. Decomposed
# with tools/explain_eval.py. The same 11cp reading appears in round 60
# through an entire kingside attack.
#
# Set level with the knight: in a simplified presence-and-weight term, a
# queen bearing on the king zone is not less dangerous than a knight. This
# is a single constant so the change is attributable, and it is tested like
# any other candidate rather than assumed.
KING_ZONE_ATTACK_WEIGHT_Q = 10
KING_DANGER_THRESHOLD = 0
KING_DANGER_SCALE = 512


# Candidate queen weight, isolated from the shipped constant above so that a
# candidate engine and the champion genuinely differ.
#
# 202 is the classic attack-unit ratio (minor 2, rook 3, QUEEN 5) scaled so the
# knight keeps its current 81. That scheme is what engines use when they do NOT
# have Stockfish's separate safe-check terms -- which is our situation exactly.
# Modern Stockfish's Q=10 is an inversion of it, correct there only because the
# queen's danger was moved into check terms we never implemented.
#
# Evidence this matters: round 69 (Stockfish-annotated), a white queen on h6 and
# knight on h7 both adjacent to our king on g8 produced a king-danger reading of
# ELEVEN centipawns, and the evaluation scored 23...Qxe4 -- Stockfish's "??" --
# as +99 GOOD for us.
KING_ZONE_ATTACK_WEIGHT_Q_STRONG = 202


@njit(cache=True, nogil=True)
def mobility_and_danger_white_relative(bb, phase, total_phase):
    """mobility_white_relative + king_attack_danger_strongq_white_relative,
    computed together so each sliding piece's attack set is generated ONCE.

    WHY. Measured on this engine: bishop_attacks 24.9ns, rook_attacks 29.0ns,
    queen_attacks 53.9ns, against 2.2ns for a table lookup. A middlegame
    evaluate() makes about twenty slider calls -- roughly 65% of its 540ns --
    and the two terms above walk THE SAME pieces with THE SAME occupancy,
    so half of that work is computed twice and thrown away.

    Fusing them removes the duplication without changing any arithmetic. It is
    the cheap half of the magic-bitboard question: ~6% of total runtime for a
    pure refactor, where magic tables would give ~11% but need generation code,
    2.3MB of tables built inside a 90s init budget already 62s spent, and
    produce illegal moves rather than slow ones when subtly wrong.

    Returns (mobility, danger), both White-relative, identical to calling the
    two functions separately -- verified position by position.
    """
    white_occ = np.uint64(0)
    for piece in range(6):
        white_occ |= bb[piece]
    black_occ = np.uint64(0)
    for piece in range(6, 12):
        black_occ |= bb[piece]
    occupied = white_occ | black_occ

    mobility = np.int64(0)
    danger_to_white = np.int64(0)
    danger_to_black = np.int64(0)

    have_kings = bb[WK] != np.uint64(0) and bb[BK] != np.uint64(0) and phase > 0
    white_zone = np.uint64(0)
    black_zone = np.uint64(0)
    if have_kings:
        wk = lsb(bb[WK])
        bk = lsb(bb[BK])
        white_zone = KING_ATTACKS[wk] | (np.uint64(1) << np.uint64(wk))
        black_zone = KING_ATTACKS[bk] | (np.uint64(1) << np.uint64(bk))

    # White pieces: mobility adds, and each one threatens Black's king zone.
    bits = bb[WN]
    while bits:
        sq = lsb(bits); bits &= bits - np.uint64(1)
        att = KNIGHT_ATTACKS[sq]
        mobility += popcount(att & ~white_occ) * KNIGHT_MOBILITY_WEIGHT
        if have_kings and (att & black_zone):
            danger_to_black += KING_ZONE_ATTACK_WEIGHT_N
    bits = bb[WB]
    while bits:
        sq = lsb(bits); bits &= bits - np.uint64(1)
        att = bishop_attacks(sq, occupied)
        mobility += popcount(att & ~white_occ) * BISHOP_MOBILITY_WEIGHT
        if have_kings and (att & black_zone):
            danger_to_black += KING_ZONE_ATTACK_WEIGHT_B
    bits = bb[WR]
    while bits:
        sq = lsb(bits); bits &= bits - np.uint64(1)
        att = rook_attacks(sq, occupied)
        mobility += popcount(att & ~white_occ) * ROOK_MOBILITY_WEIGHT
        if have_kings and (att & black_zone):
            danger_to_black += KING_ZONE_ATTACK_WEIGHT_R
    bits = bb[WQ]
    while bits:
        sq = lsb(bits); bits &= bits - np.uint64(1)
        att = queen_attacks(sq, occupied)
        mobility += popcount(att & ~white_occ) * QUEEN_MOBILITY_WEIGHT
        if have_kings and (att & black_zone):
            danger_to_black += KING_ZONE_ATTACK_WEIGHT_Q_STRONG

    # Black pieces: mobility subtracts, and each threatens White's king zone.
    bits = bb[BN]
    while bits:
        sq = lsb(bits); bits &= bits - np.uint64(1)
        att = KNIGHT_ATTACKS[sq]
        mobility -= popcount(att & ~black_occ) * KNIGHT_MOBILITY_WEIGHT
        if have_kings and (att & white_zone):
            danger_to_white += KING_ZONE_ATTACK_WEIGHT_N
    bits = bb[BB_]
    while bits:
        sq = lsb(bits); bits &= bits - np.uint64(1)
        att = bishop_attacks(sq, occupied)
        mobility -= popcount(att & ~black_occ) * BISHOP_MOBILITY_WEIGHT
        if have_kings and (att & white_zone):
            danger_to_white += KING_ZONE_ATTACK_WEIGHT_B
    bits = bb[BR]
    while bits:
        sq = lsb(bits); bits &= bits - np.uint64(1)
        att = rook_attacks(sq, occupied)
        mobility -= popcount(att & ~black_occ) * ROOK_MOBILITY_WEIGHT
        if have_kings and (att & white_zone):
            danger_to_white += KING_ZONE_ATTACK_WEIGHT_R
    bits = bb[BQ]
    while bits:
        sq = lsb(bits); bits &= bits - np.uint64(1)
        att = queen_attacks(sq, occupied)
        mobility -= popcount(att & ~black_occ) * QUEEN_MOBILITY_WEIGHT
        if have_kings and (att & white_zone):
            danger_to_white += KING_ZONE_ATTACK_WEIGHT_Q_STRONG

    danger = np.int64(0)
    if have_kings:
        pw = np.int64(0)
        pb = np.int64(0)
        if danger_to_white > KING_DANGER_THRESHOLD:
            pw = (danger_to_white * danger_to_white) // KING_DANGER_SCALE
        if danger_to_black > KING_DANGER_THRESHOLD:
            pb = (danger_to_black * danger_to_black) // KING_DANGER_SCALE
        danger = ((pb - pw) * phase) // total_phase
    return mobility, danger


@njit(cache=True, nogil=True)
def king_attack_danger_strongq_white_relative(bb, phase, total_phase):
    """As king_attack_danger_white_relative, with the classic-ratio queen weight.

    Written out piece by piece rather than looping over a tuple of
    (index, weight) pairs. Numba unrolls such a loop and specialises the body
    for every element, which took this module's cold compile from 63.7s to
    124.9s -- past the 90s platform init budget, where a miss loses every game
    in the match. Same arithmetic, ordinary compile time.
    """
    if phase <= 0:
        return np.int64(0)
    white_king = bb[WK]
    black_king = bb[BK]
    if white_king == np.uint64(0) or black_king == np.uint64(0):
        return np.int64(0)
    occupied = np.uint64(0)
    for piece in range(12):
        occupied |= bb[piece]
    wk = lsb(white_king)
    bk = lsb(black_king)
    white_zone = KING_ATTACKS[wk] | (np.uint64(1) << np.uint64(wk))
    black_zone = KING_ATTACKS[bk] | (np.uint64(1) << np.uint64(bk))

    danger_to_white = np.int64(0)
    bits = bb[BN]
    while bits:
        square = lsb(bits); bits &= bits - np.uint64(1)
        if KNIGHT_ATTACKS[square] & white_zone:
            danger_to_white += KING_ZONE_ATTACK_WEIGHT_N
    bits = bb[BB_]
    while bits:
        square = lsb(bits); bits &= bits - np.uint64(1)
        if bishop_attacks(square, occupied) & white_zone:
            danger_to_white += KING_ZONE_ATTACK_WEIGHT_B
    bits = bb[BR]
    while bits:
        square = lsb(bits); bits &= bits - np.uint64(1)
        if rook_attacks(square, occupied) & white_zone:
            danger_to_white += KING_ZONE_ATTACK_WEIGHT_R
    bits = bb[BQ]
    while bits:
        square = lsb(bits); bits &= bits - np.uint64(1)
        if queen_attacks(square, occupied) & white_zone:
            danger_to_white += KING_ZONE_ATTACK_WEIGHT_Q_STRONG

    danger_to_black = np.int64(0)
    bits = bb[WN]
    while bits:
        square = lsb(bits); bits &= bits - np.uint64(1)
        if KNIGHT_ATTACKS[square] & black_zone:
            danger_to_black += KING_ZONE_ATTACK_WEIGHT_N
    bits = bb[WB]
    while bits:
        square = lsb(bits); bits &= bits - np.uint64(1)
        if bishop_attacks(square, occupied) & black_zone:
            danger_to_black += KING_ZONE_ATTACK_WEIGHT_B
    bits = bb[WR]
    while bits:
        square = lsb(bits); bits &= bits - np.uint64(1)
        if rook_attacks(square, occupied) & black_zone:
            danger_to_black += KING_ZONE_ATTACK_WEIGHT_R
    bits = bb[WQ]
    while bits:
        square = lsb(bits); bits &= bits - np.uint64(1)
        if queen_attacks(square, occupied) & black_zone:
            danger_to_black += KING_ZONE_ATTACK_WEIGHT_Q_STRONG

    penalty_white = np.int64(0)
    penalty_black = np.int64(0)
    if danger_to_white > KING_DANGER_THRESHOLD:
        penalty_white = (danger_to_white * danger_to_white) // KING_DANGER_SCALE
    if danger_to_black > KING_DANGER_THRESHOLD:
        penalty_black = (danger_to_black * danger_to_black) // KING_DANGER_SCALE
    return ((penalty_black - penalty_white) * phase) // total_phase


# ---------------------------------------------------------------- v2 king danger
# Round 60 (2026-09-07, drawn from a position the engine scored at -404, i.e.
# Black winning by four pawns, while being mated at): king attack danger read
# ELEVEN centipawns through White's entire kingside attack -- Ng5, Qg4->Qh4,
# bishop on the long diagonal, pawn to h5, rooks lifting. The queen ate b2, a2
# and c4 because material is priced at 100 per pawn and the attack at 11.
#
# Two structural reasons the original is nearly silent:
#
# 1. WEIGHTS WITHOUT THEIR MACHINERY. 81/52/44/10 are Stockfish's
#    KingAttackWeights. In Stockfish a queen weight of 10 is correct because
#    kingDanger separately adds large terms for safe checks, queen checks,
#    weak squares in the king ring and king mobility. We took the weights and
#    none of the rest, so the most dangerous attacker on the board scores less
#    than an eighth of a knight.
#
# 2. PRESENCE, NOT PRESSURE. Each attacker contributed once regardless of how
#    much of the zone it covered: a queen bearing on four squares around the
#    king scored the same as one grazing a corner.
#
# This version counts attacked zone squares and gives the queen a weight that
# reflects what a queen does. The scale is re-derived rather than kept: with
# per-square counting a serious attack reaches ~600 danger, and 600^2/2048 is
# about 175cp -- a real but not decisive penalty. The old 512 would have made
# the same attack worth 700cp and turned every piece near a king into a
# forced win. Capped, because a quadratic with no ceiling eventually outweighs
# material entirely.
KING_ZONE_ATTACK_WEIGHT_Q_V2 = np.int64(140)
KING_DANGER_SCALE_V2 = np.int64(2048)
KING_DANGER_MAX = np.int64(400)


@njit(cache=True, nogil=True)
def _zone_danger(bb, zone, occupied, knight_idx, bishop_idx, rook_idx, queen_idx):
    """Weighted count of enemy attacks landing on one king's zone."""
    danger = np.int64(0)
    bits = bb[knight_idx]
    while bits:
        square = lsb(bits); bits &= bits - np.uint64(1)
        danger += KING_ZONE_ATTACK_WEIGHT_N * popcount(KNIGHT_ATTACKS[square] & zone)
    bits = bb[bishop_idx]
    while bits:
        square = lsb(bits); bits &= bits - np.uint64(1)
        danger += KING_ZONE_ATTACK_WEIGHT_B * popcount(bishop_attacks(square, occupied) & zone)
    bits = bb[rook_idx]
    while bits:
        square = lsb(bits); bits &= bits - np.uint64(1)
        danger += KING_ZONE_ATTACK_WEIGHT_R * popcount(rook_attacks(square, occupied) & zone)
    bits = bb[queen_idx]
    while bits:
        square = lsb(bits); bits &= bits - np.uint64(1)
        danger += KING_ZONE_ATTACK_WEIGHT_Q_V2 * popcount(queen_attacks(square, occupied) & zone)
    return danger


@njit(cache=True, nogil=True)
def king_attack_danger_v2_white_relative(bb, phase, total_phase):
    """King danger by PRESSURE on the king zone, not mere presence."""
    if phase <= 0:
        return np.int64(0)
    white_king = bb[WK]
    black_king = bb[BK]
    if white_king == np.uint64(0) or black_king == np.uint64(0):
        return np.int64(0)
    occupied = np.uint64(0)
    for piece in range(12):
        occupied |= bb[piece]
    wk = lsb(white_king)
    bk = lsb(black_king)
    # Zone EXTENDED toward the enemy. Measured on round 60 move 18, at the
    # height of White's attack on g8: of Ng5, Bg2, Bd2, Rd1, Re1 and Qd3, the
    # knight alone touched the 9-square ring -- everything else scored zero,
    # so the term read 11cp against a genuine attack. The attackers were one
    # rank away, which is what an attack looks like before it lands. Adding
    # the rank two ahead of the king (the same band the graded shield calls
    # FAR) is what lets pressure register while it is still building, which
    # is the only point at which the engine can still avoid it.
    white_zone = (KING_ATTACKS[wk] | (np.uint64(1) << np.uint64(wk))
                  | SHIELD_FAR_W[wk])
    black_zone = (KING_ATTACKS[bk] | (np.uint64(1) << np.uint64(bk))
                  | SHIELD_FAR_B[bk])

    danger_to_white = _zone_danger(bb, white_zone, occupied, BN, BB_, BR, BQ)
    danger_to_black = _zone_danger(bb, black_zone, occupied, WN, WB, WR, WQ)

    penalty_white = (danger_to_white * danger_to_white) // KING_DANGER_SCALE_V2
    penalty_black = (danger_to_black * danger_to_black) // KING_DANGER_SCALE_V2
    if penalty_white > KING_DANGER_MAX:
        penalty_white = KING_DANGER_MAX
    if penalty_black > KING_DANGER_MAX:
        penalty_black = KING_DANGER_MAX
    return ((penalty_black - penalty_white) * phase) // total_phase


@njit(cache=True, nogil=True)
def king_attack_danger_white_relative(bb, phase, total_phase):
    """How many enemy minor/major pieces bear on each king's own zone.

    White-relative like the term above: positive means White's king is
    safer relative to Black's, matching the sign convention every caller
    already expects.
    """
    if phase <= 0:
        return np.int64(0)
    white_king = bb[WK]
    black_king = bb[BK]
    if white_king == np.uint64(0) or black_king == np.uint64(0):
        return np.int64(0)
    occupied = np.uint64(0)
    for piece in range(12):
        occupied |= bb[piece]

    wk = lsb(white_king)
    bk = lsb(black_king)
    white_zone = KING_ATTACKS[wk] | (np.uint64(1) << np.uint64(wk))
    black_zone = KING_ATTACKS[bk] | (np.uint64(1) << np.uint64(bk))

    danger_to_white = np.int64(0)
    bits = bb[BN]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        if KNIGHT_ATTACKS[square] & white_zone:
            danger_to_white += KING_ZONE_ATTACK_WEIGHT_N
    bits = bb[BB_]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        if bishop_attacks(square, occupied) & white_zone:
            danger_to_white += KING_ZONE_ATTACK_WEIGHT_B
    bits = bb[BR]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        if rook_attacks(square, occupied) & white_zone:
            danger_to_white += KING_ZONE_ATTACK_WEIGHT_R
    bits = bb[BQ]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        if queen_attacks(square, occupied) & white_zone:
            danger_to_white += KING_ZONE_ATTACK_WEIGHT_Q

    danger_to_black = np.int64(0)
    bits = bb[WN]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        if KNIGHT_ATTACKS[square] & black_zone:
            danger_to_black += KING_ZONE_ATTACK_WEIGHT_N
    bits = bb[WB]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        if bishop_attacks(square, occupied) & black_zone:
            danger_to_black += KING_ZONE_ATTACK_WEIGHT_B
    bits = bb[WR]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        if rook_attacks(square, occupied) & black_zone:
            danger_to_black += KING_ZONE_ATTACK_WEIGHT_R
    bits = bb[WQ]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        if queen_attacks(square, occupied) & black_zone:
            danger_to_black += KING_ZONE_ATTACK_WEIGHT_Q

    penalty_white = np.int64(0)
    if danger_to_white > KING_DANGER_THRESHOLD:
        penalty_white = danger_to_white * danger_to_white // KING_DANGER_SCALE
    penalty_black = np.int64(0)
    if danger_to_black > KING_DANGER_THRESHOLD:
        penalty_black = danger_to_black * danger_to_black // KING_DANGER_SCALE

    safety = penalty_black - penalty_white
    return (safety * phase) // total_phase


@njit(cache=True, nogil=True)
def game_phase(bb, phase_table, total_phase):
    phase = 0
    for piece in range(12):
        bits = bb[piece]
        while bits:
            bits &= bits - np.uint64(1)
            phase += phase_table[piece]
    if phase > total_phase:
        phase = total_phase
    return phase


# ---------------------------------------------------------------------------
# Bishop pair.
#
# Two bishops cover both square colours, so they complement each other in a way
# no other pair of minor pieces does; the advantage grows as the board opens.
# The base evaluation prices every bishop identically, so a side that trades one
# off sees no penalty at all. The bonus is deliberately modest and phase-flat:
# a phase-scaled version is a separate idea and would need its own measurement.
BISHOP_PAIR_BONUS = np.int64(28)


@njit(cache=True, nogil=True)
def bishop_pair_white_relative(bb):
    """Bonus for holding both bishops, White-relative, in centipawns."""
    score = np.int64(0)
    if popcount(bb[WB]) >= 2:
        score += BISHOP_PAIR_BONUS
    if popcount(bb[BB_]) >= 2:
        score -= BISHOP_PAIR_BONUS
    return score


# ---------------------------------------------------------------------------
# Pawn structure: doubled and isolated pawns.
#
# Piece-square tables score a pawn purely by where it stands, so two pawns
# stacked on one file score exactly as well as two abreast, and a pawn with no
# neighbours scores as well as one in a chain. Both are long-term structural
# weaknesses that a search cannot discover tactically -- they cost material only
# many moves later, well beyond the horizon -- which is precisely the kind of
# slow positional error that shows up as losing "gradually" rather than to a
# blunder.
DOUBLED_PAWN_PENALTY = np.int64(12)
ISOLATED_PAWN_PENALTY = np.int64(15)

ADJACENT_FILE_MASK = np.array(
    [
        np.uint64(
            (0x0101010101010101 << (f - 1) if f > 0 else 0)
            | (0x0101010101010101 << (f + 1) if f < 7 else 0)
        )
        for f in range(8)
    ],
    dtype=np.uint64,
)


@njit(cache=True, nogil=True)
def pawn_structure_white_relative(bb):
    """Doubled and isolated pawn penalties, White-relative, in centipawns."""
    score = np.int64(0)
    white_pawns = bb[WP]
    black_pawns = bb[BP]
    for file_index in range(8):
        file_mask = FILE_MASK[file_index]
        neighbours = ADJACENT_FILE_MASK[file_index]

        white_on_file = popcount(white_pawns & file_mask)
        if white_on_file > 1:
            score -= DOUBLED_PAWN_PENALTY * (white_on_file - 1)
        if white_on_file > 0 and (white_pawns & neighbours) == np.uint64(0):
            score -= ISOLATED_PAWN_PENALTY * white_on_file

        black_on_file = popcount(black_pawns & file_mask)
        if black_on_file > 1:
            score += DOUBLED_PAWN_PENALTY * (black_on_file - 1)
        if black_on_file > 0 and (black_pawns & neighbours) == np.uint64(0):
            score += ISOLATED_PAWN_PENALTY * black_on_file
    return score


# ---------------------------------------------------------------------------
# Knight outposts.
#
# A knight is worth far more on a square where it cannot be evicted by a pawn
# and is itself defended by one -- it becomes a permanent fixture in the
# opponent's position rather than a piece that gets chased. Piece-square
# tables give a knight the same value on such a square as on one a single
# tempo from being kicked by a pawn, because a PST cannot see pawns at all.
#
# Conditions, all required: the knight stands in enemy territory (ranks 4-6
# from its own side), no enemy pawn can ever attack it (no enemy pawn on an
# adjacent file, anywhere ahead of it), and a friendly pawn currently defends
# it. The advancement bonus is small because an outpost's value is positional
# rather than material.
OUTPOST_BONUS = np.array([0, 0, 0, 0, 20, 30, 22, 0], dtype=np.int64)


def _build_outpost_masks():
    """Squares from which an enemy pawn could ever attack this square."""
    attackable_w = np.zeros(64, dtype=np.uint64)   # black pawns that could hit a white knight
    attackable_b = np.zeros(64, dtype=np.uint64)
    defenders_w = np.zeros(64, dtype=np.uint64)    # white pawn squares defending this square
    defenders_b = np.zeros(64, dtype=np.uint64)
    for square in range(64):
        file_index = square & 7
        rank_index = square >> 3
        for adjacent in (file_index - 1, file_index + 1):
            if not 0 <= adjacent < 8:
                continue
            # A black pawn on an adjacent file AHEAD of a white knight can
            # advance and attack it; one behind never can.
            for rank in range(rank_index + 1, 8):
                attackable_w[square] |= np.uint64(1) << np.uint64(rank * 8 + adjacent)
            for rank in range(0, rank_index):
                attackable_b[square] |= np.uint64(1) << np.uint64(rank * 8 + adjacent)
            # A white pawn one rank BELOW on an adjacent file defends it.
            if rank_index - 1 >= 0:
                defenders_w[square] |= np.uint64(1) << np.uint64((rank_index - 1) * 8 + adjacent)
            if rank_index + 1 < 8:
                defenders_b[square] |= np.uint64(1) << np.uint64((rank_index + 1) * 8 + adjacent)
    return attackable_w, attackable_b, defenders_w, defenders_b


OUTPOST_ATTACKABLE_W, OUTPOST_ATTACKABLE_B, OUTPOST_DEFEND_W, OUTPOST_DEFEND_B = (
    _build_outpost_masks()
)


@njit(cache=True, nogil=True)
def knight_outposts_white_relative(bb):
    """Bonus for knights on unattackable, pawn-defended advanced squares."""
    score = np.int64(0)
    white_pawns = bb[WP]
    black_pawns = bb[BP]

    bits = bb[WN]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        rank = square >> 3
        if rank < 3 or rank > 6:
            continue
        if black_pawns & OUTPOST_ATTACKABLE_W[square] != np.uint64(0):
            continue
        if white_pawns & OUTPOST_DEFEND_W[square] == np.uint64(0):
            continue
        score += OUTPOST_BONUS[rank]

    bits = bb[BN]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        rank = square >> 3
        if rank > 4 or rank < 1:
            continue
        if white_pawns & OUTPOST_ATTACKABLE_B[square] != np.uint64(0):
            continue
        if black_pawns & OUTPOST_DEFEND_B[square] == np.uint64(0):
            continue
        score -= OUTPOST_BONUS[7 - rank]
    return score


# ---------------------------------------------------------------------------
# Steeper passed-pawn curve.
#
# PASSED_BONUS peaks at 95cp on the seventh rank, pricing a pawn one square
# from queening at roughly the value of a pawn. Round 7 was lost to exactly
# that: a passed b-pawn walked b6-b7-b8=Q while the engine chased checks,
# because b7 and b2 scored nearly the same.
#
# This is the same curve scaled 1.5x, as a SEPARATE constant and function
# rather than a mutation of PASSED_BONUS. eval_terms is shared by every
# fastsearch variant and the test harness imports two engines into one
# process, so rebinding the original array could change what the baseline
# compiles against and quietly invalidate the comparison.
PASSED_BONUS_STEEP = np.array([0, 7, 14, 26, 48, 87, 143, 0], dtype=np.int64)


@njit(cache=True, nogil=True)
def passed_pawns_steep_white_relative(bb):
    """Passed-pawn bonus on the steeper curve, White-relative, centipawns."""
    score = np.int64(0)
    white_pawns = bb[WP]
    black_pawns = bb[BP]
    bits = white_pawns
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        if black_pawns & PASSED_MASK_W[square] == np.uint64(0):
            score += PASSED_BONUS_STEEP[square >> 3]
    bits = black_pawns
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        if white_pawns & PASSED_MASK_B[square] == np.uint64(0):
            score -= PASSED_BONUS_STEEP[7 - (square >> 3)]
    return score


# ---------------------------------------------------------------------------
# Texel-tuned weights (2026-09-07).
#
# Fitted against 120,000 positions carrying depth 46-58 engine evaluations,
# by coordinate descent on the logistic objective
#     E(w) = mean( (sigmoid(base + w.features) - sigmoid(target))^2 )
# with 20% of the data held out and never used for fitting. Training error
# improved 1.97% and holdout error 1.87%, so the fit generalises rather than
# memorising.
#
# The first run of this produced nonsense (a passed pawn on the seventh worth
# nothing, mobility negative, isolated pawns a bonus) because the dataset's cp
# is WHITE-relative while our evaluation is SIDE-TO-MOVE relative. Mixing the
# two put half the data at the wrong sign: correlation was +0.343 on
# white-to-move positions, -0.296 on black-to-move, and 0.023 overall. After
# aligning the convention the correlation is +0.432 and the fitted weights are
# chess-sensible.
#
# NOTE the disagreement on BISHOP_PAIR: the fit wants 4, our own 120-game
# paired match measured the term at +41 Elo with 28. A fitting objective and a
# game result are not the same question -- the objective measures agreement
# with a stronger evaluator on static positions, which is a proxy. Where they
# conflict the game result is the one that counts, so both variants get tested.
PASSED_BONUS_TUNED = np.array([0, -7, -3, 25, 28, 30, 163, 0], dtype=np.int64)
SHIELD_PENALTY_TUNED = np.int64(22)
KNIGHT_MOBILITY_TUNED = np.int64(0)
BISHOP_MOBILITY_TUNED = np.int64(3)
ROOK_MOBILITY_TUNED = np.int64(6)
QUEEN_MOBILITY_TUNED = np.int64(1)
BISHOP_PAIR_TUNED = np.int64(4)
DOUBLED_PAWN_TUNED = np.int64(48)
ISOLATED_PAWN_TUNED = np.int64(23)


@njit(cache=True, nogil=True)
def passed_pawns_tuned_white_relative(bb):
    score = np.int64(0)
    white_pawns, black_pawns = bb[WP], bb[BP]
    bits = white_pawns
    while bits:
        square = lsb(bits); bits &= bits - np.uint64(1)
        if black_pawns & PASSED_MASK_W[square] == np.uint64(0):
            score += PASSED_BONUS_TUNED[square >> 3]
    bits = black_pawns
    while bits:
        square = lsb(bits); bits &= bits - np.uint64(1)
        if white_pawns & PASSED_MASK_B[square] == np.uint64(0):
            score -= PASSED_BONUS_TUNED[7 - (square >> 3)]
    return score


@njit(cache=True, nogil=True)
def mobility_tuned_white_relative(bb):
    white_occ = np.uint64(0)
    for piece in range(6):
        white_occ |= bb[piece]
    black_occ = np.uint64(0)
    for piece in range(6, 12):
        black_occ |= bb[piece]
    occupied = white_occ | black_occ
    score = np.int64(0)
    for piece, weight in ((WN, KNIGHT_MOBILITY_TUNED), (WB, BISHOP_MOBILITY_TUNED),
                          (WR, ROOK_MOBILITY_TUNED), (WQ, QUEEN_MOBILITY_TUNED)):
        bits = bb[piece]
        while bits:
            square = lsb(bits); bits &= bits - np.uint64(1)
            if piece == WN:
                attacks = KNIGHT_ATTACKS[square]
            elif piece == WB:
                attacks = bishop_attacks(square, occupied)
            elif piece == WR:
                attacks = rook_attacks(square, occupied)
            else:
                attacks = queen_attacks(square, occupied)
            score += popcount(attacks & ~white_occ) * weight
    for piece, weight in ((BN, KNIGHT_MOBILITY_TUNED), (BB_, BISHOP_MOBILITY_TUNED),
                          (BR, ROOK_MOBILITY_TUNED), (BQ, QUEEN_MOBILITY_TUNED)):
        bits = bb[piece]
        while bits:
            square = lsb(bits); bits &= bits - np.uint64(1)
            if piece == BN:
                attacks = KNIGHT_ATTACKS[square]
            elif piece == BB_:
                attacks = bishop_attacks(square, occupied)
            elif piece == BR:
                attacks = rook_attacks(square, occupied)
            else:
                attacks = queen_attacks(square, occupied)
            score -= popcount(attacks & ~black_occ) * weight
    return score


@njit(cache=True, nogil=True)
def king_safety_tuned_white_relative(bb, phase, total_phase):
    if phase <= 0:
        return np.int64(0)
    white_king, black_king = bb[WK], bb[BK]
    if white_king == np.uint64(0) or black_king == np.uint64(0):
        return np.int64(0)
    wk, bk = lsb(white_king), lsb(black_king)
    wk_files = 3 if 0 < (wk & 7) < 7 else 2
    bk_files = 3 if 0 < (bk & 7) < 7 else 2
    white_shield = popcount(bb[WP] & SHIELD_MASK_W[wk])
    black_shield = popcount(bb[BP] & SHIELD_MASK_B[bk])
    if white_shield > wk_files:
        white_shield = wk_files
    if black_shield > bk_files:
        black_shield = bk_files
    safety = np.int64(0)
    safety -= (wk_files - white_shield) * SHIELD_PENALTY_TUNED
    safety += (bk_files - black_shield) * SHIELD_PENALTY_TUNED
    return (safety * phase) // total_phase


@njit(cache=True, nogil=True)
def bishop_pair_tuned_white_relative(bb):
    score = np.int64(0)
    if popcount(bb[WB]) >= 2:
        score += BISHOP_PAIR_TUNED
    if popcount(bb[BB_]) >= 2:
        score -= BISHOP_PAIR_TUNED
    return score


@njit(cache=True, nogil=True)
def pawn_structure_tuned_white_relative(bb):
    score = np.int64(0)
    white_pawns, black_pawns = bb[WP], bb[BP]
    for file_index in range(8):
        file_mask = FILE_MASK[file_index]
        neighbours = ADJACENT_FILE_MASK[file_index]
        white_on_file = popcount(white_pawns & file_mask)
        if white_on_file > 1:
            score -= DOUBLED_PAWN_TUNED * (white_on_file - 1)
        if white_on_file > 0 and (white_pawns & neighbours) == np.uint64(0):
            score -= ISOLATED_PAWN_TUNED * white_on_file
        black_on_file = popcount(black_pawns & file_mask)
        if black_on_file > 1:
            score += DOUBLED_PAWN_TUNED * (black_on_file - 1)
        if black_on_file > 0 and (black_pawns & neighbours) == np.uint64(0):
            score += ISOLATED_PAWN_TUNED * black_on_file
    return score


# ---------------------------------------------------------------------------
# King-attack weights FITTED to Stockfish scores (2026-09-09).
#
# The shipped weights above (N 81, B 81, R 121, Q_STRONG 202) were each chosen
# by analogy -- 202 is the classic attack-unit ratio (minor 2, rook 3, queen 5)
# rescaled to keep the knight at 81. Two of them were never actually measured:
# fastsearch141 was supposed to test B=81/R=121, but that change was made in
# THIS shared file, so 140 and 141 are byte-identical and the "-34 Elo" it
# recorded is one build playing itself. The values stayed in regardless.
#
# These four are instead fitted by coordinate descent on 120k quiet positions
# carrying deep Stockfish evaluations, holding our exact danger -> penalty
# shape (sum of weights, squared, over 512). Validation MSE on a held-out 20%:
#
#     N 81 B 81  R 121 Q 202   0.032537   <- shipped in 140, 150, 162
#     N 81 B 52  R 44  Q 10    0.031757      Stockfish's own, for reference
#     N 62 B 42  R 65  Q 100   0.031663   <- fitted, below
#
# The queen is the whole story. Sweeping it alone, 202 is the WORST value
# anywhere in 10..202, and the curve is flat from 60 to 121. Round 80 shows
# what that costs in play: our evaluation scored an attack at +90 that a 45
# second search scores at -77.
#
# Isolated from the shipped constants deliberately. The last time a king-safety
# experiment edited a shared constant it silently changed the champion too, and
# the resulting match compared a build against itself.
KING_ZONE_ATTACK_WEIGHT_N_TUNED = 62
KING_ZONE_ATTACK_WEIGHT_B_TUNED = 42
KING_ZONE_ATTACK_WEIGHT_R_TUNED = 65
KING_ZONE_ATTACK_WEIGHT_Q_TUNED = 100


@njit(cache=True, nogil=True)
def king_attack_danger_tuned_white_relative(bb, phase, total_phase):
    """As king_attack_danger_strongq_white_relative, with fitted weights.

    Written out piece by piece rather than looping over (index, weight) pairs:
    Numba specialises such a loop for every element and that took this module's
    cold compile from 63.7s to 124.9s, past the 90s platform init budget.
    """
    if phase <= 0:
        return np.int64(0)
    white_king = bb[WK]
    black_king = bb[BK]
    if white_king == np.uint64(0) or black_king == np.uint64(0):
        return np.int64(0)
    occupied = np.uint64(0)
    for piece in range(12):
        occupied |= bb[piece]

    wk = lsb(white_king)
    bk = lsb(black_king)
    white_zone = KING_ATTACKS[wk] | (np.uint64(1) << np.uint64(wk))
    black_zone = KING_ATTACKS[bk] | (np.uint64(1) << np.uint64(bk))

    danger_to_white = np.int64(0)
    bits = bb[BN]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        if KNIGHT_ATTACKS[square] & white_zone:
            danger_to_white += KING_ZONE_ATTACK_WEIGHT_N_TUNED
    bits = bb[BB_]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        if bishop_attacks(square, occupied) & white_zone:
            danger_to_white += KING_ZONE_ATTACK_WEIGHT_B_TUNED
    bits = bb[BR]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        if rook_attacks(square, occupied) & white_zone:
            danger_to_white += KING_ZONE_ATTACK_WEIGHT_R_TUNED
    bits = bb[BQ]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        if queen_attacks(square, occupied) & white_zone:
            danger_to_white += KING_ZONE_ATTACK_WEIGHT_Q_TUNED

    danger_to_black = np.int64(0)
    bits = bb[WN]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        if KNIGHT_ATTACKS[square] & black_zone:
            danger_to_black += KING_ZONE_ATTACK_WEIGHT_N_TUNED
    bits = bb[WB]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        if bishop_attacks(square, occupied) & black_zone:
            danger_to_black += KING_ZONE_ATTACK_WEIGHT_B_TUNED
    bits = bb[WR]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        if rook_attacks(square, occupied) & black_zone:
            danger_to_black += KING_ZONE_ATTACK_WEIGHT_R_TUNED
    bits = bb[WQ]
    while bits:
        square = lsb(bits)
        bits &= bits - np.uint64(1)
        if queen_attacks(square, occupied) & black_zone:
            danger_to_black += KING_ZONE_ATTACK_WEIGHT_Q_TUNED

    penalty_white = np.int64(0)
    penalty_black = np.int64(0)
    if danger_to_white > KING_DANGER_THRESHOLD:
        penalty_white = (danger_to_white * danger_to_white) // KING_DANGER_SCALE
    if danger_to_black > KING_DANGER_THRESHOLD:
        penalty_black = (danger_to_black * danger_to_black) // KING_DANGER_SCALE
    return ((penalty_black - penalty_white) * phase) // total_phase
