"""Extract evaluation FEATURE COUNTS so the hand-picked weights can be tuned.

Every weight in deepblue/eval_terms.py was chosen by hand or copied, and none
has ever been fitted to data: SHIELD_PENALTY 18, BISHOP_PAIR_BONUS 28, the
mobility weights 4/3/2/1, the passed-pawn curve, DOUBLED 12, ISOLATED 15.
Published figures put Texel tuning at +2-10 Elo, but those are for engines
whose weights are ALREADY tuned; a first pass over guessed values is a
different proposition.

The reason this is tractable at all: every one of those terms is LINEAR in its
weight. mobility contributes KNIGHT_WEIGHT * (knight mobility count), the
bishop pair contributes BISHOP_PAIR_BONUS * (1 if the side has two), and so
on. So

    eval(position) = base_pst(position) + sum_i  w_i * f_i(position)

and if the f_i are extracted once, tuning w is an optimisation over ~15
numbers against precomputed features -- no Numba recompilation per candidate,
which is what would otherwise make this impossible (each weight change would
force a fresh JIT compile of the whole search).

Deliberately EXCLUDED: king_attack_danger. It is quadratic in the attack count
(danger*danger // KING_DANGER_SCALE), so it is not linear in its weights and
cannot join this fit. Tuning it needs a separate, non-linear pass.

Features are all White-relative and are then sign-flipped for the side to
move, matching evaluate()'s own convention.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepblue.eval_terms import (  # noqa: E402
    ADJACENT_FILE_MASK,
    FILE_MASK,
    PASSED_MASK_B,
    PASSED_MASK_W,
    SHIELD_MASK_B,
    SHIELD_MASK_W,
)
from deepblue.fastcore import (  # noqa: E402
    KNIGHT_ATTACKS,
    bishop_attacks,
    from_fen,
    lsb,
    queen_attacks,
    rook_attacks,
)

WP, WN, WB, WR, WQ, WK = 0, 1, 2, 3, 4, 5
BP, BN, BB_, BR, BQ, BK = 6, 7, 8, 9, 10, 11

# Order defines the weight vector layout used by the tuner.
FEATURE_NAMES = (
    [f"passed_rank{r}" for r in range(1, 7)]
    + ["shield_missing", "knight_mob", "bishop_mob", "rook_mob", "queen_mob",
       "bishop_pair", "doubled", "isolated"]
)
NUM_FEATURES = len(FEATURE_NAMES)


def _popcount(value: int) -> int:
    return int(value).bit_count()


def extract(bb, occ, phase=24.0, total_phase=24.0) -> np.ndarray:
    """White-relative feature counts for one position.

    Each feature is the exact coefficient its weight is multiplied by inside
    evaluate(), so that base_evaluate + weights.features reproduces the real
    evaluation (minus king_attack_danger, which is quadratic and excluded).
    Getting this exactly right matters more than it looks: tuning against
    features that do not match the engine optimises a different function, and
    the resulting weights would not transfer.
    """
    f = np.zeros(NUM_FEATURES, dtype=np.float64)
    white_pawns, black_pawns = int(bb[WP]), int(bb[BP])

    # passed pawns, per rank (index 0..5 == ranks 1..6 from the mover's side)
    bits = white_pawns
    while bits:
        square = lsb(np.uint64(bits)); bits &= bits - 1
        if black_pawns & int(PASSED_MASK_W[square]) == 0:
            rank = square >> 3
            if 1 <= rank <= 6:
                f[rank - 1] += 1
    bits = black_pawns
    while bits:
        square = lsb(np.uint64(bits)); bits &= bits - 1
        if white_pawns & int(PASSED_MASK_B[square]) == 0:
            rank = 7 - (square >> 3)
            if 1 <= rank <= 6:
                f[rank - 1] -= 1

    # King pawn shield. Mirrors king_safety_white_relative exactly: the
    # expected shield is 3 files (2 if the king stands on the a- or h-file),
    # the count is CAPPED at that, and the whole term is scaled by game phase
    # -- an exposed king is normal in an endgame. The coefficient of
    # SHIELD_PENALTY is therefore (black_missing - white_missing) * phase/total.
    white_king_bb, black_king_bb = int(bb[WK]), int(bb[BK])
    if white_king_bb and black_king_bb and phase > 0:
        wk = lsb(np.uint64(white_king_bb))
        bk = lsb(np.uint64(black_king_bb))
        wk_files = 3 if 0 < (wk & 7) < 7 else 2
        bk_files = 3 if 0 < (bk & 7) < 7 else 2
        white_shield = min(_popcount(white_pawns & int(SHIELD_MASK_W[wk])), wk_files)
        black_shield = min(_popcount(black_pawns & int(SHIELD_MASK_B[bk])), bk_files)
        missing = (bk_files - black_shield) - (wk_files - white_shield)
        f[6] = missing * (phase / total_phase)

    # Mobility, pseudo-legal attack counts. Occupancy is rebuilt from bb (the
    # separate occ array is not a scalar), and squares the side ALREADY
    # occupies are excluded -- mobility_white_relative masks with ~own_occ,
    # and omitting that would inflate every count by the piece's own
    # defended squares.
    white_occ = 0
    for piece in range(6):
        white_occ |= int(bb[piece])
    black_occ = 0
    for piece in range(6, 12):
        black_occ |= int(bb[piece])
    occupancy = np.uint64(white_occ | black_occ)

    for piece, index, attacker in (
        (WN, 7, None), (WB, 8, bishop_attacks), (WR, 9, rook_attacks), (WQ, 10, queen_attacks),
        (BN, 7, None), (BB_, 8, bishop_attacks), (BR, 9, rook_attacks), (BQ, 10, queen_attacks),
    ):
        white_side = piece <= WK
        sign = 1.0 if white_side else -1.0
        own = white_occ if white_side else black_occ
        bits = int(bb[piece])
        while bits:
            square = lsb(np.uint64(bits)); bits &= bits - 1
            if attacker is None:
                attacks = int(KNIGHT_ATTACKS[square])
            else:
                attacks = int(attacker(np.uint64(square), occupancy))
            f[index] += sign * _popcount(attacks & ~own)

    # bishop pair
    f[11] = (1.0 if _popcount(int(bb[WB])) >= 2 else 0.0) - (
        1.0 if _popcount(int(bb[BB_])) >= 2 else 0.0)

    # Doubled and isolated pawns. The feature is the RAW white-relative count;
    # the penalty sign lives in the weight. Carrying the sign in both places
    # cancels it and turns the penalty into a reward.
    for pawns, sign in ((white_pawns, 1.0), (black_pawns, -1.0)):
        for file_index in range(8):
            file_mask = int(FILE_MASK[file_index])
            count = _popcount(pawns & file_mask)
            if count > 1:
                f[12] += sign * (count - 1)
            if count > 0 and pawns & int(ADJACENT_FILE_MASK[file_index]) == 0:
                f[13] += sign * count
    return f


def load(paths, limit, clip_cp=1500):
    """Return (features, base_eval, target_cp) arrays, side-to-move relative."""
    from deepblue.eval_terms import game_phase
    from deepblue.fastsearch109 import EG_TABLE, MG_TABLE, PHASE_TABLE, base_evaluate

    feats, bases, targets = [], [], []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if len(feats) >= limit:
                    break
                row = json.loads(line)
                if row.get("mate") is not None or row.get("cp") is None:
                    continue
                cp = int(row["cp"])
                if abs(cp) > clip_cp:
                    continue
                try:
                    bb, occ, mail, st = from_fen(row["fen"] + " 0 1")
                except Exception:  # noqa: BLE001 - a bad FEN is just skipped
                    continue
                phase = float(game_phase(bb, PHASE_TABLE, 24))
                vector = extract(bb, occ, phase, 24.0)
                # CONVENTION. The dataset's cp is WHITE-relative; base_evaluate
                # and evaluate() are SIDE-TO-MOVE relative. Measured directly:
                # correlation of base_evaluate against cp is +0.343 on
                # white-to-move positions and -0.296 on black-to-move ones, so
                # mixing them cancels to 0.023 and any fit is against noise.
                # Everything here is put in the side-to-move frame: features are
                # flipped for black, and so is the target.
                if st[0] != 0:
                    vector = -vector
                    cp = -cp
                feats.append(vector)
                bases.append(float(base_evaluate(bb, st, MG_TABLE, EG_TABLE, PHASE_TABLE)))
                targets.append(float(cp))
        if len(feats) >= limit:
            break
    return np.array(feats), np.array(bases), np.array(targets)
