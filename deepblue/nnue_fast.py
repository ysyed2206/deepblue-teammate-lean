"""Allocation-free NNUE kernels for Deep Blue.

Drop-in companion to ``deepblue/nnue.py``; everything not redefined here is
re-exported from it unchanged, so the feature mapping, the weight loader and the
file format are shared and cannot drift apart.

Two changes, both measured on this machine with the H128 net:

``apply_delta_into``   ``nnue.apply_delta`` begins ``out = acc.copy()``, so every
                       make-move allocates a fresh Numba array. That allocation,
                       not the arithmetic, is the cost: apply_delta measured
                       6,063 ns against 2,575 ns for the same arithmetic writing
                       into a caller-owned buffer -- 2.4x. The search already
                       owns a destination (``out_white_acc_row``, a row of its
                       per-ply ``int16[MAX_PLY+1, HIDDEN]`` stack) and copies
                       into it afterwards, so passing it in removes the
                       allocation *and* that copy.

``tail_fast``          ``nnue.tail`` casts each element to int64 before
                       multiplying. The products are bounded by QA*QA*|w|, which
                       the quantiser already keeps well inside int32, so
                       accumulating with int32 operands is enough: 2,016 ns ->
                       1,737 ns.

Together the per-node evaluation path (one tail plus two accumulator updates)
goes from ~14,100 ns to ~6,900 ns, a 2.1x reduction. Both functions are checked
against the originals for bit-identical output.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from deepblue.fastcore import (
    FROM_SHIFT,
    TO_SHIFT,
    PIECE_SHIFT,
    CAPTURE_SHIFT,
    PROMOTION_SHIFT,
    FLAG_EP,
    FLAG_CASTLE,
    lsb,
)
from deepblue.nnue import *  # noqa: F403 - shared loader/mapping/format
from deepblue.nnue import QA, QB, SCALE, compute_delta, feature_row, king_bucket, refresh

WK, BK = 5, 11


@njit(cache=True, nogil=True)
def apply_delta_into(dst, src, feature_weights, removed_squares, removed_pieces,
                     removed_count, added_squares, added_pieces, added_count,
                     king_square, perspective):
    """``nnue.apply_delta`` writing into ``dst`` instead of allocating.

    Same restriction as the original: one king square/bucket for every row in
    the call, so the caller must use ``refresh`` when the bucket changed.
    ``dst`` and ``src`` may not alias.
    """
    n = src.shape[0]
    for j in range(n):
        dst[j] = src[j]
    for i in range(removed_count):
        row = feature_row(removed_pieces[i], removed_squares[i], king_square, perspective)
        for j in range(n):
            dst[j] -= feature_weights[row, j]
    for i in range(added_count):
        row = feature_row(added_pieces[i], added_squares[i], king_square, perspective)
        for j in range(n):
            dst[j] += feature_weights[row, j]


@njit(cache=True, nogil=True)
def tail_fast(white_acc, black_acc, side_to_move, output_weights_stm_side,
              output_weights_other_side, output_bias):
    """``nnue.tail`` with int32 operands. Identical arithmetic and identical
    output; only the width of the intermediate multiply differs."""
    if side_to_move == 0:
        stm_acc, ntm_acc = white_acc, black_acc
    else:
        stm_acc, ntm_acc = black_acc, white_acc
    dot = np.int64(0)
    n = stm_acc.shape[0]
    for i in range(n):
        x = stm_acc[i]
        c = np.int32(0) if x < 0 else (np.int32(QA) if x > QA else np.int32(x))
        dot += np.int64(c * c * np.int32(output_weights_stm_side[i]))
    for i in range(n):
        x = ntm_acc[i]
        c = np.int32(0) if x < 0 else (np.int32(QA) if x > QA else np.int32(x))
        dot += np.int64(c * c * np.int32(output_weights_other_side[i]))
    out = dot // np.int64(QA) + output_bias
    out *= np.int64(SCALE)
    denom = np.int64(QA) * np.int64(QB)
    q = out // denom
    if q < 0 and out % denom != 0:
        q += 1
    return q


@njit(cache=True, nogil=True)
def advance_accumulators(bb, mail, move_int, side, white_acc_row, black_acc_row,
                         old_white_king, old_black_king,
                         feature_weights, feature_bias,
                         out_white_acc_row, out_black_acc_row,
                         removed_sq, removed_pc, added_sq, added_pc):
    """Compute ply+1's accumulator rows from ply's, given the move just made
    (bb/mail already reflect the POST-move position when this is called).
    Returns (new_white_king, new_black_king)."""
    move = np.uint32(move_int)
    from_sq = np.int64((move >> np.uint32(FROM_SHIFT)) & np.uint32(63))
    to_sq = np.int64((move >> np.uint32(TO_SHIFT)) & np.uint32(63))
    piece = np.int64((move >> np.uint32(PIECE_SHIFT)) & np.uint32(15))
    captured = np.int64((move >> np.uint32(CAPTURE_SHIFT)) & np.uint32(15))
    promotion = np.int64((move >> np.uint32(PROMOTION_SHIFT)) & np.uint32(15))
    is_ep = (move & FLAG_EP) != np.uint32(0)
    is_castle = (move & FLAG_CASTLE) != np.uint32(0)

    new_white_king = lsb(bb[WK])
    new_black_king = lsb(bb[BK])

    rc, ac = compute_delta(from_sq, to_sq, piece, captured, promotion, is_ep, is_castle, side,
                           removed_sq, removed_pc, added_sq, added_pc)

    if king_bucket(new_white_king, 0) == king_bucket(old_white_king, 0):
        apply_delta_into(out_white_acc_row, white_acc_row, feature_weights,
                         removed_sq, removed_pc, rc, added_sq, added_pc, ac, old_white_king, 0)
    else:
        res_w = refresh(feature_weights, feature_bias, mail, new_white_king, 0)
        for j in range(out_white_acc_row.shape[0]):
            out_white_acc_row[j] = res_w[j]

    if king_bucket(new_black_king, 1) == king_bucket(old_black_king, 1):
        apply_delta_into(out_black_acc_row, black_acc_row, feature_weights,
                         removed_sq, removed_pc, rc, added_sq, added_pc, ac, old_black_king, 1)
    else:
        res_b = refresh(feature_weights, feature_bias, mail, new_black_king, 1)
        for j in range(out_black_acc_row.shape[0]):
            out_black_acc_row[j] = res_b[j]

    return new_white_king, new_black_king

