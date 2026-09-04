"""Reference (exact int16) NNUE evaluator -- donor ``Network::EvaluateExact``.

Deliberately simple over fast: this is "obviously correct", not optimised.
The Numba/incremental fast path lives in ``incremental.py``; this module is
what that path is checked against, and what ``DIFFERENTIAL_RESULTS.md`` was
generated with.

Forward pass (verbatim from ``src/nnue.h`` ``Network::EvaluateExact`` / ``Dequant``):

    stm_acc, ntm_acc   = the two perspective accumulators, side-to-move first
    screlu(x)          = clamp(x, 0, QA) ** 2
    dot                = sum(screlu(stm_acc) * w_stm) + sum(screlu(ntm_acc) * w_ntm)   # int64
    dot               //= QA          # SCReLU leaves QA*QA*QB units; reduce one QA
    dot               += output_bias  # bias is in QA*QB units
    dot               *= SCALE
    dot               //= (QA * QB)   # -> centipawns, side-to-move relative

All divisions are C++ ``/=`` on ``int64_t``, i.e. truncation toward zero --
NOT Python's floor-dividing ``//``. ``trunc_div`` below replicates that.
"""

from __future__ import annotations

import chess
import numpy as np

import features as feat
from weights import QA, QB, SCALE, NNUEWeights

MAX_EVALUATION = 30000  # generous bound; the donor's kMaxEvaluation is search-tuned, not part of the format


def trunc_div(a: int, b: int) -> int:
    """C++ int64_t division: truncates toward zero (unlike Python's ``//``,
    which floors toward -infinity)."""
    q = a // b
    if q < 0 and a % b != 0:
        q += 1
    return q


def screlu(x: np.ndarray) -> np.ndarray:
    """clamp(x, 0, QA) ** 2, as int64 (avoids int16/int32 overflow on the square)."""
    clamped = np.clip(x.astype(np.int64), 0, QA)
    return clamped * clamped


def refresh_accumulator(weights: NNUEWeights, board: chess.Board, perspective: int) -> np.ndarray:
    """Full rebuild of one perspective's accumulator: bias + every active
    feature's weight column, with int16 wraparound addition (matches the
    donor's ``AddColumn`` on ``std::array<int16_t, H>``)."""
    acc = weights.feature_bias.astype(np.int16).copy()
    for row in feat.active_features(board, perspective):
        acc = (acc + weights.feature_weights[row]).astype(np.int16)
    return acc


def dequantize(dot: int, output_bias: int) -> int:
    output = trunc_div(int(dot), QA)
    output += output_bias
    output *= SCALE
    output = trunc_div(output, QA * QB)
    return max(-MAX_EVALUATION, min(MAX_EVALUATION, output))


def evaluate_from_accumulators(weights: NNUEWeights, white_acc: np.ndarray, black_acc: np.ndarray, side_to_move: bool) -> int:
    """``side_to_move``: True for White to move (python-chess ``chess.WHITE``),
    False for Black -- matches ``board.turn``. Returns centipawns, relative to
    the side to move (donor convention)."""
    if side_to_move:  # White to move
        stm_acc, ntm_acc = white_acc, black_acc
    else:
        stm_acc, ntm_acc = black_acc, white_acc

    dot = int(np.sum(screlu(stm_acc) * weights.output_weights[0].astype(np.int64)))
    dot += int(np.sum(screlu(ntm_acc) * weights.output_weights[1].astype(np.int64)))
    return dequantize(dot, weights.output_bias)


def evaluate(weights: NNUEWeights, board: chess.Board) -> int:
    """Full-refresh reference evaluation of ``board``. Centipawns, relative to
    the side to move."""
    white_acc = refresh_accumulator(weights, board, feat.WHITE)
    black_acc = refresh_accumulator(weights, board, feat.BLACK)
    return evaluate_from_accumulators(weights, white_acc, black_acc, board.turn)
