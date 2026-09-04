"""The donor's shipped **int8 output** fast path (``Network::Evaluate``), as
distinct from the exact int16 reference path in ``reference_eval.py``
(``Network::EvaluateExact``).

The donor's feature transformer stays **int16** end-to-end (an int8 feature
transformer was tried and reverted after a measured -8 Elo regression -- see
`PAWNSTAR_FORMAT.md`). Only the *output* layer is requantised to int8, for a
faster dot product:

    u8(x)   = (clamp(x, 0, QA)^2 + round) >> kInt8Shift     # kInt8Shift = 9
    w8      = round(output_weight / output_w_scale)          # clamped to [-127, 127]
    dot     = sum(u8(stm) * w8_stm) + sum(u8(ntm) * w8_ntm)  # int32-safe accumulation
    output  = dot * output_w_scale * (1 << kInt8Shift)       # undo the u8/w8 scaling
    (then the same Dequant() as the int16 path)

``output_w_scale = max(1, ceil(max(|output_weight|) / 127))`` -- for the
shipped v12 net this is 1 (its output weights already fit int8), so the
w8 requantisation is lossless there; the ``>> 9`` on activations is where the
approximation actually happens.
"""

from __future__ import annotations

import chess
import numpy as np

import features as feat
from reference_eval import dequantize, refresh_accumulator
from weights import QA, NNUEWeights

INT8_SHIFT = 9
INT8_ROUND = 1 << (INT8_SHIFT - 1)


def output_weight_scale(output_weights: np.ndarray) -> int:
    max_abs = int(np.abs(output_weights.astype(np.int64)).max())
    return max(1, -(-max_abs // 127))  # ceil division, donor: (max_ow + 126) / 127


def quantize_output_weights_int8(output_weights: np.ndarray) -> tuple[np.ndarray, int]:
    """Return (int8 weights, scale) such that ``weight ~= int8 * scale``."""
    scale = output_weight_scale(output_weights)
    scaled = np.round(output_weights.astype(np.float64) / scale)
    clamped = np.clip(scaled, -127, 127).astype(np.int8)
    return clamped, scale


def screlu_u8(x: np.ndarray) -> np.ndarray:
    """``(clamp(x,0,QA)^2 + round) >> INT8_SHIFT``, clamped to uint8 range --
    the donor's scalar fallback formula (``u8`` lambda in ``Network::Evaluate``)."""
    clamped = np.clip(x.astype(np.int64), 0, QA)
    squared = clamped * clamped
    shifted = (squared + INT8_ROUND) >> INT8_SHIFT
    return np.minimum(shifted, 255)


def evaluate_int8(weights: NNUEWeights, white_acc: np.ndarray, black_acc: np.ndarray, side_to_move: bool) -> int:
    """The donor's shipped (fast, approximate) evaluator: int8 output dot."""
    w8, scale = quantize_output_weights_int8(weights.output_weights)
    if side_to_move:
        stm_acc, ntm_acc = white_acc, black_acc
    else:
        stm_acc, ntm_acc = black_acc, white_acc

    dot8 = int(np.sum(screlu_u8(stm_acc).astype(np.int64) * w8[0].astype(np.int64)))
    dot8 += int(np.sum(screlu_u8(ntm_acc).astype(np.int64) * w8[1].astype(np.int64)))
    output = dot8 * scale * (1 << INT8_SHIFT)
    return dequantize(output, weights.output_bias)


def evaluate_int8_from_board(weights: NNUEWeights, board: chess.Board) -> int:
    white_acc = refresh_accumulator(weights, board, feat.WHITE)
    black_acc = refresh_accumulator(weights, board, feat.BLACK)
    return evaluate_int8(weights, white_acc, black_acc, board.turn)
