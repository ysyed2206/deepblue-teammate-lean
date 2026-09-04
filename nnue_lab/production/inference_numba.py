"""Integer Numba runtime for no-bucket Perspective Chess768.

The incremental routines depend only on a packed Deep Blue move and the old
side to move.  They cannot refresh on king movement: no feature row depends on
a king anchor, so kings are updated through the same remove/add delta as every
other ordinary piece.
"""

# mypy: disable-error-code="no-untyped-def"

from __future__ import annotations

import numpy as np
from numba import njit

WHITE = 0
BLACK = 1
NO_PIECE = 15
NUM_FEATURES = 768
FLAG_EP = np.uint32(1 << 24)
FLAG_CASTLE = np.uint32(1 << 25)


@njit(cache=True, nogil=True, inline="always")
def oriented_square(square, perspective):
    return square if perspective == WHITE else square ^ 56


@njit(cache=True, nogil=True, inline="always")
def feature_row(piece, square, perspective):
    relative_colour = (piece // 6) ^ perspective
    return relative_colour * 384 + (piece % 6) * 64 + oriented_square(
        square, perspective
    )


@njit(cache=True, nogil=True)
def refresh_perspective(mail, feature_weights, feature_bias, perspective, accumulator):
    """Fully rebuild one fixed accumulator in caller-owned int32 storage."""
    width = feature_bias.shape[0]
    for neuron in range(width):
        accumulator[neuron] = np.int32(feature_bias[neuron])
    for square in range(64):
        piece = mail[square]
        if piece < 12:
            row = feature_row(piece, square, perspective)
            for neuron in range(width):
                accumulator[neuron] += np.int32(feature_weights[row, neuron])


@njit(cache=True, nogil=True)
def refresh_all(mail, feature_weights, feature_bias, accumulators):
    """Fully rebuild fixed white and black accumulators."""
    refresh_perspective(
        mail, feature_weights, feature_bias, WHITE, accumulators[WHITE]
    )
    refresh_perspective(
        mail, feature_weights, feature_bias, BLACK, accumulators[BLACK]
    )
    return 0


@njit(cache=True, nogil=True, inline="always")
def castle_rook_delta(to_square):
    if to_square == 6:
        return 7, 5, 3
    if to_square == 2:
        return 0, 3, 3
    if to_square == 62:
        return 63, 61, 9
    return 56, 59, 9


@njit(cache=True, nogil=True, inline="always")
def _apply_move_delta(move, old_side, feature_weights, accumulators, direction):
    from_square = np.int64(move & np.uint32(63))
    to_square = np.int64((move >> np.uint32(6)) & np.uint32(63))
    piece = np.int64((move >> np.uint32(12)) & np.uint32(15))
    captured = np.int64((move >> np.uint32(16)) & np.uint32(15))
    promotion = np.int64((move >> np.uint32(20)) & np.uint32(15))
    is_ep = (move & FLAG_EP) != np.uint32(0)
    is_castle = (move & FLAG_CASTLE) != np.uint32(0)
    landed = promotion if promotion != NO_PIECE else piece
    capture_square = to_square
    if is_ep:
        capture_square = to_square - 8 if old_side == WHITE else to_square + 8

    for perspective in (WHITE, BLACK):
        from_row = feature_row(piece, from_square, perspective)
        to_row = feature_row(landed, to_square, perspective)
        capture_row = -1
        if captured != NO_PIECE:
            capture_row = feature_row(captured, capture_square, perspective)
        rook_from_row = -1
        rook_to_row = -1
        if is_castle:
            rook_from, rook_to, rook = castle_rook_delta(to_square)
            rook_from_row = feature_row(rook, rook_from, perspective)
            rook_to_row = feature_row(rook, rook_to, perspective)

        accumulator = accumulators[perspective]
        for neuron in range(accumulator.shape[0]):
            delta = np.int32(feature_weights[to_row, neuron]) - np.int32(
                feature_weights[from_row, neuron]
            )
            if capture_row >= 0:
                delta -= np.int32(feature_weights[capture_row, neuron])
            if rook_from_row >= 0:
                delta += np.int32(feature_weights[rook_to_row, neuron])
                delta -= np.int32(feature_weights[rook_from_row, neuron])
            accumulator[neuron] += direction * delta


@njit(cache=True, nogil=True)
def incremental_make(move, old_side, feature_weights, accumulators):
    """Apply feature deltas after ``fastcore.make_move``."""
    _apply_move_delta(move, old_side, feature_weights, accumulators, 1)


@njit(cache=True, nogil=True)
def incremental_unmake(move, old_side, feature_weights, accumulators):
    """Exactly reverse ``incremental_make`` after ``fastcore.unmake_move``."""
    _apply_move_delta(move, old_side, feature_weights, accumulators, -1)


@njit(cache=True, nogil=True, inline="always")
def truncating_divide(numerator, denominator):
    """Signed integer division with C/C++ truncation-towards-zero semantics."""
    if numerator >= 0:
        return numerator // denominator
    return -((-numerator) // denominator)


@njit(cache=True, nogil=True)
def evaluate_ready(
    accumulators,
    side_to_move,
    output_weights,
    output_bias,
    qa,
    qb,
    evaluation_scale,
):
    """Evaluate two ready accumulators with exact int64 SCReLU arithmetic."""
    width = accumulators.shape[1]
    dot = np.int64(0)
    for neuron in range(width):
        stm = accumulators[side_to_move, neuron]
        opponent = accumulators[1 - side_to_move, neuron]
        if stm < 0:
            stm = 0
        elif stm > qa:
            stm = qa
        if opponent < 0:
            opponent = 0
        elif opponent > qa:
            opponent = qa
        stm64 = np.int64(stm)
        opponent64 = np.int64(opponent)
        dot += np.int64(output_weights[neuron]) * stm64 * stm64
        dot += (
            np.int64(output_weights[width + neuron]) * opponent64 * opponent64
        )
    # Pawnstar-compatible Q1 dequantisation: the dot starts in QA^2*QB
    # units, while output_bias is serialized directly in QA*QB units.
    dequantized = truncating_divide(dot, np.int64(qa)) + np.int64(output_bias)
    return truncating_divide(
        dequantized * np.int64(evaluation_scale),
        np.int64(qa) * np.int64(qb),
    )


@njit(cache=True, nogil=True)
def full_evaluate(
    mail,
    side_to_move,
    feature_weights,
    feature_bias,
    output_weights,
    output_bias,
    qa,
    qb,
    evaluation_scale,
    accumulators,
):
    refresh_all(mail, feature_weights, feature_bias, accumulators)
    return evaluate_ready(
        accumulators,
        side_to_move,
        output_weights,
        output_bias,
        qa,
        qb,
        evaluation_scale,
    )


@njit(cache=True, nogil=True)
def benchmark_refresh_loop(mail, weights, bias, accumulators, iterations):
    checksum = np.int64(0)
    for _ in range(iterations):
        refresh_all(mail, weights, bias, accumulators)
        checksum += accumulators[WHITE, 0]
    return checksum


@njit(cache=True, nogil=True)
def benchmark_tail_loop(
    accumulators, output_weights, output_bias, qa, qb, scale, iterations
):
    checksum = np.int64(0)
    for iteration in range(iterations):
        checksum += evaluate_ready(
            accumulators,
            iteration & 1,
            output_weights,
            output_bias,
            qa,
            qb,
            scale,
        )
    return checksum


@njit(cache=True, nogil=True)
def benchmark_alternating_update_loop(
    move, old_side, weights, accumulators, iterations
):
    checksum = np.int64(0)
    for iteration in range(iterations):
        if (iteration & 1) == 0:
            incremental_make(move, old_side, weights, accumulators)
        else:
            incremental_unmake(move, old_side, weights, accumulators)
        checksum += accumulators[WHITE, 0]
    return checksum


@njit(cache=True, nogil=True)
def benchmark_alternating_update_tail_loop(
    move,
    old_side,
    weights,
    accumulators,
    output_weights,
    output_bias,
    qa,
    qb,
    scale,
    iterations,
):
    checksum = np.int64(0)
    for iteration in range(iterations):
        if (iteration & 1) == 0:
            incremental_make(move, old_side, weights, accumulators)
            side = 1 - old_side
        else:
            incremental_unmake(move, old_side, weights, accumulators)
            side = old_side
        checksum += evaluate_ready(
            accumulators, side, output_weights, output_bias, qa, qb, scale
        )
    return checksum
