"""Exact int16/int32/int64 Numba inference and incremental-update prototype.

This module reads Deep Blue-compatible mailbox and packed-move values, but it
does not import, wrap, or modify ``fastcore.make_move``.  Tests call the two
systems side by side.
"""

# Numba's nopython functions follow the unannotated-array convention used by
# Deep Blue fastcore; their concrete signatures are compiled and inspected at
# runtime rather than expressed through Python's type system.
# mypy: disable-error-code="no-untyped-def"

from __future__ import annotations

import numpy as np
from numba import njit

WHITE = 0
BLACK = 1
NO_PIECE = 15
NUM_FEATURES = 6144
FLAG_EP = np.uint32(1 << 24)
FLAG_CASTLE = np.uint32(1 << 25)


@njit(cache=True, nogil=True, inline="always")
def oriented_square(square, perspective):
    return square if perspective == WHITE else square ^ 56


@njit(cache=True, nogil=True, inline="always")
def bucket_of(king_square, perspective):
    square = oriented_square(king_square, perspective)
    return ((square & 7) >> 1) + (4 if (square >> 3) >= 4 else 0)


@njit(cache=True, nogil=True, inline="always")
def feature_row(piece, square, king_square, perspective):
    relative_colour = (piece // 6) ^ perspective
    piece_type = piece % 6
    return (
        bucket_of(king_square, perspective) * 768
        + relative_colour * 384
        + piece_type * 64
        + oriented_square(square, perspective)
    )


@njit(cache=True, nogil=True, inline="always")
def find_king(mail, perspective):
    king_piece = 5 if perspective == WHITE else 11
    for square in range(64):
        if mail[square] == king_piece:
            return square
    return -1


@njit(cache=True, nogil=True)
def refresh_perspective(mail, feature_weights, feature_bias, perspective, accumulator):
    """Full refresh of one fixed perspective into caller-owned int32 storage."""
    width = feature_bias.shape[0]
    for neuron in range(width):
        accumulator[neuron] = np.int32(feature_bias[neuron])
    king_square = find_king(mail, perspective)
    if king_square < 0:
        return -1
    for square in range(64):
        piece = mail[square]
        if piece < 12:
            row = feature_row(piece, square, king_square, perspective)
            for neuron in range(width):
                accumulator[neuron] += np.int32(feature_weights[row, neuron])
    return 0


@njit(cache=True, nogil=True)
def refresh_all(mail, feature_weights, feature_bias, accumulators):
    """Full refresh of white and black accumulators into int32[2,H]."""
    status_white = refresh_perspective(
        mail, feature_weights, feature_bias, WHITE, accumulators[WHITE]
    )
    status_black = refresh_perspective(
        mail, feature_weights, feature_bias, BLACK, accumulators[BLACK]
    )
    return status_white | status_black


@njit(cache=True, nogil=True, inline="always")
def apply_feature_delta(accumulator, feature_weights, row, direction):
    for neuron in range(accumulator.shape[0]):
        accumulator[neuron] += direction * np.int32(feature_weights[row, neuron])


@njit(cache=True, nogil=True, inline="always")
def rook_move_for_castle(to_square):
    if to_square == 6:
        return 7, 5, 3
    if to_square == 2:
        return 0, 3, 3
    if to_square == 62:
        return 63, 61, 9
    return 56, 59, 9


@njit(cache=True, nogil=True)
def incremental_after_move(
    post_mail,
    move,
    old_side,
    feature_weights,
    feature_bias,
    accumulators,
):
    """Update pre-move accumulators after fastcore has made ``move``."""
    # Explicit signed normalisation avoids Numba unifying uint move fields with
    # signed square transforms (``square ^ 56``) through float64.
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
        own_king = 5 if perspective == WHITE else 11
        bucket_changed = piece == own_king and (
            bucket_of(from_square, perspective) != bucket_of(to_square, perspective)
        )
        if bucket_changed:
            refresh_perspective(
                post_mail,
                feature_weights,
                feature_bias,
                perspective,
                accumulators[perspective],
            )
            continue
        king_square = find_king(post_mail, perspective)
        apply_feature_delta(
            accumulators[perspective],
            feature_weights,
            feature_row(piece, from_square, king_square, perspective),
            -1,
        )
        if captured != NO_PIECE:
            apply_feature_delta(
                accumulators[perspective],
                feature_weights,
                feature_row(captured, capture_square, king_square, perspective),
                -1,
            )
        apply_feature_delta(
            accumulators[perspective],
            feature_weights,
            feature_row(landed, to_square, king_square, perspective),
            1,
        )
        if is_castle:
            rook_from, rook_to, rook = rook_move_for_castle(to_square)
            apply_feature_delta(
                accumulators[perspective],
                feature_weights,
                feature_row(rook, rook_from, king_square, perspective),
                -1,
            )
            apply_feature_delta(
                accumulators[perspective],
                feature_weights,
                feature_row(rook, rook_to, king_square, perspective),
                1,
            )
    return 0


@njit(cache=True, nogil=True)
def incremental_unmake_move(
    restored_mail,
    move,
    old_side,
    feature_weights,
    feature_bias,
    accumulators,
):
    """Reverse accumulator deltas after fastcore has unmade ``move``."""
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
        own_king = 5 if perspective == WHITE else 11
        bucket_changed = piece == own_king and (
            bucket_of(from_square, perspective) != bucket_of(to_square, perspective)
        )
        if bucket_changed:
            refresh_perspective(
                restored_mail,
                feature_weights,
                feature_bias,
                perspective,
                accumulators[perspective],
            )
            continue
        king_square = find_king(restored_mail, perspective)
        apply_feature_delta(
            accumulators[perspective],
            feature_weights,
            feature_row(landed, to_square, king_square, perspective),
            -1,
        )
        if is_castle:
            rook_from, rook_to, rook = rook_move_for_castle(to_square)
            apply_feature_delta(
                accumulators[perspective],
                feature_weights,
                feature_row(rook, rook_to, king_square, perspective),
                -1,
            )
            apply_feature_delta(
                accumulators[perspective],
                feature_weights,
                feature_row(rook, rook_from, king_square, perspective),
                1,
            )
        apply_feature_delta(
            accumulators[perspective],
            feature_weights,
            feature_row(piece, from_square, king_square, perspective),
            1,
        )
        if captured != NO_PIECE:
            apply_feature_delta(
                accumulators[perspective],
                feature_weights,
                feature_row(captured, capture_square, king_square, perspective),
                1,
            )
    return 0


@njit(cache=True, nogil=True, inline="always")
def symmetric_round_divide_scalar(numerator, denominator):
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


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
    """Integer scalar tail from two already-built fixed accumulators."""
    width = accumulators.shape[1]
    stm_perspective = side_to_move
    non_stm_perspective = 1 - side_to_move
    numerator = np.int64(output_bias) * np.int64(qa) * np.int64(qa)
    for neuron in range(width):
        stm_value = accumulators[stm_perspective, neuron]
        non_stm_value = accumulators[non_stm_perspective, neuron]
        if stm_value < 0:
            stm_value = 0
        elif stm_value > qa:
            stm_value = qa
        if non_stm_value < 0:
            non_stm_value = 0
        elif non_stm_value > qa:
            non_stm_value = qa
        numerator += np.int64(output_weights[neuron]) * np.int64(stm_value * stm_value)
        numerator += np.int64(output_weights[width + neuron]) * np.int64(
            non_stm_value * non_stm_value
        )
    scaled = numerator * np.int64(evaluation_scale)
    denominator = np.int64(qb) * np.int64(qa) * np.int64(qa)
    return symmetric_round_divide_scalar(scaled, denominator)


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
def benchmark_tail_loop(
    accumulators, side, output_weights, output_bias, qa, qb, scale, iterations
):
    checksum = np.int64(0)
    for index in range(iterations):
        checksum += evaluate_ready(
            accumulators, side ^ (index & 1), output_weights, output_bias, qa, qb, scale
        )
    return checksum


@njit(cache=True, nogil=True)
def benchmark_refresh_loop(mail, feature_weights, feature_bias, accumulators, iterations):
    checksum = np.int64(0)
    for _ in range(iterations):
        refresh_all(mail, feature_weights, feature_bias, accumulators)
        checksum += accumulators[0, 0]
    return checksum


@njit(cache=True, nogil=True)
def benchmark_perspective_refresh_loop(
    mail, feature_weights, feature_bias, perspective, accumulator, iterations
):
    checksum = np.int64(0)
    for _ in range(iterations):
        refresh_perspective(mail, feature_weights, feature_bias, perspective, accumulator)
        checksum += accumulator[0]
    return checksum


@njit(cache=True, nogil=True)
def benchmark_update_pair_loop(
    pre_mail,
    post_mail,
    move,
    old_side,
    feature_weights,
    feature_bias,
    accumulators,
    iterations,
):
    checksum = np.int64(0)
    for _ in range(iterations):
        incremental_after_move(
            post_mail, move, old_side, feature_weights, feature_bias, accumulators
        )
        checksum += accumulators[0, 0]
        incremental_unmake_move(
            pre_mail, move, old_side, feature_weights, feature_bias, accumulators
        )
    return checksum
