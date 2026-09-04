"""Tapered evaluation over a python-chess board.

The score is always returned from the point of view of the side to move, which
is what negamax expects. Internally everything is accumulated from White's
point of view and flipped once at the end.

Two scores are accumulated in parallel - a middlegame score and an endgame
score - and interpolated on the game phase. The phase runs from 24 (all pieces
on) to 0 (bare kings), computed from non-pawn material only, so trading pawns
does not shift the interpolation.
"""

from __future__ import annotations

import chess

from deepblue.constants import (
    BISHOP_PAIR_EG,
    BISHOP_PAIR_MG,
    BLACK_EG,
    BLACK_MG,
    DOUBLED_PAWN_EG,
    DOUBLED_PAWN_MG,
    FILE_MASK,
    ISOLATED_PAWN_EG,
    ISOLATED_PAWN_MG,
    MOBILITY_EG,
    MOBILITY_MG,
    PASSED_MASK,
    PASSED_PAWN_EG,
    PASSED_PAWN_MG,
    PHASE_WEIGHT,
    ROOK_OPEN_FILE_EG,
    ROOK_OPEN_FILE_MG,
    ROOK_SEMI_OPEN_FILE_EG,
    ROOK_SEMI_OPEN_FILE_MG,
    TEMPO_BONUS,
    TOTAL_PHASE,
    WHITE_EG,
    WHITE_MG,
)
from deepblue.constants import ADJACENT_FILE_MASK as ADJ

popcount = chess.popcount
scan = chess.scan_forward

# Mobility is the most expensive term in this evaluation: measured at 21 us of a
# 50 us evaluation, 42% of the cost for one term. It is off by default because at
# S0's node rate the depth that buys is worth more, and it is behind a flag so
# its Elo can be measured by A/B rather than assumed. See EXPERIMENTS.md EXP-002.
USE_MOBILITY = False


def game_phase(board: chess.Board) -> int:
    """24 with a full board of pieces, 0 with bare kings."""
    phase = (
        PHASE_WEIGHT[chess.KNIGHT] * popcount(board.knights)
        + PHASE_WEIGHT[chess.BISHOP] * popcount(board.bishops)
        + PHASE_WEIGHT[chess.ROOK] * popcount(board.rooks)
        + PHASE_WEIGHT[chess.QUEEN] * popcount(board.queens)
    )
    return phase if phase < TOTAL_PHASE else TOTAL_PHASE


def evaluate(board: chess.Board) -> int:
    """Static evaluation in centipawns, from the side to move's point of view."""
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    pawns = board.pawns
    knights = board.knights
    bishops = board.bishops
    rooks = board.rooks
    queens = board.queens

    mg = 0
    eg = 0

    # ---- material and piece-square tables -------------------------------
    for piece_type, board_bb in (
        (chess.PAWN, pawns),
        (chess.KNIGHT, knights),
        (chess.BISHOP, bishops),
        (chess.ROOK, rooks),
        (chess.QUEEN, queens),
        (chess.KING, board.kings),
    ):
        white_mg, white_eg = WHITE_MG[piece_type], WHITE_EG[piece_type]
        black_mg, black_eg = BLACK_MG[piece_type], BLACK_EG[piece_type]
        for square in scan(board_bb & white):
            mg += white_mg[square]
            eg += white_eg[square]
        for square in scan(board_bb & black):
            mg -= black_mg[square]
            eg -= black_eg[square]

    # ---- bishop pair -----------------------------------------------------
    if popcount(bishops & white) >= 2:
        mg += BISHOP_PAIR_MG
        eg += BISHOP_PAIR_EG
    if popcount(bishops & black) >= 2:
        mg -= BISHOP_PAIR_MG
        eg -= BISHOP_PAIR_EG

    # ---- pawn structure --------------------------------------------------
    white_pawns = pawns & white
    black_pawns = pawns & black
    for square in scan(white_pawns):
        file_mask = FILE_MASK[square]
        if white_pawns & file_mask & ~chess.BB_SQUARES[square]:
            mg += DOUBLED_PAWN_MG
            eg += DOUBLED_PAWN_EG
        if not white_pawns & ADJ[square]:
            mg += ISOLATED_PAWN_MG
            eg += ISOLATED_PAWN_EG
        if not black_pawns & PASSED_MASK[chess.WHITE][square]:
            rank = square >> 3
            mg += PASSED_PAWN_MG[rank]
            eg += PASSED_PAWN_EG[rank]
    for square in scan(black_pawns):
        file_mask = FILE_MASK[square]
        if black_pawns & file_mask & ~chess.BB_SQUARES[square]:
            mg -= DOUBLED_PAWN_MG
            eg -= DOUBLED_PAWN_EG
        if not black_pawns & ADJ[square]:
            mg -= ISOLATED_PAWN_MG
            eg -= ISOLATED_PAWN_EG
        if not white_pawns & PASSED_MASK[chess.BLACK][square]:
            rank = 7 - (square >> 3)
            mg -= PASSED_PAWN_MG[rank]
            eg -= PASSED_PAWN_EG[rank]

    # ---- rooks on open and half-open files -------------------------------
    for square in scan(rooks & white):
        file_mask = FILE_MASK[square]
        if not pawns & file_mask:
            mg += ROOK_OPEN_FILE_MG
            eg += ROOK_OPEN_FILE_EG
        elif not white_pawns & file_mask:
            mg += ROOK_SEMI_OPEN_FILE_MG
            eg += ROOK_SEMI_OPEN_FILE_EG
    for square in scan(rooks & black):
        file_mask = FILE_MASK[square]
        if not pawns & file_mask:
            mg -= ROOK_OPEN_FILE_MG
            eg -= ROOK_OPEN_FILE_EG
        elif not black_pawns & file_mask:
            mg -= ROOK_SEMI_OPEN_FILE_MG
            eg -= ROOK_SEMI_OPEN_FILE_EG

    # ---- mobility --------------------------------------------------------
    if USE_MOBILITY:
        attacks = board.attacks_mask
        for piece_type, board_bb in (
            (chess.KNIGHT, knights),
            (chess.BISHOP, bishops),
            (chess.ROOK, rooks),
            (chess.QUEEN, queens),
        ):
            weight_mg = MOBILITY_MG[piece_type]
            weight_eg = MOBILITY_EG[piece_type]
            for square in scan(board_bb & white):
                count = popcount(attacks(square) & ~white)
                mg += weight_mg * count
                eg += weight_eg * count
            for square in scan(board_bb & black):
                count = popcount(attacks(square) & ~black)
                mg -= weight_mg * count
                eg -= weight_eg * count

    # ---- taper and orient -------------------------------------------------
    phase = game_phase(board)
    blended = mg * phase + eg * (TOTAL_PHASE - phase)
    # Truncate toward zero, not floor. Floor division is asymmetric about zero,
    # which made evaluate(position) != evaluate(mirrored position) by one
    # centipawn on roughly 40% of positions. Colour symmetry is a property the
    # evaluation must have exactly, not nearly.
    score = blended // TOTAL_PHASE if blended >= 0 else -((-blended) // TOTAL_PHASE)
    if board.turn == chess.BLACK:
        score = -score
    return score + TEMPO_BONUS
