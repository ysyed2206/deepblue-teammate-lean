"""Material values and piece-square tables, generated from interpretable rules.

Nothing here is transcribed from another engine. Every table is produced by an
explicit rule from the small parameter set at the top of its section, so the
parameters are what a tuner will later optimise and what we can defend at review.

Square indexing follows python-chess: a1 = 0, h1 = 7, a8 = 56, h8 = 63. Every
table is written from White's point of view; Black reads the same table through
``square ^ 56``, which mirrors vertically.
"""

from __future__ import annotations

import chess

# --------------------------------------------------------------------------
# Material
# --------------------------------------------------------------------------
# Two values per piece: one for the middlegame, one for the endgame. The
# evaluation interpolates between them on the game phase. Pawns are worth more
# in the endgame (they promote), knights slightly less (fewer targets), rooks
# and queens more (open lines).

MG_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 335,
    chess.ROOK: 500,
    chess.QUEEN: 975,
    chess.KING: 0,
}
EG_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 120,
    chess.KNIGHT: 320,
    chess.BISHOP: 340,
    chess.ROOK: 545,
    chess.QUEEN: 1000,
    chess.KING: 0,
}

# Phase weights. A full board of pieces is 24; a bare-king endgame is 0.
PHASE_WEIGHT: dict[chess.PieceType, int] = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
    chess.KING: 0,
}
TOTAL_PHASE = 24

PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)

# --------------------------------------------------------------------------
# Geometry helpers used by the table generators
# --------------------------------------------------------------------------


def _file_of(square: int) -> int:
    return square & 7


def _rank_of(square: int) -> int:
    return square >> 3


def _centre_distance(square: int) -> float:
    """Manhattan distance from the centre of the board, 1.0 (centre) to 7.0 (corner)."""
    return abs(_file_of(square) - 3.5) + abs(_rank_of(square) - 3.5)


def _on_long_diagonal(square: int) -> bool:
    file, rank = _file_of(square), _rank_of(square)
    return file == rank or file + rank == 7


# --------------------------------------------------------------------------
# Table generators. One function per piece, each driven by named parameters.
# --------------------------------------------------------------------------

# Pawns: advancement matters, and it matters far more in the endgame where a
# passed pawn is a promotion threat. Central files are worth more early.
PAWN_RANK_MG = (0, 0, 2, 6, 14, 28, 60, 0)
PAWN_RANK_EG = (0, 0, 8, 20, 42, 78, 130, 0)
PAWN_FILE_MG = (-6, -2, 4, 10, 10, 4, -2, -6)

# Knights: sharply centralised, punished on the rim, punished on the back rank
# where they are undeveloped.
KNIGHT_CENTRE_MG = 8.0
KNIGHT_CENTRE_EG = 6.0
KNIGHT_CENTRE_PIVOT = 2.0
KNIGHT_BACK_RANK_MG = -12

# Bishops: mildly centralised, rewarded on a long diagonal, punished at home.
BISHOP_CENTRE_MG = 4.0
BISHOP_CENTRE_EG = 3.0
BISHOP_CENTRE_PIVOT = 2.5
BISHOP_LONG_DIAGONAL_MG = 6
BISHOP_BACK_RANK_MG = -8

# Rooks: files matter more than squares, and the seventh rank is the classic
# rook square. In the endgame the table is nearly flat.
ROOK_FILE_MG = (-4, -2, 0, 4, 4, 0, -2, -4)
ROOK_SEVENTH_MG = 20
ROOK_EIGHTH_MG = 6
ROOK_ADVANCED_EG = 4

# Queens: mild centralisation, stronger in the endgame.
QUEEN_CENTRE_MG = 2.0
QUEEN_CENTRE_EG = 4.0
QUEEN_CENTRE_PIVOT = 3.0

# Kings: hide behind a castled shelter in the middlegame, walk to the centre in
# the endgame. This sign flip is the single largest tapered term in the table.
KING_FILE_MG = (14, 20, 8, -12, -18, 0, 20, 12)
KING_RANK_PENALTY_MG = 14
KING_CENTRE_EG = 8.0
KING_CENTRE_PIVOT_EG = 3.0


def _pawn_tables() -> tuple[list[int], list[int]]:
    mg, eg = [], []
    for square in range(64):
        rank, file = _rank_of(square), _file_of(square)
        mg.append(PAWN_RANK_MG[rank] + PAWN_FILE_MG[file])
        eg.append(PAWN_RANK_EG[rank])
    return mg, eg


def _knight_tables() -> tuple[list[int], list[int]]:
    mg, eg = [], []
    for square in range(64):
        distance = _centre_distance(square)
        value = round(-KNIGHT_CENTRE_MG * (distance - KNIGHT_CENTRE_PIVOT))
        if _rank_of(square) == 0:
            value += KNIGHT_BACK_RANK_MG
        mg.append(value)
        eg.append(round(-KNIGHT_CENTRE_EG * (distance - KNIGHT_CENTRE_PIVOT)))
    return mg, eg


def _bishop_tables() -> tuple[list[int], list[int]]:
    mg, eg = [], []
    for square in range(64):
        distance = _centre_distance(square)
        value = round(-BISHOP_CENTRE_MG * (distance - BISHOP_CENTRE_PIVOT))
        if _on_long_diagonal(square):
            value += BISHOP_LONG_DIAGONAL_MG
        if _rank_of(square) == 0:
            value += BISHOP_BACK_RANK_MG
        mg.append(value)
        eg.append(round(-BISHOP_CENTRE_EG * (distance - BISHOP_CENTRE_PIVOT)))
    return mg, eg


def _rook_tables() -> tuple[list[int], list[int]]:
    mg, eg = [], []
    for square in range(64):
        rank = _rank_of(square)
        value = ROOK_FILE_MG[_file_of(square)]
        if rank == 6:
            value += ROOK_SEVENTH_MG
        elif rank == 7:
            value += ROOK_EIGHTH_MG
        mg.append(value)
        eg.append(ROOK_ADVANCED_EG if rank >= 4 else 0)
    return mg, eg


def _queen_tables() -> tuple[list[int], list[int]]:
    mg, eg = [], []
    for square in range(64):
        distance = _centre_distance(square)
        mg.append(round(-QUEEN_CENTRE_MG * (distance - QUEEN_CENTRE_PIVOT)))
        eg.append(round(-QUEEN_CENTRE_EG * (distance - QUEEN_CENTRE_PIVOT)))
    return mg, eg


def _king_tables() -> tuple[list[int], list[int]]:
    mg, eg = [], []
    for square in range(64):
        rank = _rank_of(square)
        mg.append(KING_FILE_MG[_file_of(square)] - KING_RANK_PENALTY_MG * rank)
        distance = _centre_distance(square)
        eg.append(round(-KING_CENTRE_EG * (distance - KING_CENTRE_PIVOT_EG)))
    return mg, eg


_GENERATORS = {
    chess.PAWN: _pawn_tables,
    chess.KNIGHT: _knight_tables,
    chess.BISHOP: _bishop_tables,
    chess.ROOK: _rook_tables,
    chess.QUEEN: _queen_tables,
    chess.KING: _king_tables,
}

# PST_MG[piece_type][square] for White. Black uses square ^ 56.
PST_MG: dict[chess.PieceType, list[int]] = {}
PST_EG: dict[chess.PieceType, list[int]] = {}
for _piece_type, _generator in _GENERATORS.items():
    PST_MG[_piece_type], PST_EG[_piece_type] = _generator()

# Fused material + PST lookups, so evaluation is one table read per piece.
# WHITE_MG[piece_type][square] already contains the material value.
WHITE_MG: dict[chess.PieceType, list[int]] = {
    pt: [MG_VALUE[pt] + PST_MG[pt][sq] for sq in range(64)] for pt in PIECE_TYPES
}
WHITE_EG: dict[chess.PieceType, list[int]] = {
    pt: [EG_VALUE[pt] + PST_EG[pt][sq] for sq in range(64)] for pt in PIECE_TYPES
}
BLACK_MG: dict[chess.PieceType, list[int]] = {
    pt: [WHITE_MG[pt][sq ^ 56] for sq in range(64)] for pt in PIECE_TYPES
}
BLACK_EG: dict[chess.PieceType, list[int]] = {
    pt: [WHITE_EG[pt][sq ^ 56] for sq in range(64)] for pt in PIECE_TYPES
}

# --------------------------------------------------------------------------
# Precomputed pawn-structure masks
# --------------------------------------------------------------------------

FILE_MASK: list[int] = [chess.BB_FILES[_file_of(sq)] for sq in range(64)]
ADJACENT_FILE_MASK: list[int] = []
for _sq in range(64):
    _f = _file_of(_sq)
    _mask = 0
    if _f > 0:
        _mask |= chess.BB_FILES[_f - 1]
    if _f < 7:
        _mask |= chess.BB_FILES[_f + 1]
    ADJACENT_FILE_MASK.append(_mask)

# Squares strictly in front of a square, on its own file, for each colour.
FRONT_FILE_MASK: dict[chess.Color, list[int]] = {chess.WHITE: [], chess.BLACK: []}
for _sq in range(64):
    _f, _r = _file_of(_sq), _rank_of(_sq)
    _white = 0
    for _rr in range(_r + 1, 8):
        _white |= chess.BB_SQUARES[_rr * 8 + _f]
    _black = 0
    for _rr in range(0, _r):
        _black |= chess.BB_SQUARES[_rr * 8 + _f]
    FRONT_FILE_MASK[chess.WHITE].append(_white)
    FRONT_FILE_MASK[chess.BLACK].append(_black)

# A pawn is passed if no enemy pawn stands on its own file or either adjacent
# file, on any rank strictly ahead of it. An enemy pawn abreast on an adjacent
# file cannot stop it, so squares on the pawn's own rank are deliberately not
# part of the mask.
PASSED_MASK: dict[chess.Color, list[int]] = {chess.WHITE: [], chess.BLACK: []}
for _colour in (chess.WHITE, chess.BLACK):
    for _sq in range(64):
        _f, _r = _file_of(_sq), _rank_of(_sq)
        _ahead = range(_r + 1, 8) if _colour == chess.WHITE else range(0, _r)
        _span = 0
        for _ff in range(max(0, _f - 1), min(7, _f + 1) + 1):
            for _rr in _ahead:
                _span |= chess.BB_SQUARES[_rr * 8 + _ff]
        PASSED_MASK[_colour].append(_span)

# --------------------------------------------------------------------------
# Structural bonuses
# --------------------------------------------------------------------------

BISHOP_PAIR_MG = 28
BISHOP_PAIR_EG = 48
DOUBLED_PAWN_MG = -8
DOUBLED_PAWN_EG = -22
ISOLATED_PAWN_MG = -14
ISOLATED_PAWN_EG = -12
PASSED_PAWN_MG = (0, 4, 8, 18, 36, 62, 96, 0)
PASSED_PAWN_EG = (0, 10, 20, 38, 68, 112, 168, 0)
ROOK_OPEN_FILE_MG = 22
ROOK_OPEN_FILE_EG = 10
ROOK_SEMI_OPEN_FILE_MG = 10
ROOK_SEMI_OPEN_FILE_EG = 6
TEMPO_BONUS = 12

# Mobility is scored per safe destination square, tapered per piece type.
MOBILITY_MG: dict[chess.PieceType, int] = {
    chess.KNIGHT: 4,
    chess.BISHOP: 4,
    chess.ROOK: 2,
    chess.QUEEN: 1,
}
MOBILITY_EG: dict[chess.PieceType, int] = {
    chess.KNIGHT: 4,
    chess.BISHOP: 5,
    chess.ROOK: 4,
    chess.QUEEN: 2,
}

# --------------------------------------------------------------------------
# Search score conventions
# --------------------------------------------------------------------------

MATE_SCORE = 30_000
MATE_THRESHOLD = MATE_SCORE - 1_000  # scores above this are mate-in-n
INFINITY = 31_000
DRAW_SCORE = 0
