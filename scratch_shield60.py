"""What did our king safety term see in round 60, flat vs graded?

We are BLACK. King on g8. The flat shield mask counts rank-1 AND rank-2 as an
equally intact shield, so ...g6 (two squares from the king) scores the same as
a pawn still on g7. Both terms are White-relative, so a NEGATIVE number here
means the term thinks Black's king is worse off.
"""
import chess, chess.pgn
from deepblue.fastcore import from_fen
from deepblue.eval_terms import (king_safety_white_relative,
                                 king_safety_graded_white_relative, game_phase,
                                 SHIELD_ADVANCED_PENALTY_STRONG, SHIELD_MISSING_PENALTY_STRONG)
from deepblue.fastsearch118 import PHASE_TABLE

g = chess.pgn.read_game(open(r"C:/Users/uniqu/Downloads/aichessathon-round-60-jlu-vs-yumo.pgn"))
board = g.board()
print("%-6s %-9s %-22s %7s %7s" % ("move", "black", "black king pawns", "flat", "graded"))
node = g
while node.variations:
    node = node.variation(0)
    mv = node.move
    san = board.san(mv)
    board.push(mv)
    if board.turn == chess.WHITE and board.fullmove_number - 1 in (12,13,14,15,16,17,20,21,22,24,26):
        bb, occ, mail, st = from_fen(board.fen())
        ph = game_phase(bb, PHASE_TABLE, 24)
        flat = int(king_safety_white_relative(bb, ph, 24))
        grad = int(king_safety_graded_white_relative(bb, ph, 24,
                   SHIELD_ADVANCED_PENALTY_STRONG, SHIELD_MISSING_PENALTY_STRONG))
        ksq = board.king(chess.BLACK)
        shelter = [chess.square_name(s) for s in (chess.F7,chess.G7,chess.H7,chess.F6,chess.G6,chess.H6)
                   if board.piece_at(s) == chess.Piece(chess.PAWN, chess.BLACK)]
        print("%-6d %-9s %-22s %+7d %+7d" % (board.fullmove_number - 1, san,
              chess.square_name(ksq) + ": " + ",".join(shelter), flat, grad))
