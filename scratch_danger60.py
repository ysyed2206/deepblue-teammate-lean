import chess, chess.pgn
from deepblue.fastcore import from_fen
from deepblue.eval_terms import king_attack_danger_white_relative, game_phase
from deepblue.fastsearch118 import PHASE_TABLE, evaluate, MG_TABLE, EG_TABLE
g = chess.pgn.read_game(open(r"C:/Users/uniqu/Downloads/aichessathon-round-60-jlu-vs-yumo.pgn"))
board = g.board(); node = g
print("%-5s %-8s %9s %9s   (all White-relative; negative = good for Black)" %
      ("move","black","danger","full eval"))
while node.variations:
    node = node.variation(0); mv = node.move; san = board.san(mv); board.push(mv)
    if board.turn == chess.WHITE and 14 <= board.fullmove_number - 1 <= 27:
        bb, occ, mail, st = from_fen(board.fen())
        ph = game_phase(bb, PHASE_TABLE, 24)
        dz = int(king_attack_danger_white_relative(bb, ph, 24))
        ev = int(evaluate(bb, st, MG_TABLE, EG_TABLE, PHASE_TABLE))
        # evaluate() is side-to-move relative and it is White to move here
        print("%-5d %-8s %9d %9d" % (board.fullmove_number - 1, san, dz, ev))
