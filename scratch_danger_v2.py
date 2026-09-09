import chess, chess.pgn
from deepblue.fastcore import from_fen
from deepblue.eval_terms import (king_attack_danger_white_relative,
                                 king_attack_danger_v2_white_relative, game_phase)
from deepblue.fastsearch118 import PHASE_TABLE
g = chess.pgn.read_game(open(r"C:/Users/uniqu/Downloads/aichessathon-round-60-jlu-vs-yumo.pgn"))
board = g.board(); node = g
print("%-5s %-8s %8s %8s" % ("move","black","old","v2"))
while node.variations:
    node = node.variation(0); mv = node.move; san = board.san(mv); board.push(mv)
    if board.turn == chess.WHITE and 14 <= board.fullmove_number - 1 <= 24:
        bb, occ, mail, st = from_fen(board.fen())
        ph = game_phase(bb, PHASE_TABLE, 24)
        print("%-5d %-8s %8d %8d" % (board.fullmove_number - 1, san,
              int(king_attack_danger_white_relative(bb, ph, 24)),
              int(king_attack_danger_v2_white_relative(bb, ph, 24))))
# symmetry check
print("\ncolour symmetry (must all be 0):")
worst = 0
for f in ["r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
          "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
          "r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 11"]:
    b = chess.Board(f)
    a = int(king_attack_danger_v2_white_relative(from_fen(f)[0], 24, 24))
    c = int(king_attack_danger_v2_white_relative(from_fen(b.mirror().fen())[0], 24, 24))
    worst = max(worst, abs(a-c)); print("  %+5d vs %+5d  diff %d" % (a, c, a-c))
print("max asymmetry: %d" % worst)
