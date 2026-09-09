import chess, chess.pgn
g = chess.pgn.read_game(open(r"C:/Users/uniqu/Downloads/aichessathon-round-58-yumo-vs-highestelo.pgn"))
b = g.board()
prev_clk = None
print("%-5s %-8s %6s %7s %6s" % ("move", "played", "legal", "spent", "inchk"))
import re
node = g
while node.variations:
    node = node.variation(0)
    board_before = node.parent.board()
    if board_before.turn == chess.WHITE:
        clk = node.clock()
        spent = ""
        if clk is not None and prev_clk is not None:
            spent = "%.2fs" % (prev_clk + 0.5 - clk)
        print("%-5d %-8s %6d %7s %6s" % (
            board_before.fullmove_number, board_before.san(node.move),
            board_before.legal_moves.count(), spent,
            "yes" if board_before.is_check() else ""))
        prev_clk = clk
