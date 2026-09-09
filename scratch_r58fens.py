import chess, chess.pgn
g = chess.pgn.read_game(open(r"C:/Users/uniqu/Downloads/aichessathon-round-58-yumo-vs-highestelo.pgn"))
b = g.board()
want = {18, 19, 21, 25, 33}
for mv in g.mainline_moves():
    if b.turn == chess.WHITE and b.fullmove_number in want:
        print("%d. %-6s  %s" % (b.fullmove_number, b.san(mv), b.fen()))
    b.push(mv)
