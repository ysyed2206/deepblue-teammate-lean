import chess, chess.pgn, io
g = chess.pgn.read_game(open(r"C:/Users/uniqu/Downloads/aichessathon-round-58-yumo-vs-highestelo.pgn"))
b = g.board()
forced = 0; ours = 0
for mv in g.mainline_moves():
    if b.turn == chess.WHITE:            # Yumo is White
        ours += 1
        n = b.legal_moves.count()
        if n == 1:
            forced += 1
            print("  forced at move %d: %s (only legal move)" % (b.fullmove_number, b.san(mv)))
    b.push(mv)
print("our moves: %d, of which exactly one legal reply: %d" % (ours, forced))
