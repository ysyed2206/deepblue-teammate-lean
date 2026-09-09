import chess
from deepblue.fastcore import from_fen, KNIGHT_ATTACKS, bishop_attacks, rook_attacks, queen_attacks, lsb
from deepblue.eval_terms import KING_ATTACKS
import numpy as np
# position after 17...Qxa2 (White to move), the height of White's attack
FEN = None
import chess.pgn
g = chess.pgn.read_game(open(r"C:/Users/uniqu/Downloads/aichessathon-round-60-jlu-vs-yumo.pgn"))
b = g.board()
for mv in g.mainline_moves():
    b.push(mv)
    if b.fullmove_number == 18 and b.turn == chess.WHITE:
        FEN = b.fen(); break
print(FEN); print(b, "\n")
bb, occ, mail, st = from_fen(FEN)
occupied = np.uint64(0)
for i in range(12): occupied |= bb[i]
bk = lsb(bb[11]); wk = lsb(bb[5])
for name, ksq in (("black king", bk), ("white king", wk)):
    zone = KING_ATTACKS[ksq] | (np.uint64(1) << np.uint64(ksq))
    print("%s on %s, zone = %s" % (name, chess.square_name(int(ksq)),
          " ".join(chess.square_name(s) for s in range(64) if int(zone) >> s & 1)))
    idxs = (1,2,3,4) if name.startswith("black") else (7,8,9,10)
    for label, i, fn in (("N",idxs[0],None),("B",idxs[1],bishop_attacks),
                         ("R",idxs[2],rook_attacks),("Q",idxs[3],queen_attacks)):
        bits = int(bb[i])
        while bits:
            sq = lsb(np.uint64(bits)); bits &= bits-1
            att = int(KNIGHT_ATTACKS[sq]) if fn is None else int(fn(np.uint64(sq), occupied))
            hit = att & int(zone)
            print("   %s on %-3s hits %d zone squares %s" % (label, chess.square_name(int(sq)),
                  bin(hit).count("1"),
                  " ".join(chess.square_name(s) for s in range(64) if hit >> s & 1)))
    print()
