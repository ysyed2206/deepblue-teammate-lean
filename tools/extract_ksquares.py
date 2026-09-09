"""Per-piece-type KING-ZONE SQUARE COUNTS, per side.

Our danger term adds a piece's full weight if it attacks the king zone at all.
Every published attack-units scheme instead multiplies the weight by HOW MANY
zone squares the piece attacks (Chessprogramming wiki, King Safety). This
writes both quantities so the two can be compared on the same positions:

  cols 0-3   pieces attacking White's king zone   (presence)
  cols 4-7   pieces attacking Black's king zone   (presence)
  cols 8-11  zone squares attacked, White's king  (sum of popcounts)
  cols 12-15 zone squares attacked, Black's king
"""
from __future__ import annotations
import multiprocessing as mp, sys
import chess, numpy as np

TYPES=(chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN)

def side(b, us):
    ksq=b.king(us)
    if ksq is None: return [0.]*4,[0.]*4
    zone=chess.SquareSet(chess.BB_KING_ATTACKS[ksq])|chess.SquareSet(chess.BB_SQUARES[ksq])
    pres=[];sq=[]
    for pt in TYPES:
        p=0;s=0
        for f in b.pieces(pt, not us):
            hit=chess.SquareSet(b.attacks_mask(f))&zone
            if hit: p+=1; s+=len(hit)
        pres.append(float(p)); sq.append(float(s))
    return pres,sq

def one(line):
    try: b=chess.Board(line.rstrip("\n").split("\t",1)[1])
    except ValueError: return None
    pw,sw=side(b,chess.WHITE); pb,sb=side(b,chess.BLACK)
    return pw+pb+sw+sb

def main():
    src,out,limit,workers=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
    lines=open(src,encoding="utf-8").readlines()[:limit]
    with mp.Pool(workers) as pool: rows=pool.map(one,lines,chunksize=400)
    keep=[i for i,r in enumerate(rows) if r is not None]
    F=np.array([rows[i] for i in keep],dtype=np.float64)
    np.savez_compressed(out,F=F,keep=np.array(keep))
    print("wrote %s %s"%(out,F.shape))

if __name__=="__main__":
    mp.freeze_support(); main()
