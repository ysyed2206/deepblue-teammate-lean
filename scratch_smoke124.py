import time
from deepblue.fastcore import from_fen
from deepblue.fastsearch124 import FastEngine124
e=FastEngine124()
t=time.time(); e.search(from_fen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'), 200, 400)
print('compile %.1fs'%(time.time()-t))
for name,f in [('start','rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'),
               ('italian','r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4'),
               ('endgame','8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1'),
               ('kiwipete','r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1'),
               ('midgame','r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 11')]:
    mv,sc,d,n,ms = e.search(from_fen(f), 2000, 2800)
    print('%-9s %s d%-2d %8d nodes'%(name,mv,d,n))
