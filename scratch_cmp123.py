import time
from deepblue.fastcore import from_fen
from deepblue.fastsearch118 import FastEngine118
from deepblue.fastsearch123 import FastEngine123
tests=[('start','rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'),
       ('italian','r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4'),
       ('endgame','8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1'),
       ('kiwipete','r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1'),
       ('midgame','r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 11')]
eng={}
for label,cls in (('118',FastEngine118),('123',FastEngine123)):
    e=cls(); e.search(from_fen(tests[0][1]), 200, 400)   # warm the JIT
    eng[label]=e
print('%-9s %-22s %-22s'%('pos','118 (move d nodes)','123 (move d nodes)'))
for name,f in tests:
    row=[]
    for label in ('118','123'):
        mv,sc,d,n,ms = eng[label].search(from_fen(f), 2000, 2800)
        row.append('%s d%-2d %8d'%(mv,d,n))
    print('%-9s %-22s %-22s'%(name,row[0],row[1]))
