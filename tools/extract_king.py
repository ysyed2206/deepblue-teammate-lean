"""Extract raw king-attack features per position, White-relative.

The current danger term collapses everything into one number before we ever
see it: sum of per-piece weights, thresholded, squared, scaled. Fitting a
single scale to that cannot separate "the weights are wrong" from "the
threshold is wrong" from "the inputs are missing". This writes the raw
inputs so all three can be asked separately, and adds the inputs the term
does not currently have at all.

Written with python-chess for clarity -- this is offline analysis, nothing
here runs at match time. If a feature earns its place it gets a Numba
implementation and a match.

Features per side (then differenced White-minus-Black):
  n_attackers   distinct enemy N/B/R/Q attacking the 9-square king zone
  wsum          those pieces' current weights (what the term sums today)
  n_squares     distinct zone squares attacked by the enemy
  pawn_att      enemy PAWNS attacking the zone      <- absent from our eval
  pawn_wedge    enemy pawns standing on the zone     <- absent from our eval
  shield        own pawns on the king file+adjacent, ahead of the king
  open_file     enemy R/Q on a file with no own pawn, within 1 of king file
  safe_check    squares an enemy piece could check from, undefended by us
"""
from __future__ import annotations

import multiprocessing as mp
import sys

import chess
import numpy as np

W = {chess.KNIGHT: 81, chess.BISHOP: 81, chess.ROOK: 121, chess.QUEEN: 202}
NAMES = ["n_attackers", "wsum", "n_squares", "pawn_att", "pawn_wedge",
         "shield", "open_file", "safe_check"]


def side_features(b: chess.Board, us: bool):
    """Danger TO `us`, i.e. produced by the side that is not `us`."""
    them = not us
    ksq = b.king(us)
    if ksq is None:
        return [0.0] * len(NAMES)
    zone = chess.SquareSet(chess.BB_KING_ATTACKS[ksq]) | chess.SquareSet(chess.BB_SQUARES[ksq])

    n_att = wsum = 0
    squares = chess.SquareSet()
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in b.pieces(pt, them):
            hit = chess.SquareSet(b.attacks_mask(sq)) & zone
            if hit:
                n_att += 1
                wsum += W[pt]
                squares |= hit

    pawn_att = 0
    for sq in b.pieces(chess.PAWN, them):
        if chess.SquareSet(b.attacks_mask(sq)) & zone:
            pawn_att += 1
    pawn_wedge = len(chess.SquareSet(b.pieces_mask(chess.PAWN, them)) & zone)

    kf = chess.square_file(ksq)
    kr = chess.square_rank(ksq)
    shield = 0
    for df in (-1, 0, 1):
        f = kf + df
        if not 0 <= f <= 7:
            continue
        for sq in b.pieces(chess.PAWN, us):
            if chess.square_file(sq) == f:
                dr = chess.square_rank(sq) - kr
                if (dr > 0) == (us == chess.WHITE) and abs(dr) <= 3:
                    shield += 1
                    break

    open_file = 0
    for df in (-1, 0, 1):
        f = kf + df
        if not 0 <= f <= 7:
            continue
        own_pawn = any(chess.square_file(s) == f for s in b.pieces(chess.PAWN, us))
        if own_pawn:
            continue
        for pt in (chess.ROOK, chess.QUEEN):
            if any(chess.square_file(s) == f for s in b.pieces(pt, them)):
                open_file += 1
                break

    safe_check = 0
    for mv in b.legal_moves:
        pass
    # count enemy check squares cheaply: flip side to move, look at checks
    if b.turn == us:
        probe = b.copy(stack=False)
        try:
            probe.push(chess.Move.null())
        except Exception:
            probe = None
    else:
        probe = b
    if probe is not None:
        seen = set()
        for mv in probe.legal_moves:
            if mv.to_square in seen or not probe.gives_check(mv):
                continue
            pc = probe.piece_at(mv.from_square)
            if pc is None or pc.piece_type == chess.PAWN:
                continue
            if not probe.is_attacked_by(us, mv.to_square):
                seen.add(mv.to_square)
        safe_check = len(seen)

    return [float(n_att), float(wsum), float(len(squares)), float(pawn_att),
            float(pawn_wedge), float(shield), float(open_file), float(safe_check)]


def one(line: str):
    cp, fen = line.rstrip("\n").split("\t", 1)
    try:
        b = chess.Board(fen)
    except ValueError:
        return None
    w = side_features(b, chess.WHITE)      # danger to White
    k = side_features(b, chess.BLACK)      # danger to Black
    # White-relative: positive = good for White = Black is the one in danger
    return [k[i] - w[i] for i in range(len(NAMES))]


def main() -> None:
    src, out, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
    lines = open(src, encoding="utf-8").readlines()[:limit]
    with mp.Pool(6) as pool:
        rows = pool.map(one, lines, chunksize=400)
    keep = [i for i, r in enumerate(rows) if r is not None]
    F = np.array([rows[i] for i in keep], dtype=np.float64)
    np.savez_compressed(out, F=F, keep=np.array(keep), names=np.array(NAMES))
    print("wrote %s  shape %s" % (out, F.shape))
    for j, nm in enumerate(NAMES):
        print("  %-12s mean|v| %6.2f   nonzero %5.1f%%"
              % (nm, np.mean(np.abs(F[:, j])), 100 * np.mean(F[:, j] != 0)))


if __name__ == "__main__":
    mp.freeze_support()
    main()
