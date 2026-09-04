"""Paired strength/regression gate between arbitrary Deep Blue fast-search variants.

Each opening is played twice with colours reversed. This is a regression/strength
signal, not a precise Elo estimator.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepblue.fastcore import from_fen

OPENINGS = [
    ("start", chess.STARTING_FEN),
    ("open ruy", "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
    ("sicilian", "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"),
    ("french", "rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2"),
    ("caro-kann", "rnbqkbnr/pp1ppppp/2p5/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2"),
    ("queens gambit", "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2"),
    ("kings indian", "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3"),
    ("english", "rnbqkbnr/pppp1ppp/8/4p3/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 2"),
    ("closed centre", "r1bqkb1r/pp1n1ppp/2p1pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 6"),
    ("symmetrical", "rnbqkbnr/pp2pppp/8/2pp4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3"),
]

PLY_CAP = 200


def load_variant(name: str):
    mod = importlib.import_module(f"deepblue.{name}")
    suffix = name.removeprefix("fastsearch")
    cls = getattr(mod, "FastEngine" + suffix)
    mod.warm_up()
    return mod, cls


def make(spec):
    mod, cls = spec
    return cls()


def play(white_spec, black_spec, fen: str, move_ms: int, hard_mult: float):
    engines = {chess.WHITE: make(white_spec), chess.BLACK: make(black_spec)}
    board = chess.Board(fen)
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < PLY_CAP:
        engine = engines[board.turn]
        position = from_fen(board.fen())
        if hasattr(engine, "record_game_position"):
            engine.record_game_position(position)
        result = engine.search(position, move_ms, move_ms * hard_mult)
        uci = result[0]
        if uci is None:
            return "problem", "no move"
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            return "problem", f"illegal {uci} at {board.fen()}"
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return "draw", "ply cap"
    if outcome.winner is None:
        return "draw", outcome.termination.name.lower()
    return ("white" if outcome.winner == chess.WHITE else "black"), outcome.termination.name.lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--move-ms", type=int, default=60)
    ap.add_argument("--hard-mult", type=float, default=1.4)
    ap.add_argument("--openings", type=int, default=8)
    args = ap.parse_args()

    print(f"warming {args.baseline} ...", flush=True)
    base = load_variant(args.baseline)
    print(f"warming {args.candidate} ...", flush=True)
    cand = load_variant(args.candidate)

    wins = draws = losses = problems = 0
    terminations: dict[str, int] = {}

    for name, fen in OPENINGS[:args.openings]:
        for cand_white in (True, False):
            white = cand if cand_white else base
            black = base if cand_white else cand
            result, detail = play(white, black, fen, args.move_ms, args.hard_mult)
            terminations[detail] = terminations.get(detail, 0) + 1
            if result == "problem":
                problems += 1
                mark = "!"
            elif result == "draw":
                draws += 1
                mark = "="
            elif (result == "white") == cand_white:
                wins += 1
                mark = "+"
            else:
                losses += 1
                mark = "-"
            print(f"  {name:<16} {args.candidate} as {'W' if cand_white else 'B'}: {mark} ({detail})")

    games = wins + draws + losses
    score = (wins + 0.5 * draws) / games if games else 0.0
    print(f"\n{args.openings} openings x 2 colours = {games} completed games, {args.move_ms} ms/move")
    print(f"{args.candidate} vs {args.baseline}: +{wins} ={draws} -{losses}   paired score {score:.1%}")
    print("terminations: " + ", ".join(f"{k} {v}" for k, v in sorted(terminations.items())))
    print(f"problems (illegal/no move): {problems}")
    print("Too few games for a precise Elo claim; this is a promotion gate.")


if __name__ == "__main__":
    main()
