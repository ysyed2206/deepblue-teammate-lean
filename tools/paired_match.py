"""Paired match between two S1 search variants over a varied opening suite.

Each opening is played twice with colours reversed, so an opening that simply
favours White cannot bias the result. Playing hundreds of games from
STARTING_FEN with deterministic engines produces the same two games over and
over; the confidence interval then looks scientific while the samples are
almost perfectly correlated.

Engines are separate objects with separate tables and separate game history,
so no state is shared between the two sides.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepblue import fastsearch as FS  # noqa: E402
from deepblue import fastsearch1 as FS1  # noqa: E402
from deepblue.fastcore import from_fen  # noqa: E402

# Deliberately varied and roughly balanced: open, closed, semi-open, gambit
# and symmetrical structures, all reached by normal play.
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


def make(kind: str):
    if kind == "search0":
        FS.warm_up()
        engine = FS.FastEngine()
        return engine, None
    FS1.warm_up()
    engine = FS1.FastEngine1()
    return engine, "record"


def play(white_kind: str, black_kind: str, fen: str, move_ms: int) -> tuple[str, str]:
    engines = {}
    for colour, kind in ((chess.WHITE, white_kind), (chess.BLACK, black_kind)):
        engines[colour] = make(kind)
    board = chess.Board(fen)
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < PLY_CAP:
        engine, records = engines[board.turn]
        position = from_fen(board.fen())
        if records:
            engine.record_game_position(position)
        uci, _, _, _, _ = engine.search(position, move_ms, move_ms * 1.3)
        if uci is None:
            break
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            return "illegal", f"{uci} at {board.fen()}"
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return "adjudicated", "ply cap"
    if outcome.winner is None:
        return "draw", outcome.termination.name.lower()
    return ("white" if outcome.winner == chess.WHITE else "black"), outcome.termination.name.lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--move-ms", type=int, default=60)
    parser.add_argument("--openings", type=int, default=10)
    arguments = parser.parse_args()

    wins = draws = losses = 0
    problems: list[str] = []
    terminations: dict[str, int] = {}

    for name, fen in OPENINGS[: arguments.openings]:
        for search1_is_white in (True, False):
            white = "search1" if search1_is_white else "search0"
            black = "search0" if search1_is_white else "search1"
            result, detail = play(white, black, fen, arguments.move_ms)
            terminations[detail] = terminations.get(detail, 0) + 1
            if result == "illegal":
                problems.append(f"illegal move: {detail}")
                continue
            if result in ("draw", "adjudicated"):
                draws += 1
                mark = "="
            elif (result == "white") == search1_is_white:
                wins += 1
                mark = "+"
            else:
                losses += 1
                mark = "-"
            side = "W" if search1_is_white else "B"
            print(f"  {name:<16} search1 as {side}: {mark} ({detail})")

    games = wins + draws + losses
    score = (wins + draws / 2) / games if games else 0.0
    print(f"\n{arguments.openings} openings x 2 colours = {games} games, "
          f"{arguments.move_ms} ms per move")
    print(f"search1 vs search0: +{wins} ={draws} -{losses}   paired score {score:.1%}")
    print("terminations: " + ", ".join(f"{k} {v}" for k, v in sorted(terminations.items())))
    print(f"illegal moves / crashes / flags: {len(problems)}")
    for problem in problems:
        print(f"  {problem}")
    print("\nToo few games for an Elo claim. This is a regression gate, not a rating.")


if __name__ == "__main__":
    main()
