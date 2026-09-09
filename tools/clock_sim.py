"""Replay a real game against a real simulated clock and show where time goes.

WHY. We know 32% of the clock goes unspent across 70 games. Two very different
explanations fit that fact and they imply opposite actions:

  (a) the search stops early and wastes budget it was given   -> fix the search
  (b) the allocator reserves for moves the game never reaches -> not a defect

Nothing measured so far distinguishes them, and a mechanism was asserted for
(a) on a statistic that turned out to include 274 forced moves. This measures
it instead: it drives the engine through a real game with a real ticking clock,
and for every move records what it was ALLOWED, what it SPENT, and why it
stopped -- so the surplus can be attributed rather than guessed at.

    python tools/clock_sim.py <module> <pgn> [start_ms]
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepblue.fastcore import from_fen          # noqa: E402
from deepblue.time_manager import allocate      # noqa: E402

INCREMENT_MS = 500.0


def main() -> None:
    module_name = sys.argv[1] if len(sys.argv) > 1 else "fastsearch118"
    pgn_path = sys.argv[2]
    clock_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 120_000.0

    module = importlib.import_module(f"deepblue.{module_name}")
    engine = getattr(module, "FastEngine" + module_name.replace("fastsearch", ""))()
    engine.search(from_fen(chess.STARTING_FEN), 200, 400)      # JIT only

    game = chess.pgn.read_game(open(pgn_path, encoding="utf-8"))
    us = (chess.WHITE if game.headers.get("White", "").lower().startswith("yumo")
          else chess.BLACK)
    board = game.board()

    print(f"{module_name} on {Path(pgn_path).name}, starting clock {clock_ms/1000:.1f}s\n")
    print(f"{'move':>5} {'soft':>7} {'spent':>7} {'used%':>6} {'depth':>6} {'clock':>8}  stop")
    moves_played = 0
    total_soft = total_spent = 0.0
    reasons: dict[str, int] = {}
    for mv in game.mainline_moves():
        if board.turn == us:
            soft, hard = allocate(int(clock_ms), int(INCREMENT_MS), moves_played)
            legal = board.legal_moves.count()
            if legal == 1:
                spent, depth, reason = 0.0, 0, "forced"
            else:
                _, score, depth, _, spent = engine.search(from_fen(board.fen()), soft, hard)
                if spent >= soft * 0.97:
                    reason = "soft budget"
                elif abs(score) > 29_000 - 1_000:
                    reason = "mate found"
                else:
                    reason = "predicted overrun"
            clock_ms += INCREMENT_MS - spent
            total_soft += soft
            total_spent += spent
            reasons[reason] = reasons.get(reason, 0) + 1
            pct = 100.0 * spent / soft if soft > 0 else 0.0
            print(f"{board.fullmove_number:>5} {soft:>7.0f} {spent:>7.0f} {pct:>5.0f}% "
                  f"{depth:>6} {clock_ms/1000:>7.1f}s  {reason}")
            moves_played += 1
        board.push(mv)

    print(f"\nallocated {total_soft/1000:>7.1f}s   spent {total_spent/1000:>7.1f}s "
          f"({100*total_spent/total_soft:.0f}% of allocation)")
    print(f"clock left at end: {clock_ms/1000:.1f}s of 120s "
          f"({100*clock_ms/120000:.0f}% unspent)")
    print("\nwhy the search stopped:")
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<20} {count:>3}")


if __name__ == "__main__":
    main()
