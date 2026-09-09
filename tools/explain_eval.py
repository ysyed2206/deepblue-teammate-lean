"""Decompose evaluate() into its terms for a position, or for a move.

WHY THIS EXISTS (2026-09-07). Every blunder investigation in this project has
recorded WHAT the engine scored and never WHY. "It plays g4 at +9" does not
say which term produced the +9, so every fix that followed was aimed at a
mechanism chosen by guesswork -- two of which turned out, on measurement, not
to be the cause at all.

This prints the same terms evaluate() sums, in the same order and the same
White-relative convention, so the number at the bottom reconciles exactly.
With a move, it prints the breakdown before and after and the delta per term,
which names the term responsible for liking a move.

    python tools/explain_eval.py "<fen>"
    python tools/explain_eval.py "<fen>" g2g4
"""
from __future__ import annotations

import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepblue.fastcore import from_fen                       # noqa: E402
from deepblue.eval_terms import (                            # noqa: E402
    bishop_pair_white_relative, game_phase, king_attack_danger_white_relative,
    king_safety_white_relative, knight_outposts_white_relative,
    mobility_white_relative, passed_pawns_white_relative,
    pawn_structure_white_relative, rook_files_white_relative,
)
from deepblue.fastsearch118 import (                         # noqa: E402
    EG_TABLE, MG_TABLE, PHASE_TABLE, base_evaluate, evaluate,
)

_TOTAL_PHASE = 24


def terms(fen: str) -> tuple[dict[str, int], int]:
    """Every term evaluate() sums, all normalised to WHITE-relative.

    base_evaluate is side-to-move relative while the rest are White-relative.
    Across a move the side to move flips, so leaving base in its own
    convention silently subtracts one frame from the other -- which is exactly
    the class of sign bug that made the first texel run fit against noise.
    """
    bb, occ, mail, st = from_fen(fen)
    phase = game_phase(bb, PHASE_TABLE, _TOTAL_PHASE)
    base_stm = int(base_evaluate(bb, st, MG_TABLE, EG_TABLE, PHASE_TABLE))
    out = {
        "material+PST": base_stm if st[0] == 0 else -base_stm,
        "passed pawns": int(passed_pawns_white_relative(bb)),
        "king shield": int(king_safety_white_relative(bb, phase, _TOTAL_PHASE)),
        "king danger": int(king_attack_danger_white_relative(bb, phase, _TOTAL_PHASE)),
        "mobility": int(mobility_white_relative(bb)),
        "bishop pair": int(bishop_pair_white_relative(bb)),
        "pawn structure": int(pawn_structure_white_relative(bb)),
        "rook files": int(rook_files_white_relative(bb)),
        "knight outposts": int(knight_outposts_white_relative(bb)),
    }
    return out, int(evaluate(bb, st, MG_TABLE, EG_TABLE, PHASE_TABLE))


def show(fen: str, move_uci: str | None) -> None:
    before, total_before = terms(fen)
    if move_uci is None:
        print(f"{fen}\n")
        for name, value in before.items():
            print(f"  {name:<16} {value:+7d}")
        print(f"  {'-' * 24}\n  {'TOTAL (stm)':<16} {total_before:+7d}")
        return

    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    san = board.san(move)
    board.push(move)
    after, total_after = terms(board.fen())

    print(f"{fen}\nmove: {san}\n")
    print(f"  {'term':<16} {'before':>8} {'after':>8} {'delta':>8}")
    for name in before:
        # Both sides are White-relative, so the delta is directly comparable
        # even though the side to move has changed.
        d = after[name] - before[name]
        flag = "   <<<" if abs(d) >= 25 else ""
        print(f"  {name:<16} {before[name]:+8d} {after[name]:+8d} {d:+8d}{flag}")
    print(f"  {'-' * 42}")
    wb, wa = sum(before.values()), sum(after.values())
    print(f"  {'WHITE-relative':<16} {wb:+8d} {wa:+8d} {wa - wb:+8d}")
    print(f"\n  white-relative swing from this move: "
          f"{sum(after.values()) - sum(before.values()):+d} cp")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    show(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)