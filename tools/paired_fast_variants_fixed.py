"""Deterministic FIXED-DEPTH paired-game gate between Deep Blue fast-search variants.

Companion to `tools/paired_fast_variants.py`, which is left untouched and
remains the real promotion gate (equal wall-clock time, the condition the
engine will actually be judged under). This tool exists for a narrower job:
screening candidates during development without the timing-jitter noise
documented in `DEEPBLUE_AUDIT_CORRECTIONS.md` AC-001.

Background (AC-001, read the full section before touching this file): every
FastEngineN.search() runs iterative deepening and decides whether to start
the next depth using a single-sample iteration-time-growth *prediction*
compared against a wall-clock soft budget. That prediction is a hard
threshold with no margin, so ordinary OS/CPU scheduling jitter can flip which
depth a move stops at -- on an IDENTICAL 16-game replay with zero code
changes, AC-001 measured paired scores swinging from 56.2% to 50.0%, and in
one case flipping sign (56.2% positive to 43.8% negative). That noise
contaminates full games, not just single-position node counts, because the
move actually played at a jitter-flipped depth diverges the rest of the game
tree.

How this tool removes that noise, without touching any engine source file
-----------------------------------------------------------------------
Every FastEngineN.search(position, soft_ms, hard_ms, max_depth=64) already
ships a `max_depth` parameter (verified present with the same signature in
fastsearch4/18/20/21/22 before writing this tool). We drive the search with:

  * `max_depth = <fixed depth requested>`  -- the actual mechanism that
    controls how deep the search goes.
  * `soft_ms`  = an astronomically large budget (1e9 ms, ~11.5 days). Both of
    the wall-clock-based early-stop checks in the iterative-deepening loop
    (`elapsed >= soft_ms` and the AC-001 predictive-growth check
    `elapsed + iteration_ms*growth > soft_ms`) compare against this value, so
    with any realistic per-move search time neither can ever fire. The loop
    is therefore bounded purely by `range(1, max_depth + 1)` -- i.e. by
    `max_depth` -- not by a timing decision. This is exactly why fixed-depth
    search is reproducible: AC-001 already proved (same section, "Node
    counts at depth 5 are bit-identical between run 0 and run 2") that the
    search tree itself is deterministic once the *decision of how deep to
    go* is taken out of the wall clock's hands.
  * `hard_ms = --safety-ms` (default 120_000 ms / 2 minutes) -- a real,
    finite value, but purely a SAFETY VALVE against a pathological position
    (see "Safety valve, not a real constraint" below), not something meant
    to bind in normal use. It is the existing `hard_ms` parameter the engine
    already ships and already always accepts; nothing new is added to the
    engine to support this.

No change to any deepblue/*.py file was made or is needed for this mode.

Fixed-NODE mode: investigated, not implemented (documented limitation)
------------------------------------------------------------------------
A fixed-*node* budget (stop once N nodes have been searched, independent of
depth or wall clock) was considered as a secondary mode. It is not offered
here. Reason: the node counter (`self.counters[NODES]`) is only read back
*after* each depth's `search_root` call returns inside FastEngineN.search();
there is no node-budget check anywhere inside the Numba-jitted `negamax` /
`quiescence` move loops themselves, which is where a node-accurate stop would
have to live to cut a search off mid-depth at an exact node count. Adding
that would mean adding a new stop condition to the engine's own search loop
-- i.e. modifying deepblue/fastsearch*.py's actual competition search logic,
which is explicitly out of scope for a test-tooling task ("do not modify the
competition search logic merely for the test"). A *between-depths* node cap
(stop iterative deepening once the cumulative node count after a completed
depth exceeds a threshold, using the existing max_depth loop structure) would
be possible without engine changes, but it degenerates back into a
depth-granularity decision -- it cannot land mid-depth at an exact node
count, so it would not actually test "the same amount of search work" any
more precisely than fixed-depth already does, while adding an extra
non-determinism source (node counts differ across positions/moves, so the
resulting depth reached would itself have to be discovered by trial, per
move). Fixed-depth is therefore the practical deterministic mode available
without engine changes; true fixed-node determinism would need a genuine,
explicitly-approved engine change and is out of scope for this tool.

Safety valve, not a real constraint
------------------------------------
Fixed depth does not map cleanly onto elapsed wall-clock time: a quiet
endgame at depth 8 might take milliseconds, a wild tactical middlegame at the
same depth could take 10x+ longer. `--safety-ms` (fed as the engine's
existing `hard_ms` parameter) exists purely so a pathological position can't
hang a batch run forever; it is set generously (default 2 minutes/move) and
is not expected to bind at the depths this tool is meant to be used at (5-8).
If it ever does bind, the affected move is flagged in the "safety-cap
triggers" count in the summary -- treat any nonzero count there as a signal
that either the requested depth is too deep for batch use or the default
needs raising, not as a silent partial-depth result.

Determinism verification: run this tool twice on the same arguments with no
code changes in between and diff stdout; per-game output includes a short
digest of the exact move sequence so a diff shows determinism at the
move-by-move level, not just the final score.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from deepblue.fastcore import from_fen

from paired_fast_variants import OPENINGS, load_variant, make  # noqa: E402  (reuses the existing tool's data/loader unmodified)

PLY_CAP = 200

# Both wall-clock knobs are fed to the engine's existing, unmodified
# search(position, soft_ms, hard_ms, max_depth) signature.
#
# SOFT_MS_DISABLE: large enough that neither of the two time-based early-stop
# checks inside the iterative-deepening loop (see module docstring) can ever
# fire for any realistic per-iteration search time, so the loop is bounded
# purely by max_depth.
SOFT_MS_DISABLE = 1_000_000_000.0


def play_fixed(white_spec, black_spec, fen: str, depth: int, safety_ms: float):
    """Play one game with both sides searching to an exact fixed depth.

    Returns (result, detail, moves_uci, total_nodes, min_completed_depth,
    safety_triggers) where result is "white"/"black"/"draw"/"problem".
    """
    engines = {chess.WHITE: make(white_spec), chess.BLACK: make(black_spec)}
    board = chess.Board(fen)
    moves_uci: list[str] = []
    total_nodes = 0
    min_completed_depth = depth
    safety_triggers = 0

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < PLY_CAP:
        engine = engines[board.turn]
        position = from_fen(board.fen())
        if hasattr(engine, "record_game_position"):
            engine.record_game_position(position)
        result = engine.search(position, SOFT_MS_DISABLE, safety_ms, max_depth=depth)
        uci, score, completed, nodes, elapsed_ms = result
        total_nodes += nodes
        if completed < min_completed_depth:
            min_completed_depth = completed
        # completed < depth with no mate-score justification means the
        # hard_ms safety timer fired mid-search -- the safety valve bound,
        # not the requested fixed depth. Flag it; do not silently accept it
        # as if it were a clean fixed-depth result.
        from deepblue.constants import MATE_THRESHOLD  # local import: avoid polluting the module namespace

        if completed < depth and abs(score) <= MATE_THRESHOLD:
            safety_triggers += 1
        if uci is None:
            return "problem", "no move", moves_uci, total_nodes, min_completed_depth, safety_triggers
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            return "problem", f"illegal {uci} at {board.fen()}", moves_uci, total_nodes, min_completed_depth, safety_triggers
        moves_uci.append(uci)
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return "draw", "ply cap", moves_uci, total_nodes, min_completed_depth, safety_triggers
    if outcome.winner is None:
        return "draw", outcome.termination.name.lower(), moves_uci, total_nodes, min_completed_depth, safety_triggers
    return (
        "white" if outcome.winner == chess.WHITE else "black",
        outcome.termination.name.lower(),
        moves_uci,
        total_nodes,
        min_completed_depth,
        safety_triggers,
    )


def digest(moves_uci: list[str]) -> str:
    return hashlib.sha1(" ".join(moves_uci).encode()).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Deterministic fixed-depth paired-game gate (candidate screening; "
        "NOT a replacement for the equal-time arena / paired_fast_variants.py, which "
        "remains the final promotion gate)."
    )
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--depth", type=int, default=6, help="exact search depth for both sides, every move")
    ap.add_argument("--safety-ms", type=float, default=120_000.0, help="wall-clock safety valve per move (hard_ms); should essentially never bind")
    ap.add_argument("--openings", type=int, default=8)
    args = ap.parse_args()

    print(f"warming {args.baseline} ...", flush=True)
    base = load_variant(args.baseline)
    print(f"warming {args.candidate} ...", flush=True)
    cand = load_variant(args.candidate)

    wins = draws = losses = problems = 0
    terminations: dict[str, int] = {}
    safety_triggers_total = 0
    nodes_total = 0
    started = time.monotonic()

    for name, fen in OPENINGS[:args.openings]:
        for cand_white in (True, False):
            white = cand if cand_white else base
            black = base if cand_white else cand
            result, detail, moves_uci, nodes, min_depth, safety_triggers = play_fixed(
                white, black, fen, args.depth, args.safety_ms
            )
            nodes_total += nodes
            safety_triggers_total += safety_triggers
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
            flag = "" if min_depth >= args.depth else f" [SAFETY-CAP min_depth={min_depth}]"
            print(
                f"  {name:<16} {args.candidate} as {'W' if cand_white else 'B'}: {mark} "
                f"({detail}) {len(moves_uci)}ply digest={digest(moves_uci)}{flag}"
            )

    elapsed_s = time.monotonic() - started
    games = wins + draws + losses
    score = (wins + 0.5 * draws) / games if games else 0.0
    print(
        f"\n{args.openings} openings x 2 colours = {games} completed games, "
        f"fixed depth {args.depth}/side (safety valve {args.safety_ms:.0f} ms/move, wall time {elapsed_s:.1f}s total)"
    )
    print(f"{args.candidate} vs {args.baseline}: +{wins} ={draws} -{losses}   paired score {score:.1%}")
    print("terminations: " + ", ".join(f"{k} {v}" for k, v in sorted(terminations.items())))
    print(f"problems (illegal/no move): {problems}")
    print(f"safety-cap triggers (hard_ms fired before reaching depth {args.depth}): {safety_triggers_total}")
    print(f"total nodes searched (both sides, all games): {nodes_total:,}")
    print(
        "This is a deterministic development/candidate-screening signal (fixed depth, "
        "not fixed time) -- it isolates search-quality from the AC-001 timing-jitter "
        "noise, but it is NOT the promotion gate. The equal-time arena "
        "(tools/paired_fast_variants.py) remains the final confirmation before any "
        "promotion decision."
    )


if __name__ == "__main__":
    main()
