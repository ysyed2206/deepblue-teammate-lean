"""Multi-replay equal-time promotion gate, one warm-up cost for N replays.

Companion to `paired_fast_variants.py` (unmodified, reused directly -- same
`OPENINGS`/`load_variant`/`play`). That tool is the real promotion gate
(equal wall-clock time), but each process launch pays the engine's Numba
JIT-compile cost fresh (`cache=False` on the recursive negamax/quiescence/
search_root functions -- recursive @njit functions cannot reliably use
Numba's disk cache, so there is no way to avoid this cost once per process).
For a fast variant with heavy compiled functions this can be 10+ minutes,
which makes running the multiple independent replays this project's own
history says are necessary before trusting an equal-time result (see
AC-001 in DEEPBLUE_AUDIT_CORRECTIONS.md: identical 16-game replays have been
observed to swing from 56.2% to 50.0%, once flipping sign entirely) far more
expensive than it needs to be if done as N separate process launches.

This script warms both engines up exactly once, then plays N independent
replay rounds (fresh games each round, same openings/move-ms every round --
the noise source is OS/CPU scheduling jitter affecting which iterative-
deepening depth a move completes at, not a random seed) and reports both
the per-round score and the pooled score across all rounds, exactly as this
project's own fastsearch15 promotion evidence was assembled (52 games
pooled across 3 independent replays).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paired_fast_variants import OPENINGS, load_variant, play  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run N independent equal-time replay rounds against one warm-up, "
        "reporting per-round and pooled scores (this project's own promotion "
        "evidence standard needs multiple replays, not one lucky/unlucky run)."
    )
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--move-ms", type=int, default=60)
    ap.add_argument("--hard-mult", type=float, default=1.4)
    ap.add_argument("--openings", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    print(f"warming {args.baseline} ...", flush=True)
    base = load_variant(args.baseline)
    print(f"warming {args.candidate} ...", flush=True)
    cand = load_variant(args.candidate)

    pooled_wins = pooled_draws = pooled_losses = pooled_problems = 0
    round_scores: list[float] = []

    for round_index in range(1, args.rounds + 1):
        started = time.monotonic()
        wins = draws = losses = problems = 0
        terminations: dict[str, int] = {}
        for name, fen in OPENINGS[: args.openings]:
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
                print(f"  round {round_index} {name:<16} {args.candidate} as {'W' if cand_white else 'B'}: {mark} ({detail})", flush=True)
        games = wins + draws + losses
        score = (wins + 0.5 * draws) / games if games else 0.0
        round_scores.append(score)
        pooled_wins += wins
        pooled_draws += draws
        pooled_losses += losses
        pooled_problems += problems
        elapsed_s = time.monotonic() - started
        print(
            f"round {round_index}: +{wins} ={draws} -{losses}   score {score:.1%}   "
            f"({games} games, {elapsed_s:.1f}s, terminations: "
            + ", ".join(f'{k} {v}' for k, v in sorted(terminations.items()))
            + ")\n",
            flush=True,
        )

    pooled_games = pooled_wins + pooled_draws + pooled_losses
    pooled_score = (pooled_wins + 0.5 * pooled_draws) / pooled_games if pooled_games else 0.0
    spread = (max(round_scores) - min(round_scores)) if round_scores else 0.0
    print(f"=== {args.candidate} vs {args.baseline}: {args.rounds} independent replays, {pooled_games} games pooled ===")
    print(f"per-round scores: {', '.join(f'{s:.1%}' for s in round_scores)}  (spread: {spread:.1%})")
    print(f"pooled: +{pooled_wins} ={pooled_draws} -{pooled_losses}   pooled score {pooled_score:.1%}")
    if pooled_problems:
        print(f"problems (illegal/no move) across all rounds: {pooled_problems}")
    if any(s < 0.5 for s in round_scores) and any(s > 0.5 for s in round_scores):
        print("NOTE: sign flip across replays (some rounds positive, some negative) -- "
              "consistent with this project's documented AC-001 timing-jitter noise. "
              "Treat the pooled score, not any single round, as the signal.")
    print("Too few games for a precise Elo claim even pooled; use alongside fixed-depth "
          "structural evidence and correctness gates, not in isolation.")


if __name__ == "__main__":
    main()
