"""Run many arena games at once, with a confidence interval on the result.

This is a thin wrapper around harness.referee.play_match. It adds no game
logic - the referee, the clock and the legality rules are the harness's, so
results stay comparable with `make arena`.

Two things it adds that matter:

*   Concurrency. Each game is a pair of single-core agent processes and only
    one side thinks at a time, so one concurrent game costs about one core.
*   A confidence interval. A raw score over 20 games is close to meaningless;
    reporting the interval makes it obvious when a result is noise.
"""

from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.referee import FAILED_TERMINATIONS, play_match  # noqa: E402
from harness.rules import PLY_CAP  # noqa: E402
from harness.sandbox import local  # noqa: E402


def play_one(
    agent: str, opponent: str, plays_white: bool, base_ms: int, increment_ms: int, ply_cap: int
) -> tuple[str, str, bool]:
    white, black = (agent, opponent) if plays_white else (opponent, agent)
    outcome = play_match(
        local(Path(white)), local(Path(black)), base_ms, increment_ms, ply_cap=ply_cap
    )
    return outcome.result, outcome.termination, plays_white


def elo_from_score(score: float) -> float:
    if score <= 0.0:
        return -800.0
    if score >= 1.0:
        return 800.0
    return -400.0 * math.log10(1.0 / score - 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel arena with a confidence interval.")
    parser.add_argument("--agent", default=".")
    parser.add_argument("--opponent", default="baselines/greedy")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--base-ms", type=int, default=10_000)
    parser.add_argument("--increment-ms", type=int, default=100)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    arguments = parser.parse_args()

    agent = str(Path(arguments.agent).resolve())
    opponent = str(Path(arguments.opponent).resolve())

    wins = draws = losses = 0
    terminations: dict[str, int] = {}
    failures: list[str] = []

    with ProcessPoolExecutor(max_workers=arguments.concurrency) as pool:
        futures = [
            pool.submit(
                play_one,
                agent,
                opponent,
                game % 2 == 0,
                arguments.base_ms,
                arguments.increment_ms,
                arguments.ply_cap,
            )
            for game in range(arguments.games)
        ]
        for done in as_completed(futures):
            result, termination, plays_white = done.result()
            terminations[termination] = terminations.get(termination, 0) + 1
            if result in ("draw", "void"):
                draws += 1
                won = None
            elif (result == "white") == plays_white:
                wins += 1
                won = True
            else:
                losses += 1
                won = False
            # A failing termination belongs to whoever lost the game. Counting
            # the opponent's flag as our failure made a 24-0 win look like 21
            # faults; attribute it to the side that actually lost.
            if termination in FAILED_TERMINATIONS and won is False:
                failures.append(termination)
            elif termination in FAILED_TERMINATIONS and won is None:
                failures.append(f"{termination} (void)")

    games = arguments.games
    score = (wins + draws / 2) / games
    # Standard error of the mean game score, treating a draw as a half point.
    variance = (wins * (1 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * score**2) / games
    stderr = math.sqrt(variance / games) if games > 1 else 0.5
    low, high = max(0.0, score - 1.96 * stderr), min(1.0, score + 1.96 * stderr)

    print(f"\n{arguments.agent}")
    print(f"  vs {arguments.opponent}")
    print(f"  {games} games at {arguments.base_ms} ms + {arguments.increment_ms} ms, "
          f"concurrency {arguments.concurrency}")
    print(f"  +{wins} ={draws} -{losses}   score {score:.1%}  95% CI [{low:.1%}, {high:.1%}]")
    print(f"  elo  {elo_from_score(score):+.0f}   95% CI [{elo_from_score(low):+.0f}, "
          f"{elo_from_score(high):+.0f}]")
    print("  terminations: " + ", ".join(f"{k} {v}" for k, v in sorted(terminations.items())))
    if failures:
        print(f"  *** OUR FAILURES: {len(failures)} -> {sorted(set(failures))}")
        raise SystemExit(1)
    print("  our failures: none")


if __name__ == "__main__":
    main()
