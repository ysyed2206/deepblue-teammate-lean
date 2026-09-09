"""Fit the hand-picked evaluation weights to engine-labelled positions.

Method (Texel tuning). For each position we have the engine's own
base_evaluate (piece-square tables and material, held FIXED here) plus a
vector of feature counts, and a target score from a depth 46-58 search. The
evaluation is linear in the weights:

    eval(position) = base(position) + w . features(position)

Both eval and target are squashed through a logistic before comparison,
because the difference between +300 and +400 centipawns matters far less to
the result of a game than the difference between 0 and +100:

    sigmoid(s) = 1 / (1 + 10 ** (-K * s / 400))
    E(w) = mean( (sigmoid(base + w.f) - sigmoid(target)) ** 2 )

K is fitted first with the current weights, then the weights are fitted by
coordinate descent -- the standard approach, and robust here because there
are only 14 parameters and the objective is smooth in each.

The tuned weights are a HYPOTHESIS, not a result. A lower fitting error means
the evaluation agrees more closely with a strong engine on these positions;
it does not by itself mean the engine plays better, because the search
interacts with the evaluation in ways this objective cannot see. Everything
here goes through the same 80-120 game paired match as every other candidate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from texel_features import FEATURE_NAMES  # noqa: E402


def sigmoid(scores: np.ndarray, k: float) -> np.ndarray:
    return 1.0 / (1.0 + np.power(10.0, -k * scores / 400.0))


def error(weights, features, base, target_prob, k) -> float:
    predicted = sigmoid(base + features.dot(weights), k)
    return float(np.mean((predicted - target_prob) ** 2))


def fit_k(weights, features, base, target) -> float:
    """Pick the logistic steepness that best fits the CURRENT weights."""
    best_k, best_e = 1.0, float("inf")
    for k in np.arange(0.05, 3.01, 0.05):
        e = error(weights, features, base, sigmoid(target, k), k)
        if e < best_e:
            best_k, best_e = float(k), e
    return best_k


def tune(weights, features, base, target, k, rounds=12) -> np.ndarray:
    target_prob = sigmoid(target, k)
    weights = weights.astype(float).copy()
    best = error(weights, features, base, target_prob, k)
    print(f"start error {best:.8f}", flush=True)
    steps = np.full(len(weights), 8.0)
    for round_index in range(rounds):
        improved = False
        for i in range(len(weights)):
            for direction in (1.0, -1.0):
                trial = weights.copy()
                trial[i] += direction * steps[i]
                e = error(trial, features, base, target_prob, k)
                if e < best:
                    best, weights, improved = e, trial, True
                    break
        if not improved:
            steps = steps / 2.0
            if steps.max() < 0.5:
                break
        print(f"  round {round_index + 1:2d}  error {best:.8f}  step {steps.max():.2f}", flush=True)
    return weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="C:/Users/uniqu/AppData/Local/Temp/texel_data.npz")
    parser.add_argument("--holdout", type=float, default=0.2)
    args = parser.parse_args()

    from deepblue import eval_terms as ET

    data = np.load(args.data)
    X, B, Y = data["X"], data["B"], data["Y"]

    # Hold out a slice never used for fitting, so a lower training error that
    # is only memorisation shows up as a worse holdout error.
    split = int(len(X) * (1.0 - args.holdout))
    Xtr, Btr, Ytr = X[:split], B[:split], Y[:split]
    Xho, Bho, Yho = X[split:], B[split:], Y[split:]
    print(f"{len(Xtr):,} training positions, {len(Xho):,} held out")

    current = np.array(
        list(ET.PASSED_BONUS[1:7])
        + [int(ET.SHIELD_PENALTY),
           int(ET.KNIGHT_MOBILITY_WEIGHT), int(ET.BISHOP_MOBILITY_WEIGHT),
           int(ET.ROOK_MOBILITY_WEIGHT), int(ET.QUEEN_MOBILITY_WEIGHT),
           int(ET.BISHOP_PAIR_BONUS), -int(ET.DOUBLED_PAWN_PENALTY),
           -int(ET.ISOLATED_PAWN_PENALTY)],
        dtype=float,
    )

    k = fit_k(current, Xtr, Btr, Ytr)
    print(f"fitted K = {k:.2f}")

    before_tr = error(current, Xtr, Btr, sigmoid(Ytr, k), k)
    before_ho = error(current, Xho, Bho, sigmoid(Yho, k), k)
    tuned = tune(current, Xtr, Btr, Ytr, k)
    after_tr = error(tuned, Xtr, Btr, sigmoid(Ytr, k), k)
    after_ho = error(tuned, Xho, Bho, sigmoid(Yho, k), k)

    print()
    print(f"{'weight':<16} {'current':>9} {'tuned':>9} {'change':>9}")
    for name, a, b in zip(FEATURE_NAMES, current, tuned):
        print(f"{name:<16} {a:>9.0f} {b:>9.0f} {b - a:>+9.0f}")
    print()
    print(f"training error {before_tr:.8f} -> {after_tr:.8f}  ({(1 - after_tr / before_tr) * 100:+.2f}%)")
    print(f"holdout  error {before_ho:.8f} -> {after_ho:.8f}  ({(1 - after_ho / before_ho) * 100:+.2f}%)")
    if after_ho >= before_ho:
        print("HOLDOUT DID NOT IMPROVE -- the fit is not generalising; do not ship these.")


if __name__ == "__main__":
    main()
