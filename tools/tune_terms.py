"""Fit per-term scales against deep Stockfish scores (Texel objective).

Our eval is base_evaluate plus eight terms whose weights were hand-picked one
at a time, each judged by a self-play match too small to resolve it. This asks
a different question -- given the terms we already compute, what RELATIVE
weighting best reproduces Stockfish on 120k quiet positions?

base is pinned at 1.0. Scaling every term together is a global gain, a
separate question from the relative balance, and pinning base keeps the fit
interpretable as "worth this much material".

Train/validation split guards against reading noise as signal.
"""
from __future__ import annotations

import argparse

import numpy as np

K = 1.0 / 400.0


def wp(cp):
    return 1.0 / (1.0 + np.power(10.0, -K * cp))


def mse(a, X, ytarget):
    return float(np.mean((wp(X @ a) - ytarget) ** 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="scratch_terms.npz")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2.0)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X, y, names = d["X"], d["y"], [str(s) for s in d["names"]]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    ntr = int(0.8 * len(y))
    Xtr, ytr, Xva, yva = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    ttr, tva = wp(ytr), wp(yva)

    a = np.ones(X.shape[1])
    print("baseline (all scales 1.0)   train %.6f   val %.6f"
          % (mse(a, Xtr, ttr), mse(a, Xva, tva)))

    ln10K = np.log(10.0) * K
    for it in range(args.iters):
        s = Xtr @ a
        p = wp(s)
        # d/da  mean (p - t)^2  =  mean 2(p-t) * p(1-p)*ln10*K * X
        g = (Xtr * (2.0 * (p - ttr) * p * (1.0 - p) * ln10K)[:, None]).mean(axis=0)
        g[0] = 0.0                      # base pinned
        a -= args.lr * g / (np.abs(g).max() + 1e-12) * 0.01
        a[1:] = np.clip(a[1:], 0.0, 6.0)

    print("tuned                       train %.6f   val %.6f"
          % (mse(a, Xtr, ttr), mse(a, Xva, tva)))
    print("\n  %-12s %6s   %s" % ("term", "scale", "effect"))
    for j, nm in enumerate(names):
        if j == 0:
            print("  %-12s %6.2f   (pinned)" % (nm, a[j]))
            continue
        b = a.copy(); b[j] = 1.0
        delta = mse(b, Xva, tva) - mse(a, Xva, tva)
        print("  %-12s %6.2f   val MSE worsens by %+.6f if left at 1.0" % (nm, a[j], delta))

    # what each term is worth on its own, holding the rest tuned
    print("\n  dropping a term entirely (scale 0) from the tuned set:")
    for j, nm in enumerate(names):
        if j == 0:
            continue
        b = a.copy(); b[j] = 0.0
        print("  %-12s val MSE %+.6f" % (nm, mse(b, Xva, tva) - mse(a, Xva, tva)))

    np.save("scratch_term_scales.npy", a)


if __name__ == "__main__":
    main()
