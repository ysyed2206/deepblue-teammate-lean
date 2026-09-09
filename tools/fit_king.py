"""Ask whether a better king-attack term reduces eval error against Stockfish.

Four models, each fitted the same way on the same 120k quiet positions with
the same 80/20 split, so the differences are attributable:

  A  current eval, untouched
  B  current eval with per-term scales tuned            (no new information)
  C  B, but the danger+shield terms replaced by a linear fit on the raw
     attack features we ALREADY compute                 (better shape, same inputs)
  D  C plus the inputs we do not compute at all: enemy pawns attacking and
     occupying the king zone, open files bearing on the king, safe checks

C-vs-B isolates "our term has the wrong shape". D-vs-C isolates "our term is
missing inputs". Only D-vs-C justifies writing new evaluation code.
"""
from __future__ import annotations

import numpy as np

K = 1.0 / 400.0
OLD = ["n_attackers", "wsum", "n_squares"]          # information we already have
NEW = ["pawn_att", "pawn_wedge", "open_file", "safe_check"]


def wp(cp):
    return 1.0 / (1.0 + np.power(10.0, -K * cp))


def fit(Xtr, ttr, Xva, tva, iters=6000, lr=0.02, pin0=True):
    a = np.zeros(Xtr.shape[1])
    a[0] = 1.0
    ln10K = np.log(10.0) * K
    for _ in range(iters):
        p = wp(Xtr @ a)
        g = (Xtr * (2.0 * (p - ttr) * p * (1 - p) * ln10K)[:, None]).mean(axis=0)
        if pin0:
            g[0] = 0.0
        a -= lr * g / (np.abs(g).max() + 1e-12)
    return a, float(np.mean((wp(Xva @ a) - tva) ** 2))


def main() -> None:
    d = np.load("scratch_terms.npz", allow_pickle=True)
    kd = np.load("scratch_king.npz", allow_pickle=True)
    keep = kd["keep"]
    X, y = d["X"][keep], d["y"][keep]
    names = [str(s) for s in d["names"]]
    F, fnames = kd["F"], [str(s) for s in kd["names"]]

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    X, y, F = X[idx], y[idx], F[idx]
    n = int(0.8 * len(y))
    t = wp(y)

    def split(M):
        return M[:n], M[n:]

    ttr, tva = t[:n], t[n:]

    # A: everything at 1.0
    a0 = np.ones(X.shape[1])
    mA = float(np.mean((wp(X[n:] @ a0) - tva) ** 2))

    # B: tune the nine scales
    Btr, Bva = split(X)
    aB, mB = fit(Btr, ttr, Bva, tva)

    # keep = base + the terms that are not king-related
    other = [j for j, nm in enumerate(names) if nm not in ("shield", "danger")]
    Xo = X[:, other]

    def build(extra_cols):
        M = np.hstack([Xo, extra_cols])
        return split(M)

    io = [fnames.index(c) for c in OLD]
    iN = [fnames.index(c) for c in NEW]

    # C: old information, free shape (linear + squared + count*weight)
    wsum = F[:, fnames.index("wsum")]
    cnt = F[:, fnames.index("n_attackers")]
    quad = np.column_stack([np.sign(wsum) * (wsum ** 2) / 512.0, cnt * wsum / 10.0])
    Ctr, Cva = build(np.hstack([F[:, io], quad]))
    aC, mC = fit(Ctr, ttr, Cva, tva)

    # D: add the missing inputs
    Dtr, Dva = build(np.hstack([F[:, io], quad, F[:, iN]]))
    aD, mD = fit(Dtr, ttr, Dva, tva)

    print("validation Texel MSE (lower is better)")
    print("  A  current eval                          %.6f" % mA)
    print("  B  + per-term scales tuned               %.6f   (%+.6f vs A)" % (mB, mB - mA))
    print("  C  + king term reshaped, same inputs     %.6f   (%+.6f vs B)" % (mC, mC - mB))
    print("  D  + missing king inputs                 %.6f   (%+.6f vs C)" % (mD, mD - mC))

    cols = [names[j] for j in other] + OLD + ["wsum^2/512", "cnt*wsum/10"] + NEW
    print("\n  model D weights:")
    for c, v in zip(cols, aD):
        print("    %-14s %+8.3f" % (c, v))

    print("\n  dropping each NEW input from D (val MSE change):")
    for c in NEW:
        j = cols.index(c)
        b = aD.copy(); b[j] = 0.0
        print("    %-14s %+.6f" % (c, float(np.mean((wp(Dva @ b) - tva) ** 2)) - mD))


if __name__ == "__main__":
    main()
