"""Correctness tests for numba_kernels.py, plus an opt-in benchmark.

Run with:
    .venv/Scripts/python.exe nnue_lab/teammate_pawnstar/tests/test_numba_kernels.py
    .venv/Scripts/python.exe nnue_lab/teammate_pawnstar/tests/test_numba_kernels.py --bench
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import features as feat  # noqa: E402
import numba_kernels as nk  # noqa: E402
import reference_eval as ref  # noqa: E402
import weights  # noqa: E402

DONOR_NET_PATH = Path(
    os.environ.get(
        "PAWNSTAR_DONOR_NET",
        r"C:\Users\uniqu\AppData\Local\Temp\deepblue-teammate-data\donor_reference\pawnstar-v12.bin",
    )
)
SKIPPED: list[str] = []


def _net():
    if DONOR_NET_PATH.exists():
        return weights.load_donor_net(DONOR_NET_PATH)
    return weights.random_synthetic_weights(seed=123)


def test_refresh_numba_matches_reference() -> None:
    nk.warm_up()
    w = _net()
    board = chess.Board()
    for perspective in (feat.WHITE, feat.BLACK):
        rows = np.array(feat.active_features(board, perspective), dtype=np.int32)
        got = nk.refresh_numba(w.feature_weights, w.feature_bias, rows)
        expected = ref.refresh_accumulator(w, board, perspective)
        assert (got == expected).all(), perspective


def test_update_numba_matches_reference_diff() -> None:
    nk.warm_up()
    w = _net()
    board_before = chess.Board()
    board_after = board_before.copy()
    board_after.push_uci("e2e4")

    for perspective in (feat.WHITE, feat.BLACK):
        acc_before = ref.refresh_accumulator(w, board_before, perspective)
        king_sq = feat.king_squares(board_after)[perspective]
        # e2 pawn -> e4 pawn: remove (e2, white, pawn), add (e4, white, pawn)
        removed = np.array(
            [feat.feature_row(feat.WHITE, 0, chess.E2, king_sq, perspective)], dtype=np.int32
        )
        added = np.array(
            [feat.feature_row(feat.WHITE, 0, chess.E4, king_sq, perspective)], dtype=np.int32
        )
        got = nk.update_numba(acc_before, w.feature_weights, removed, added)
        expected = ref.refresh_accumulator(w, board_after, perspective)
        assert (got == expected).all(), perspective


def test_tail_numba_matches_reference_eval() -> None:
    nk.warm_up()
    w = _net()
    board = chess.Board()
    acc_white = ref.refresh_accumulator(w, board, feat.WHITE)
    acc_black = ref.refresh_accumulator(w, board, feat.BLACK)
    got = nk.tail_numba(acc_white, acc_black, w.output_weights[0], w.output_weights[1], w.output_bias)
    expected = ref.evaluate_from_accumulators(w, acc_white, acc_black, True)
    assert got == expected, (got, expected)


def run_bench() -> None:
    print("Loading net and warming up kernels...")
    w = _net()
    t0 = time.perf_counter()
    nk.warm_up()
    print(f"warm-up: {time.perf_counter() - t0:.3f}s")

    board = chess.Board()
    rows_white = np.array(feat.active_features(board, feat.WHITE), dtype=np.int32)
    rows_black = np.array(feat.active_features(board, feat.BLACK), dtype=np.int32)
    acc_white = nk.refresh_numba(w.feature_weights, w.feature_bias, rows_white)
    acc_black = nk.refresh_numba(w.feature_weights, w.feature_bias, rows_black)

    def bench(fn, iters):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) / iters * 1e6

    n_refresh, n_update, n_tail = 20_000, 200_000, 200_000

    t_refresh = bench(lambda: nk.refresh_numba(w.feature_weights, w.feature_bias, rows_white), n_refresh)
    removed1 = np.array([rows_white[0]], dtype=np.int32)
    added1 = np.array([(int(rows_white[0]) + 1) % 6144], dtype=np.int32)
    t_quiet = bench(lambda: nk.update_numba(acc_white, w.feature_weights, removed1, added1), n_update)
    removed2 = np.array([rows_white[0], rows_white[1]], dtype=np.int32)
    added2 = np.array([rows_white[2]], dtype=np.int32)
    t_capture = bench(lambda: nk.update_numba(acc_white, w.feature_weights, removed2, added2), n_update)
    removed3 = np.array([rows_white[0], rows_white[1]], dtype=np.int32)
    added3 = np.array([rows_white[2], rows_white[3]], dtype=np.int32)
    t_castle = bench(lambda: nk.update_numba(acc_white, w.feature_weights, removed3, added3), n_update)
    t_tail = bench(lambda: nk.tail_numba(acc_white, acc_black, w.output_weights[0], w.output_weights[1], w.output_bias), n_tail)

    def update_plus_tail():
        a = nk.update_numba(acc_white, w.feature_weights, removed1, added1)
        nk.tail_numba(a, acc_black, w.output_weights[0], w.output_weights[1], w.output_bias)

    t_update_tail = bench(update_plus_tail, n_update)

    print(f"full refresh (one perspective, 32 rows): {t_refresh:.3f} us/call")
    print(f"quiet update (1 removed+1 added):        {t_quiet:.3f} us/call")
    print(f"capture update (2 removed+1 added):      {t_capture:.3f} us/call")
    print(f"castling update (2 removed+2 added):     {t_castle:.3f} us/call")
    print(f"tail only:                                {t_tail:.3f} us/call")
    print(f"quiet update + tail:                      {t_update_tail:.3f} us/call")
    print(f"implied evals/sec (tail only):            {1e6 / t_tail:,.0f}")
    print(f"implied evals/sec (update+tail):          {1e6 / t_update_tail:,.0f}")


def run_all() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"PASS {name}")
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    if "--bench" in sys.argv:
        run_bench()
        raise SystemExit(0)
    raise SystemExit(run_all())
