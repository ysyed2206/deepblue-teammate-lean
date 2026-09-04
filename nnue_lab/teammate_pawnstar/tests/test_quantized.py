"""Correctness tests for quantized.py (the int8 output path).

Run with: .venv/Scripts/python.exe nnue_lab/teammate_pawnstar/tests/test_quantized.py
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import quantized as q  # noqa: E402
import reference_eval as ref  # noqa: E402
import weights  # noqa: E402

DONOR_NET_PATH = Path(
    os.environ.get(
        "PAWNSTAR_DONOR_NET",
        r"C:\Users\uniqu\AppData\Local\Temp\deepblue-teammate-data\donor_reference\pawnstar-v12.bin",
    )
)
SKIPPED: list[str] = []


def test_output_weight_scale_is_one_when_weights_fit_int8() -> None:
    w = np.array([[100, -127, 50], [127, -100, 0]], dtype=np.int16)
    assert q.output_weight_scale(w) == 1


def test_output_weight_scale_scales_when_weights_exceed_int8() -> None:
    w = np.array([[300, -400, 0]], dtype=np.int16)
    scale = q.output_weight_scale(w)
    assert scale == 4  # ceil(400/127) = 4


def test_quantize_output_weights_int8_stays_in_range() -> None:
    rng = np.random.default_rng(0)
    w = rng.integers(-20000, 20000, size=(2, 1024)).astype(np.int16)
    q8, scale = q.quantize_output_weights_int8(w)
    assert q8.dtype == np.int8
    assert q8.min() >= -127
    assert q8.max() <= 127
    assert scale >= 1


def test_screlu_u8_bounded_0_255() -> None:
    x = np.array([-500, 0, 128, 255, 1000], dtype=np.int32)
    out = q.screlu_u8(x)
    assert (out >= 0).all() and (out <= 255).all()


def test_int8_eval_within_donor_bound_on_real_weights() -> None:
    if not DONOR_NET_PATH.exists():
        SKIPPED.append("test_int8_eval_within_donor_bound_on_real_weights (no donor net file)")
        return
    w = weights.load_donor_net(DONOR_NET_PATH)
    ref_path = Path(__file__).resolve().parent.parent / "nnue_reference_250.txt"
    lines = ref_path.read_text().strip().splitlines()
    max_diff = 0
    for line in lines:
        fen, _ = line.split("|")
        board = chess.Board(fen.strip())
        exact = ref.evaluate(w, board)
        approx = q.evaluate_int8_from_board(w, board)
        max_diff = max(max_diff, abs(approx - exact))
    # Donor's own measured bound is "~26 cp"; allow headroom since it is an
    # empirical, not a hard, bound.
    assert max_diff <= 60, f"max |int8 - int16| = {max_diff} cp, expected roughly <=26-30 cp"


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
    if SKIPPED:
        print(f"({len(SKIPPED)} skipped: {SKIPPED})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run_all())
