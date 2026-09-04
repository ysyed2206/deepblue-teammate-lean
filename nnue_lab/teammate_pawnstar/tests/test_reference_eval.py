"""Correctness tests for weights.py / reference_eval.py.

The headline test (`test_matches_all_250_donor_reference_evals`) needs the
real donor net file, which is intentionally kept outside the repo (see
DIFFERENTIAL_RESULTS.md). It is skipped with a clear message if the file is
not present at the expected path, rather than failing the whole suite.

Run with: .venv/Scripts/python.exe nnue_lab/teammate_pawnstar/tests/test_reference_eval.py
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reference_eval as ref  # noqa: E402
import weights  # noqa: E402

DONOR_NET_PATH = Path(
    os.environ.get(
        "PAWNSTAR_DONOR_NET",
        r"C:\Users\uniqu\AppData\Local\Temp\deepblue-teammate-data\donor_reference\pawnstar-v12.bin",
    )
)
REFERENCE_250_PATH = Path(__file__).resolve().parent.parent / "nnue_reference_250.txt"

SKIPPED: list[str] = []


def test_trunc_div_matches_cpp_truncation() -> None:
    assert ref.trunc_div(7, 2) == 3
    assert ref.trunc_div(-7, 2) == -3  # C++ truncates toward zero, NOT floors (-4)
    assert ref.trunc_div(-8, 2) == -4
    assert ref.trunc_div(0, 5) == 0
    assert ref.trunc_div(9, 3) == 3


def test_screlu_clamps_and_squares() -> None:
    import numpy as np

    x = np.array([-10, 0, 100, 255, 300], dtype=np.int16)
    got = ref.screlu(x)
    expected = np.array([0, 0, 100 * 100, 255 * 255, 255 * 255], dtype=np.int64)
    assert (got == expected).all(), got


def test_random_synthetic_weights_have_expected_shapes() -> None:
    w = weights.random_synthetic_weights(seed=1)
    assert w.feature_weights.shape == (6144, 1024)
    assert w.feature_bias.shape == (1024,)
    assert w.output_weights.shape == (2, 1024)
    assert w.header.matches_expected_architecture()


def test_loader_rejects_missing_magic() -> None:
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
        fh.write(b"NOT A NET" + b"\x00" * 64)
        path = fh.name
    try:
        weights.load_donor_net(path)
    except ValueError as exc:
        assert "not a stamped Pawnstar net" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unstamped file")
    finally:
        os.unlink(path)


def test_loader_rejects_architecture_mismatch() -> None:
    import struct
    import tempfile

    bad_header = struct.pack("<4sHHHHhhh14s", b"PSN1", 1, 768, 4, 512, 255, 64, 400, b"\x00" * 14)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
        fh.write(bad_header + b"\x00" * 100)
        path = fh.name
    try:
        weights.load_donor_net(path)
    except ValueError as exc:
        assert "architecture mismatch" in str(exc)
    else:
        raise AssertionError("expected ValueError for a mismatched architecture")
    finally:
        os.unlink(path)


def test_up_a_queen_is_a_large_positive_eval_with_real_weights() -> None:
    if not DONOR_NET_PATH.exists():
        SKIPPED.append("test_up_a_queen_is_a_large_positive_eval_with_real_weights (no donor net file)")
        return
    w = weights.load_donor_net(DONOR_NET_PATH)
    base = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    up_queen = chess.Board("4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1")
    base_eval = ref.evaluate(w, base)
    queen_eval = ref.evaluate(w, up_queen)
    assert queen_eval - base_eval > 500, (base_eval, queen_eval)


def test_matches_all_250_donor_reference_evals() -> None:
    if not DONOR_NET_PATH.exists():
        SKIPPED.append("test_matches_all_250_donor_reference_evals (no donor net file)")
        return
    w = weights.load_donor_net(DONOR_NET_PATH)
    lines = REFERENCE_250_PATH.read_text().strip().splitlines()
    assert len(lines) == 250
    mismatches = []
    for line in lines:
        fen, expected_str = line.split("|")
        expected = int(expected_str.strip())
        got = ref.evaluate(w, chess.Board(fen.strip()))
        if got != expected:
            mismatches.append((fen.strip(), expected, got))
    assert not mismatches, f"{len(mismatches)}/250 mismatches, e.g. {mismatches[:3]}"


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
