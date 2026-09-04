"""Numba njit kernels for the NNUE hot path: refresh, incremental update, and
the SCReLU-dot-dequant tail. This is the fast path ``BENCHMARKS.md`` measures
and the shape a real search integration would call (see
``INTEGRATION_CONTRACT.md``); ``reference_eval.py`` remains the source of
truth these are checked against.

Deliberately narrow scope: these functions take plain NumPy arrays (int16
accumulators/weights, int32 row-index arrays), not ``chess.Board`` objects or
the ``NNUEState``/dataclass wrappers used elsewhere in this lane -- Numba
nopython mode cannot see Python objects, so the row-index computation (which
still needs ``features.py``/python-chess) has to happen just outside the
njit boundary, exactly as a real search would resolve "which rows changed"
from its own board representation before calling into a compiled tail.
"""

from __future__ import annotations

import numpy as np
from numba import int16, int32, int64, njit

from weights import QA, QB, SCALE

_SIG_REFRESH = int16[:](int16[:, :], int16[:], int32[:])
_SIG_UPDATE = int16[:](int16[:], int16[:, :], int32[:], int32[:])
_SIG_TAIL = int64(int16[:], int16[:], int16[:], int16[:], int64)


@njit(_SIG_REFRESH, cache=True, fastmath=False)
def refresh_numba(feature_weights: np.ndarray, feature_bias: np.ndarray, active_rows: np.ndarray) -> np.ndarray:
    acc = feature_bias.copy()
    for i in range(active_rows.shape[0]):
        row = active_rows[i]
        for j in range(acc.shape[0]):
            acc[j] = np.int16(acc[j] + feature_weights[row, j])
    return acc


@njit(_SIG_UPDATE, cache=True, fastmath=False)
def update_numba(acc: np.ndarray, feature_weights: np.ndarray, removed_rows: np.ndarray, added_rows: np.ndarray) -> np.ndarray:
    out = acc.copy()
    for i in range(removed_rows.shape[0]):
        row = removed_rows[i]
        for j in range(out.shape[0]):
            out[j] = np.int16(out[j] - feature_weights[row, j])
    for i in range(added_rows.shape[0]):
        row = added_rows[i]
        for j in range(out.shape[0]):
            out[j] = np.int16(out[j] + feature_weights[row, j])
    return out


@njit(_SIG_TAIL, cache=True, fastmath=False)
def tail_numba(stm_acc: np.ndarray, ntm_acc: np.ndarray, w_stm: np.ndarray, w_ntm: np.ndarray, output_bias: int) -> int:
    dot = int64(0)
    qa = int64(QA)
    for i in range(stm_acc.shape[0]):
        x = int64(stm_acc[i])
        c = int64(0) if x < 0 else (qa if x > qa else x)
        dot += c * c * int64(w_stm[i])
    for i in range(ntm_acc.shape[0]):
        x = int64(ntm_acc[i])
        c = int64(0) if x < 0 else (qa if x > qa else x)
        dot += c * c * int64(w_ntm[i])
    out = dot // qa
    out += int64(output_bias)
    out *= int64(SCALE)
    denom = qa * int64(QB)
    # Truncate toward zero (C++ semantics), matching reference_eval.trunc_div.
    q = out // denom
    if q < 0 and out % denom != 0:
        q += 1
    return q


def warm_up(hidden_size: int = 1024) -> None:
    """Force JIT compilation of every kernel above, outside any timed region
    -- the donor's own README stresses warming compiled functions during the
    competition's import-time budget, not on the clock."""
    fw = np.zeros((8, hidden_size), dtype=np.int16)
    fb = np.zeros(hidden_size, dtype=np.int16)
    acc = np.zeros(hidden_size, dtype=np.int16)
    rows = np.array([0, 1], dtype=np.int32)
    refresh_numba(fw, fb, rows)
    update_numba(acc, fw, rows, rows)
    ow = np.zeros(hidden_size, dtype=np.int16)
    tail_numba(acc, acc, ow, ow, 0)
