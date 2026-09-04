"""Loader for a Pawnstar-stamped NNUE net file (the donor's own on-disk format).

Standalone: no dependency on the rest of ``nnue_lab``. See ``PAWNSTAR_FORMAT.md``
for the verified source citations.

File layout (verified from ``src/nnue.h``, ``NetHeader`` / ``Network::LoadFromMemory``):

    [0:32)   NetHeader: magic "PSN1"(4) + format_version(u16) + input_size(u16)
             + king_buckets(u16) + hidden_size(u16) + qa(i16) + qb(i16) + scale(i16)
             + 14 bytes reserved/zero-padding  -- 32 bytes total, little-endian.
    [32:...) payload, tightly packed little-endian int16, in this exact order:
             feature_weights : int16[FEATURE_ROWS * HIDDEN_SIZE]  (row-major: row*H + i)
             feature_bias    : int16[HIDDEN_SIZE]
             output_weights  : int16[2 * HIDDEN_SIZE]  (first H = side-to-move, next H = other)
             output_bias     : int16 scalar

A *raw* (unstamped) bullet export is rejected by the real engine and by this
loader alike -- stamp it first (donor's ``tools/stamp_net``) before loading.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from features import FEATURE_ROWS, HIDDEN_SIZE, INPUT_SIZE, NUM_KING_BUCKETS

NET_MAGIC = b"PSN1"
NET_FORMAT_VERSION = 1
HEADER_STRUCT = struct.Struct("<4sHHHHhhh14s")  # magic, version, input, buckets, hidden, qa, qb, scale, reserved
assert HEADER_STRUCT.size == 32, HEADER_STRUCT.size

QA = 255
QB = 64
SCALE = 400

_PAYLOAD_INT16_COUNT = FEATURE_ROWS * HIDDEN_SIZE + HIDDEN_SIZE + 2 * HIDDEN_SIZE + 1
PAYLOAD_BYTES = _PAYLOAD_INT16_COUNT * 2


@dataclass(frozen=True)
class NetHeader:
    magic: bytes
    format_version: int
    input_size: int
    king_buckets: int
    hidden_size: int
    qa: int
    qb: int
    scale: int

    def matches_expected_architecture(self) -> bool:
        return (
            self.magic == NET_MAGIC
            and self.format_version == NET_FORMAT_VERSION
            and self.input_size == INPUT_SIZE
            and self.king_buckets == NUM_KING_BUCKETS
            and self.hidden_size == HIDDEN_SIZE
            and self.qa == QA
            and self.qb == QB
            and self.scale == SCALE
        )


@dataclass
class NNUEWeights:
    """Donor-compatible weights, decoded to convenient NumPy shapes.

    ``feature_weights``: int16[FEATURE_ROWS, HIDDEN_SIZE] -- row ``r`` is the
      column added to an accumulator when feature row ``r`` is active.
    ``feature_bias``: int16[HIDDEN_SIZE]
    ``output_weights``: int16[2, HIDDEN_SIZE] -- index 0 weights the
      side-to-move accumulator, index 1 the other side's.
    ``output_bias``: Python int (int16 range), in QA*QB units.
    """

    feature_weights: np.ndarray
    feature_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: int
    header: NetHeader


def parse_header(data: bytes) -> NetHeader:
    if len(data) < HEADER_STRUCT.size:
        raise ValueError(f"file too short for a NetHeader: {len(data)} bytes")
    magic, version, input_size, king_buckets, hidden_size, qa, qb, scale, _reserved = HEADER_STRUCT.unpack_from(data, 0)
    return NetHeader(
        magic=magic,
        format_version=version,
        input_size=input_size,
        king_buckets=king_buckets,
        hidden_size=hidden_size,
        qa=qa,
        qb=qb,
        scale=scale,
    )


def load_donor_net(path: str | Path) -> NNUEWeights:
    """Load a Pawnstar-stamped net file. Raises ``ValueError`` on a missing
    magic, an architecture mismatch, or a truncated payload -- mirroring the
    donor engine's own ``Network::LoadFromMemory`` rejection behaviour."""
    data = Path(path).read_bytes()
    if len(data) < HEADER_STRUCT.size or data[:4] != NET_MAGIC:
        raise ValueError(f"'{path}' is not a stamped Pawnstar net (missing 'PSN1' header)")
    header = parse_header(data)
    if not header.matches_expected_architecture():
        raise ValueError(
            f"'{path}' architecture mismatch: file is "
            f"v{header.format_version}/in{header.input_size}/buckets{header.king_buckets}/"
            f"hidden{header.hidden_size}/qa{header.qa}/qb{header.qb}/scale{header.scale}, "
            f"expected v{NET_FORMAT_VERSION}/in{INPUT_SIZE}/buckets{NUM_KING_BUCKETS}/"
            f"hidden{HIDDEN_SIZE}/qa{QA}/qb{QB}/scale{SCALE}"
        )
    payload = data[HEADER_STRUCT.size :]
    if len(payload) < PAYLOAD_BYTES:
        raise ValueError(f"'{path}' truncated: payload is {len(payload)} bytes, need {PAYLOAD_BYTES}")

    offset = 0
    fw_count = FEATURE_ROWS * HIDDEN_SIZE
    feature_weights = np.frombuffer(payload, dtype="<i2", count=fw_count, offset=offset).reshape(
        FEATURE_ROWS, HIDDEN_SIZE
    )
    offset += fw_count * 2
    feature_bias = np.frombuffer(payload, dtype="<i2", count=HIDDEN_SIZE, offset=offset).copy()
    offset += HIDDEN_SIZE * 2
    output_weights_flat = np.frombuffer(payload, dtype="<i2", count=2 * HIDDEN_SIZE, offset=offset)
    output_weights = output_weights_flat.reshape(2, HIDDEN_SIZE).copy()
    offset += 2 * HIDDEN_SIZE * 2
    (output_bias,) = struct.unpack_from("<h", payload, offset)

    return NNUEWeights(
        feature_weights=feature_weights.copy(),
        feature_bias=feature_bias,
        output_weights=output_weights,
        output_bias=int(output_bias),
        header=header,
    )


def random_synthetic_weights(seed: int = 0, weight_bound: int = 90) -> NNUEWeights:
    """Small-magnitude random int16 weights for tests that need *some* net but
    not a specific trained one (correctness/incremental gates: the encoding
    logic under test does not depend on which weights are loaded). Bounded
    well under int16 range so accumulator sums over 32 pieces cannot approach
    overflow, matching how a real trained net's small weights behave."""
    rng = np.random.default_rng(seed)
    feature_weights = rng.integers(-weight_bound, weight_bound + 1, size=(FEATURE_ROWS, HIDDEN_SIZE), dtype=np.int32).astype(
        np.int16
    )
    feature_bias = rng.integers(-weight_bound, weight_bound + 1, size=HIDDEN_SIZE, dtype=np.int32).astype(np.int16)
    output_weights = rng.integers(-weight_bound, weight_bound + 1, size=(2, HIDDEN_SIZE), dtype=np.int32).astype(np.int16)
    output_bias = int(rng.integers(-1000, 1000))
    header = NetHeader(NET_MAGIC, NET_FORMAT_VERSION, INPUT_SIZE, NUM_KING_BUCKETS, HIDDEN_SIZE, QA, QB, SCALE)
    return NNUEWeights(feature_weights, feature_bias, output_weights, output_bias, header)
