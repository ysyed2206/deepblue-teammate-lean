"""Validated readers for production Chess768 training artifacts.

Training shards stay outside the repository.  This module reads one shard at a
time and converts only the current mini-batch to ``torch.int64``, avoiding the
multi-gigabyte copy that a whole-dataset tensor conversion would create.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nnue_lab.production.features import (
    MAX_PIECES,
    NUM_FEATURES,
    NUM_PERSPECTIVES,
    PADDING_FEATURE,
)

LEGACY_BUCKET_FEATURES = 8 * NUM_FEATURES
LEGACY_PADDING_FEATURE = LEGACY_BUCKET_FEATURES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetFingerprint:
    path: str
    sha256: str
    bytes: int
    examples: int
    legacy_chessbuckets_mod768: bool


@dataclass(frozen=True)
class EncodedArrays:
    indices: np.ndarray
    sides: np.ndarray
    targets: np.ndarray
    quiet: np.ndarray | None = None

    @property
    def count(self) -> int:
        return int(self.indices.shape[0])


def _validate_arrays(arrays: EncodedArrays, path: Path) -> None:
    expected_tail = (NUM_PERSPECTIVES, MAX_PIECES)
    if arrays.indices.dtype != np.uint16:
        raise TypeError(f"{path}: indices must be uint16, got {arrays.indices.dtype}")
    if arrays.indices.ndim != 3 or tuple(arrays.indices.shape[1:]) != expected_tail:
        raise ValueError(
            f"{path}: indices must have shape [N,2,32], got {arrays.indices.shape}"
        )
    if arrays.sides.ndim != 1 or arrays.targets.ndim != 1:
        raise ValueError(f"{path}: sides and targets must be one-dimensional")
    if arrays.count != len(arrays.sides) or arrays.count != len(arrays.targets):
        raise ValueError(f"{path}: dataset arrays have different lengths")
    if arrays.quiet is not None and len(arrays.quiet) != arrays.count:
        raise ValueError(f"{path}: quiet array has a different length")
    if arrays.count and int(arrays.indices.max()) > PADDING_FEATURE:
        raise ValueError(f"{path}: feature index exceeds {PADDING_FEATURE}")
    if arrays.count and np.any((arrays.sides != 0) & (arrays.sides != 1)):
        raise ValueError(f"{path}: sides must contain only 0=White and 1=Black")
    if not np.all(np.isfinite(arrays.targets)):
        raise ValueError(f"{path}: targets contain non-finite values")


def load_encoded_npz(
    path: Path,
    *,
    legacy_chessbuckets_mod768: bool = False,
    limit: int | None = None,
) -> EncodedArrays:
    """Load and validate one NPZ shard.

    ``legacy_chessbuckets_mod768`` is an explicit smoke-test bridge for the
    existing 6144-row v0 datasets.  It removes the king-bucket component with
    ``row % 768`` and maps the old padding row 6144 to the new padding row 768.
    Production runs should use native 768-row data and leave it disabled.
    """
    path = path.resolve()
    if limit is not None and limit <= 0:
        raise ValueError("dataset limit must be positive")
    with np.load(path, allow_pickle=False) as loaded:
        required = {"indices", "sides", "targets"}
        missing = required.difference(loaded.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")
        stop = None if limit is None else limit
        source_indices = np.asarray(loaded["indices"][:stop])
        sides = np.asarray(loaded["sides"][:stop])
        targets = np.asarray(loaded["targets"][:stop], dtype=np.float32)
        quiet = (
            np.asarray(loaded["quiet"][:stop], dtype=np.uint8)
            if "quiet" in loaded.files
            else None
        )

    if source_indices.dtype != np.uint16:
        raise TypeError(f"{path}: source indices must be uint16")
    if legacy_chessbuckets_mod768:
        if source_indices.size and int(source_indices.max()) > LEGACY_PADDING_FEATURE:
            raise ValueError(f"{path}: legacy feature index exceeds 6144")
        valid = source_indices < LEGACY_PADDING_FEATURE
        indices = np.where(
            valid, source_indices % NUM_FEATURES, PADDING_FEATURE
        ).astype(np.uint16, copy=False)
    else:
        indices = np.ascontiguousarray(source_indices)

    arrays = EncodedArrays(
        indices=np.ascontiguousarray(indices),
        sides=np.ascontiguousarray(sides.astype(np.uint8, copy=False)),
        targets=np.ascontiguousarray(targets),
        quiet=None if quiet is None else np.ascontiguousarray(quiet),
    )
    _validate_arrays(arrays, path)
    return arrays


def fingerprint_dataset(
    path: Path, *, legacy_chessbuckets_mod768: bool = False
) -> DatasetFingerprint:
    arrays = load_encoded_npz(
        path, legacy_chessbuckets_mod768=legacy_chessbuckets_mod768
    )
    resolved = path.resolve()
    return DatasetFingerprint(
        path=str(resolved),
        sha256=sha256_file(resolved),
        bytes=resolved.stat().st_size,
        examples=arrays.count,
        legacy_chessbuckets_mod768=legacy_chessbuckets_mod768,
    )


def reject_pristine_before_selection(paths: list[Path], selection_locked: bool) -> None:
    """Make accidental pristine-set inspection an explicit CLI decision."""
    if selection_locked:
        return
    labelled = [str(path) for path in paths if "pristine" in str(path).lower()]
    if labelled:
        raise RuntimeError(
            "refusing to read a pristine artifact before architecture selection is "
            f"declared locked: {labelled}"
        )
