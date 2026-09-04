"""Read-only integrity check for the immutable Stage-1 production freeze."""

from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_npz(entry: dict[str, Any], *, pristine: bool) -> int:
    path = Path(entry["path"])
    if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
        raise AssertionError(f"file pin mismatch: {path}")
    with np.load(path) as data:
        indices = data["indices"]
        sides = data["sides"]
        targets = data["targets"]
        rows = int(indices.shape[0])
        if indices.shape != (rows, 2, 32) or indices.dtype != np.uint16:
            raise AssertionError(f"feature shape/dtype mismatch: {path}")
        if sides.shape != (rows,) or sides.dtype != np.uint8:
            raise AssertionError(f"side shape/dtype mismatch: {path}")
        if targets.shape != (rows,) or targets.dtype != np.int16:
            raise AssertionError(f"target shape/dtype mismatch: {path}")
        if np.any(indices > 768) or np.any((sides != 0) & (sides != 1)):
            raise AssertionError(f"feature/side domain mismatch: {path}")
        if not pristine and np.any(np.abs(targets.astype(np.int32)) > 2000):
            raise AssertionError(f"target clip mismatch: {path}")
    if rows != entry["rows"]:
        raise AssertionError(f"row-count pin mismatch: {path}")
    return rows


def main() -> None:
    lab_dir = Path(__file__).resolve().parent
    manifest_path = lab_dir / "manifests" / "data_freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    old_rows = validate_npz(manifest["old_training_component"], pristine=False)
    checked: dict[str, int] = {}
    for name, dataset in manifest["datasets"].items():
        rows = sum(
            validate_npz(entry, pristine=name == "pristine")
            for entry in dataset["npz_files"]
        )
        if rows != dataset["count"] or rows != manifest["split_counts"][name]:
            raise AssertionError(f"dataset row mismatch: {name}")
        checked[name] = rows
        if name != "train":
            fen_entry = dataset["fens"]
            fen_path = Path(fen_entry["path"])
            if fen_path.stat().st_size != fen_entry["bytes"] or sha256_file(fen_path) != fen_entry["sha256"]:
                raise AssertionError(f"FEN-sidecar pin mismatch: {name}")
            with gzip.open(fen_path, "rt", encoding="utf-8") as handle:
                fen_rows = sum(1 for _ in handle)
            if fen_rows != rows:
                raise AssertionError(f"FEN-sidecar row mismatch: {name}")

    database = manifest["database"]
    database_path = Path(database["path"])
    if database_path.stat().st_size != database["bytes"] or sha256_file(database_path) != database["sha256"]:
        raise AssertionError("database pin mismatch")
    uri = f"file:{database_path.as_posix()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        valid = int(connection.execute("SELECT COUNT(*) FROM positions WHERE valid=1").fetchone()[0])
        invalid = int(connection.execute("SELECT COUNT(*) FROM positions WHERE valid=0").fetchone()[0])
        old_overlap = int(
            connection.execute(
                "SELECT COUNT(*) FROM positions JOIN old_fens USING(fen)"
            ).fetchone()[0]
        )
    finally:
        connection.close()
    if integrity != "ok" or valid != sum(checked.values()) or invalid != 471 or old_overlap != 0:
        raise AssertionError("database integrity/count/exclusion mismatch")

    report = {
        "format": "deepblue-nnue-production-freeze-validation-v1",
        "ok": True,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "split_rows": checked,
        "converted_old_rows": old_rows,
        "combined_training_rows": checked["train"] + old_rows,
        "database_integrity_check": integrity,
        "invalid_rows_preserved_outside_splits": invalid,
        "old_fen_overlap_in_new_ledger": old_overlap,
        "pristine_check_scope": "file/hash/shape/feature-domain/count only; no target statistic or model metric",
    }
    output = lab_dir / "manifests" / "data_freeze_validation.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
