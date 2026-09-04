"""Materialise the two completed 5m-acquisition sources as an append-only train set.

Stage 1 is opened read-only and only its FEN columns are queried. Its existing
holdouts and training arrays are neither rebuilt nor evaluated. Run from the
repository root with ``.venv/Scripts/python.exe -B -m
nnue_lab.production.prepare_extension``. Existing outputs are never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from nnue_lab.production.prepare_data import (
    CLIP_CP,
    SPLIT_SEED,
    canonical_fen_digest_update,
    connect_output_database,
    detach,
    export_split,
    filter_invalid_positions,
    sha256_file,
    split_for_fen,
)

DATA_ROOT = Path("C:/Users/mohib/AppData/Local/Temp/deepblue-nnue-data")
LAB_ROOT = Path(__file__).resolve().parents[1]
STAGE = "stage2_5m"
SOURCES = ("shard0013", "shard0017")


def connect_extension_database(path: Path) -> sqlite3.Connection:
    # Initialise the exact existing ledger schema, then enable SQLite URI parsing
    # on the working connection so attached inputs can be strictly read-only.
    connect_output_database(path).close()
    connection = sqlite3.connect(path.resolve().as_uri(), uri=True)
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-131072")
    return connection


def attach_readonly(connection: sqlite3.Connection, path: Path) -> None:
    """Do not create journals or write even transient state beside sealed inputs."""
    for suffix in ("-wal", "-journal"):
        companion = Path(str(path) + suffix)
        if companion.exists() and companion.stat().st_size:
            raise RuntimeError(f"input has an active SQLite journal: {companion}")
    connection.execute(
        "ATTACH DATABASE ? AS source_db",
        (path.resolve().as_uri() + "?mode=ro&immutable=1",),
    )


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def exclude_ledger(connection: sqlite3.Connection, path: Path) -> dict[str, Any]:
    """Exclude all FENs, including Stage 1's old_fens and invalid positions."""
    attach_readonly(connection, path)
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM source_db.sqlite_master WHERE type='table'"
        )
    }
    if "positions" not in tables:
        raise ValueError(f"prior ledger is missing positions: {path}")
    stats: dict[str, Any] = {"database": str(path.resolve()), "tables": {}}
    for table in ("positions", "old_fens"):
        if table not in tables:
            continue
        before = connection.total_changes
        # The table names are fixed literals, never supplied by an input file.
        connection.execute(f"INSERT OR IGNORE INTO old_fens SELECT fen FROM source_db.{table}")
        stats["tables"][table] = {
            "unique_exclusions_added": connection.total_changes - before
        }
    detach(connection)
    return stats


def merge_train_source(connection: sqlite3.Connection, path: Path) -> dict[str, Any]:
    """Use the Stage-1 depth/knodes upsert, restricted to new train-bucket FENs."""
    attach_readonly(connection, path)
    connection.create_function(
        "is_train_fen", 1, lambda fen: split_for_fen(fen) == "train", deterministic=True
    )
    rows, mates, missing_cp, train_candidates = connection.execute(
        """
        SELECT COUNT(*),SUM(mate IS NOT NULL),SUM(mate IS NULL AND cp IS NULL),
               SUM(mate IS NULL AND cp IS NOT NULL AND is_train_fen(fen))
        FROM source_db.positions
        """
    ).fetchone()
    excluded = int(connection.execute(
        """SELECT COUNT(*) FROM source_db.positions AS source
           JOIN old_fens AS old ON old.fen=source.fen
           WHERE source.mate IS NULL AND source.cp IS NOT NULL
             AND is_train_fen(source.fen)"""
    ).fetchone()[0])
    duplicate, updates = connection.execute(
        """SELECT COUNT(*),SUM(source.depth > current.depth OR
                   (source.depth=current.depth AND source.knodes > current.knodes))
           FROM source_db.positions AS source
           JOIN positions AS current ON current.fen=source.fen
           WHERE source.mate IS NULL AND source.cp IS NOT NULL
             AND is_train_fen(source.fen)
             AND NOT EXISTS (SELECT 1 FROM old_fens AS old WHERE old.fen=source.fen)"""
    ).fetchone()
    before = int(connection.execute("SELECT COUNT(*) FROM positions").fetchone()[0])
    # Same deterministic quality order and exact-tie behaviour as prepare_data.
    connection.execute(
        """
        INSERT INTO positions(fen,line,depth,knodes,cp,source_name,source_row,valid)
        SELECT source.fen,source.line,source.depth,source.knodes,source.cp,?,source.first_row,1
        FROM source_db.positions AS source
        WHERE source.mate IS NULL AND source.cp IS NOT NULL
          AND is_train_fen(source.fen)
          AND NOT EXISTS (SELECT 1 FROM old_fens AS old WHERE old.fen=source.fen)
          AND 1
        ON CONFLICT(fen) DO UPDATE SET
            line=excluded.line,depth=excluded.depth,knodes=excluded.knodes,
            cp=excluded.cp,source_name=excluded.source_name,source_row=excluded.source_row
        WHERE excluded.depth > positions.depth OR
              (excluded.depth=positions.depth AND excluded.knodes > positions.knodes)
        """,
        (path.parent.name,),
    )
    after = int(connection.execute("SELECT COUNT(*) FROM positions").fetchone()[0])
    detach(connection)
    return {
        "database": str(path.resolve()),
        "rows": int(rows),
        "mate_rows_excluded": int(mates or 0),
        "missing_cp_rows_excluded": int(missing_cp or 0),
        "nonmate_cp_train_bucket_rows": int(train_candidates or 0),
        "prior_ledger_overlap_excluded": excluded,
        "duplicates_with_earlier_sources": int(duplicate),
        "higher_quality_updates": int(updates or 0),
        "new_unique_train_candidates": after - before,
    }


def main() -> None:
    started = time.perf_counter()
    output_dir = DATA_ROOT / "production_extensions" / STAGE
    manifest_path = LAB_ROOT / "production" / "manifests" / f"{STAGE}_extension_manifest.json"
    frozen_dir = DATA_ROOT / "production_frozen"
    frozen_ledger = frozen_dir / "frozen_positions.sqlite3"
    if output_dir.exists() or manifest_path.exists():
        raise FileExistsError("append-only output already exists; refusing to overwrite")
    prior_ledgers = sorted((DATA_ROOT / "production_extensions").rglob("*.sqlite3"))
    databases = [DATA_ROOT / "production_v1" / name / "selected.sqlite3" for name in SOURCES]
    protected_paths = sorted(p for p in frozen_dir.iterdir() if p.is_file())
    protected_paths.extend(prior_ledgers)
    protected_before = [file_record(path) for path in protected_paths]
    inputs: list[dict[str, Any]] = []
    for name, database in zip(SOURCES, databases, strict=True):
        acquisition_path = LAB_ROOT / "production" / "manifests" / f"download_{name[5:]}.json"
        acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
        selected = database.with_name("selected.jsonl")
        selected_record = file_record(selected)
        if selected_record["sha256"] != acquisition["selected_jsonl_sha256"]:
            raise ValueError(f"completed acquisition hash mismatch: {selected}")
        if acquisition["unique_fens"] != acquisition["requested_unique_fens"]:
            raise ValueError(f"acquisition is incomplete: {acquisition_path}")
        inputs.append({
            "database": file_record(database),
            "selected_jsonl": selected_record,
            "acquisition_manifest": file_record(acquisition_path),
            "completed_unique_fens": acquisition["unique_fens"],
        })
    output_dir.mkdir(parents=True)
    ledger_path = output_dir / "extension_positions.sqlite3"
    connection = connect_extension_database(ledger_path)
    try:
        print("Building exact-FEN exclusion ledger (read-only inputs)", flush=True)
        exclusions = [exclude_ledger(connection, path) for path in [frozen_ledger, *prior_ledgers]]
        exclusion_count = int(connection.execute("SELECT COUNT(*) FROM old_fens").fetchone()[0])
        source_stats = []
        for database in databases:
            print(f"Merging {database.parent.name}: train buckets only", flush=True)
            source_stats.append(merge_train_source(connection, database))
        print("Validating new training feature domains", flush=True)
        validation = filter_invalid_positions(connection)
        # Invalid FENs remain in rejected_positions, never in the training ledger.
        connection.execute("DELETE FROM positions WHERE valid=0")
        connection.commit()
        count = int(connection.execute("SELECT COUNT(*) FROM positions").fetchone()[0])
        if not count:
            raise RuntimeError("no additive train positions survived")
        overlap = int(connection.execute(
            "SELECT COUNT(*) FROM positions JOIN old_fens USING(fen)"
        ).fetchone()[0])
        if overlap:
            raise AssertionError(f"extension overlaps prior ledgers: {overlap}")
        membership = hashlib.sha256()
        for (fen,) in connection.execute("SELECT fen FROM positions ORDER BY fen"):
            if split_for_fen(fen) != "train":
                raise AssertionError("non-training FEN entered extension ledger")
            canonical_fen_digest_update(membership, fen)
        print(f"Exporting {count:,} additive train rows", flush=True)
        train = export_split(connection, output_dir, "train", count)
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise RuntimeError(f"extension ledger integrity failure: {integrity}")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_mode=DELETE")
    finally:
        connection.close()
    protected_after = [file_record(path) for path in protected_paths]
    if protected_before != protected_after:
        raise AssertionError("protected Stage-1/prior-ledger bytes changed during materialisation")
    for item in inputs:
        database_record = item["database"]
        if file_record(Path(database_record["path"])) != database_record:
            raise AssertionError("source acquisition changed during materialisation")
    manifest = {
        "format": "deepblue-nnue-append-only-training-extension-v1",
        "stage": STAGE,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "complete; additive training only; no model evaluation performed",
        "network_used": False,
        "sources": inputs,
        "selection": {
            "deduplication_key": "exact FEN string",
            "quality_order": "greatest depth then knodes; earlier source wins exact ties",
            "exclusion_ledgers": exclusions,
            "unique_excluded_fens": exclusion_count,
            "source_merge": source_stats,
            "feature_domain_validation": validation,
            "invalid_ledger_policy": "invalid FENs retained only in rejected_positions",
            "prior_fen_overlap": overlap,
        },
        "split": {
            "seed": SPLIT_SEED,
            "algorithm": "BLAKE2b-64 exact FEN keyed hash, little-endian mod 10000",
            "included_buckets": "500..9999 train only",
            "membership_fen_sha256": membership.hexdigest(),
        },
        "features": (
            "unchanged prepare_data.encode_fen: fixed White/Black Chess768, uint16[2,32], pad=768"
        ),
        "targets": f"unchanged side-to-move cp; White-relative source clipped to +/-{CLIP_CP}",
        "additive_train_rows": count,
        "datasets": {"train": train},
        "database": {**file_record(ledger_path), "integrity_check": integrity},
        "protected_files_byte_identical": protected_after,
        "stage1_holdouts": "unchanged; only FEN columns queried for exclusion; no metrics computed",
        "materializer": file_record(Path(__file__)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps({
        "manifest": str(manifest_path),
        "additive_train_rows": count,
        "membership_fen_sha256": membership.hexdigest(),
        "train_files": train["npz_files"],
        "elapsed_seconds": manifest["elapsed_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
