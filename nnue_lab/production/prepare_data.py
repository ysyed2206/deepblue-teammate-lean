"""Freeze a deterministic, leakage-safe Chess768 production dataset.

This script consumes only completed local acquisitions.  It never downloads
data, never writes to the live engine, and refuses to place bulky derivatives
inside the repository.  The external SQLite ledger remains the canonical FEN
inventory; NumPy shards are a deterministic materialisation of that ledger.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np


FORMAT = "deepblue-nnue-production-staging-freeze-v1"
STAGE = "stage-1-completed-3m-acquisitions"
SPLIT_SEED = 20260910
CLIP_CP = 2000
PADDING_FEATURE = 768
MAX_PIECES = 32
TRAIN_SHARD_ROWS = 250_000

PIECE_CODES = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}

SPLIT_NAMES = ("pristine", "validation", "development", "train")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash64(text: str, seed: int) -> int:
    key = seed.to_bytes(8, "little", signed=False)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8, key=key).digest()
    return int.from_bytes(digest, "little", signed=False)


def split_for_fen(fen: str) -> str:
    """Return the fixed 2/1.5/1.5/95 percent split for an exact FEN."""
    bucket = stable_hash64(fen, SPLIT_SEED) % 10_000
    if bucket < 200:
        return "pristine"
    if bucket < 350:
        return "validation"
    if bucket < 500:
        return "development"
    return "train"


def feature_index(piece: int, square: int, perspective: int) -> int:
    relative_colour = (piece // 6) ^ perspective
    piece_type = piece % 6
    oriented_square = square if perspective == 0 else square ^ 56
    return relative_colour * 384 + piece_type * 64 + oriented_square


def encode_fen(fen: str) -> tuple[np.ndarray, int]:
    """Encode an exact FEN as sorted/padded fixed White and Black views."""
    fields = fen.split()
    if len(fields) < 2 or fields[1] not in ("w", "b"):
        raise ValueError(f"malformed FEN side-to-move field: {fen!r}")
    ranks = fields[0].split("/")
    if len(ranks) != 8:
        raise ValueError(f"malformed FEN board field: {fen!r}")

    pieces: list[tuple[int, int]] = []
    kings = [0, 0]
    for fen_rank, rank_text in enumerate(ranks):
        file_index = 0
        board_rank = 7 - fen_rank
        for token in rank_text:
            if "1" <= token <= "8":
                file_index += ord(token) - ord("0")
                continue
            piece = PIECE_CODES.get(token)
            if piece is None or file_index >= 8:
                raise ValueError(f"malformed FEN piece placement: {fen!r}")
            square = board_rank * 8 + file_index
            pieces.append((square, piece))
            if piece == 5:
                kings[0] += 1
            elif piece == 11:
                kings[1] += 1
            file_index += 1
        if file_index != 8:
            raise ValueError(f"malformed FEN rank width: {fen!r}")
    if len(pieces) > MAX_PIECES:
        raise ValueError(f"too many pieces ({len(pieces)}): {fen!r}")
    if kings != [1, 1]:
        raise ValueError(f"expected exactly one king per colour: {fen!r}")

    pieces.sort()
    encoded = np.full((2, MAX_PIECES), PADDING_FEATURE, dtype=np.uint16)
    for slot, (square, piece) in enumerate(pieces):
        encoded[0, slot] = feature_index(piece, square, 0)
        encoded[1, slot] = feature_index(piece, square, 1)
    side = 0 if fields[1] == "w" else 1
    return encoded, side


def side_to_move_target(white_relative_cp: int, side: int) -> tuple[int, bool]:
    clipped = max(-CLIP_CP, min(CLIP_CP, int(white_relative_cp)))
    return (clipped if side == 0 else -clipped), clipped != int(white_relative_cp)


def canonical_fen_digest_update(digest: Any, fen: str) -> None:
    digest.update(fen.encode("utf-8"))
    digest.update(b"\n")


def connect_output_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-131072")
    connection.executescript(
        """
        CREATE TABLE old_fens (fen TEXT PRIMARY KEY) WITHOUT ROWID;
        CREATE TABLE positions (
            fen TEXT PRIMARY KEY,
            line TEXT NOT NULL,
            depth INTEGER NOT NULL,
            knodes INTEGER NOT NULL,
            cp INTEGER NOT NULL,
            source_name TEXT NOT NULL,
            source_row INTEGER NOT NULL,
            valid INTEGER NOT NULL DEFAULT 1
        ) WITHOUT ROWID;
        CREATE TABLE rejected_positions (
            fen TEXT PRIMARY KEY,
            reason TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE freeze_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        """
    )
    connection.commit()
    return connection


def connect_existing_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-131072")
    columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(positions)")
    }
    if "valid" not in columns:
        connection.execute("ALTER TABLE positions ADD COLUMN valid INTEGER NOT NULL DEFAULT 1")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS rejected_positions (
            fen TEXT PRIMARY KEY,
            reason TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS freeze_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        """
    )
    connection.commit()
    return connection


def attach(connection: sqlite3.Connection, path: Path, alias: str = "source_db") -> None:
    connection.execute(f"ATTACH DATABASE ? AS {alias}", (str(path.resolve()),))


def detach(connection: sqlite3.Connection, alias: str = "source_db") -> None:
    connection.commit()
    connection.execute(f"DETACH DATABASE {alias}")


def build_old_exclusion(
    connection: sqlite3.Connection, old_databases: Iterable[Path]
) -> tuple[list[dict[str, Any]], int]:
    stats: list[dict[str, Any]] = []
    for database in old_databases:
        attach(connection, database)
        rows = int(connection.execute("SELECT COUNT(*) FROM source_db.positions").fetchone()[0])
        before = int(connection.execute("SELECT COUNT(*) FROM old_fens").fetchone()[0])
        connection.execute("INSERT OR IGNORE INTO old_fens SELECT fen FROM source_db.positions")
        after = int(connection.execute("SELECT COUNT(*) FROM old_fens").fetchone()[0])
        detach(connection)
        stats.append(
            {
                "database": str(database.resolve()),
                "rows": rows,
                "new_unique_old_fens": after - before,
                "duplicates_with_earlier_old_inputs": rows - (after - before),
            }
        )
    unique = int(connection.execute("SELECT COUNT(*) FROM old_fens").fetchone()[0])
    return stats, unique


def merge_new_positions(
    connection: sqlite3.Connection, new_databases: Iterable[Path]
) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for database in new_databases:
        source_name = database.parent.name
        attach(connection, database)
        source_rows = int(
            connection.execute("SELECT COUNT(*) FROM source_db.positions").fetchone()[0]
        )
        mate_rows = int(
            connection.execute(
                "SELECT COUNT(*) FROM source_db.positions WHERE mate IS NOT NULL"
            ).fetchone()[0]
        )
        missing_cp_rows = int(
            connection.execute(
                "SELECT COUNT(*) FROM source_db.positions WHERE cp IS NULL AND mate IS NULL"
            ).fetchone()[0]
        )
        eligible_rows = int(
            connection.execute(
                "SELECT COUNT(*) FROM source_db.positions WHERE mate IS NULL AND cp IS NOT NULL"
            ).fetchone()[0]
        )
        old_overlap = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM source_db.positions AS source
                JOIN old_fens AS old ON old.fen=source.fen
                WHERE source.mate IS NULL AND source.cp IS NOT NULL
                """
            ).fetchone()[0]
        )
        duplicate_new = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM source_db.positions AS source
                JOIN positions AS current ON current.fen=source.fen
                WHERE source.mate IS NULL AND source.cp IS NOT NULL
                  AND NOT EXISTS (SELECT 1 FROM old_fens AS old WHERE old.fen=source.fen)
                """
            ).fetchone()[0]
        )
        quality_updates = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM source_db.positions AS source
                JOIN positions AS current ON current.fen=source.fen
                WHERE source.mate IS NULL AND source.cp IS NOT NULL
                  AND NOT EXISTS (SELECT 1 FROM old_fens AS old WHERE old.fen=source.fen)
                  AND (source.depth > current.depth OR
                       (source.depth = current.depth AND source.knodes > current.knodes))
                """
            ).fetchone()[0]
        )
        before = int(connection.execute("SELECT COUNT(*) FROM positions").fetchone()[0])
        connection.execute(
            """
            INSERT INTO positions(fen,line,depth,knodes,cp,source_name,source_row,valid)
            SELECT source.fen,source.line,source.depth,source.knodes,source.cp,?,source.first_row,1
            FROM source_db.positions AS source
            WHERE source.mate IS NULL AND source.cp IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM old_fens AS old WHERE old.fen=source.fen)
              AND 1
            ON CONFLICT(fen) DO UPDATE SET
                line=excluded.line,
                depth=excluded.depth,
                knodes=excluded.knodes,
                cp=excluded.cp,
                source_name=excluded.source_name,
                source_row=excluded.source_row
            WHERE excluded.depth > positions.depth OR
                  (excluded.depth = positions.depth AND excluded.knodes > positions.knodes)
            """,
            (source_name,),
        )
        after = int(connection.execute("SELECT COUNT(*) FROM positions").fetchone()[0])
        detach(connection)
        stats.append(
            {
                "database": str(database.resolve()),
                "source_name": source_name,
                "rows": source_rows,
                "mate_rows_excluded": mate_rows,
                "missing_cp_rows_excluded": missing_cp_rows,
                "eligible_cp_rows": eligible_rows,
                "old_fen_overlap_excluded": old_overlap,
                "duplicates_with_earlier_new_inputs": duplicate_new,
                "higher_quality_cross_input_updates": quality_updates,
                "new_unique_positions_added": after - before,
            }
        )
    connection.commit()
    return stats


def audit_old_inputs(
    connection: sqlite3.Connection, old_databases: Iterable[Path]
) -> tuple[list[dict[str, Any]], int]:
    connection.execute("DROP TABLE IF EXISTS temp.audit_old")
    connection.execute("CREATE TEMP TABLE audit_old(fen TEXT PRIMARY KEY) WITHOUT ROWID")
    stats: list[dict[str, Any]] = []
    for database in old_databases:
        attach(connection, database)
        rows = int(connection.execute("SELECT COUNT(*) FROM source_db.positions").fetchone()[0])
        before = int(connection.execute("SELECT COUNT(*) FROM audit_old").fetchone()[0])
        connection.execute("INSERT OR IGNORE INTO audit_old SELECT fen FROM source_db.positions")
        after = int(connection.execute("SELECT COUNT(*) FROM audit_old").fetchone()[0])
        detach(connection)
        stats.append(
            {
                "database": str(database.resolve()),
                "rows": rows,
                "new_unique_old_fens": after - before,
                "duplicates_with_earlier_old_inputs": rows - (after - before),
            }
        )
    unique = int(connection.execute("SELECT COUNT(*) FROM audit_old").fetchone()[0])
    stored = int(connection.execute("SELECT COUNT(*) FROM old_fens").fetchone()[0])
    if unique != stored:
        raise AssertionError(f"old exclusion audit {unique} != stored {stored}")
    connection.execute("DROP TABLE temp.audit_old")
    return stats, unique


def audit_new_inputs(
    connection: sqlite3.Connection, new_databases: Iterable[Path]
) -> list[dict[str, Any]]:
    connection.execute("DROP TABLE IF EXISTS temp.audit_new")
    connection.execute(
        "CREATE TEMP TABLE audit_new(fen TEXT PRIMARY KEY,depth INTEGER,knodes INTEGER) WITHOUT ROWID"
    )
    stats: list[dict[str, Any]] = []
    for database in new_databases:
        source_name = database.parent.name
        attach(connection, database)
        source_rows = int(connection.execute("SELECT COUNT(*) FROM source_db.positions").fetchone()[0])
        mate_rows = int(
            connection.execute("SELECT COUNT(*) FROM source_db.positions WHERE mate IS NOT NULL").fetchone()[0]
        )
        missing_cp_rows = int(
            connection.execute(
                "SELECT COUNT(*) FROM source_db.positions WHERE cp IS NULL AND mate IS NULL"
            ).fetchone()[0]
        )
        eligible_rows = int(
            connection.execute(
                "SELECT COUNT(*) FROM source_db.positions WHERE mate IS NULL AND cp IS NOT NULL"
            ).fetchone()[0]
        )
        old_overlap = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM source_db.positions AS source
                JOIN old_fens AS old ON old.fen=source.fen
                WHERE source.mate IS NULL AND source.cp IS NOT NULL
                """
            ).fetchone()[0]
        )
        duplicate_new = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM source_db.positions AS source
                JOIN audit_new AS current ON current.fen=source.fen
                WHERE source.mate IS NULL AND source.cp IS NOT NULL
                  AND NOT EXISTS (SELECT 1 FROM old_fens AS old WHERE old.fen=source.fen)
                """
            ).fetchone()[0]
        )
        quality_updates = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM source_db.positions AS source
                JOIN audit_new AS current ON current.fen=source.fen
                WHERE source.mate IS NULL AND source.cp IS NOT NULL
                  AND NOT EXISTS (SELECT 1 FROM old_fens AS old WHERE old.fen=source.fen)
                  AND (source.depth > current.depth OR
                       (source.depth = current.depth AND source.knodes > current.knodes))
                """
            ).fetchone()[0]
        )
        before = int(connection.execute("SELECT COUNT(*) FROM audit_new").fetchone()[0])
        connection.execute(
            """
            INSERT INTO audit_new(fen,depth,knodes)
            SELECT source.fen,source.depth,source.knodes
            FROM source_db.positions AS source
            WHERE source.mate IS NULL AND source.cp IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM old_fens AS old WHERE old.fen=source.fen)
              AND 1
            ON CONFLICT(fen) DO UPDATE SET depth=excluded.depth,knodes=excluded.knodes
            WHERE excluded.depth > audit_new.depth OR
                  (excluded.depth = audit_new.depth AND excluded.knodes > audit_new.knodes)
            """
        )
        after = int(connection.execute("SELECT COUNT(*) FROM audit_new").fetchone()[0])
        violations = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM source_db.positions AS source
                JOIN positions AS final ON final.fen=source.fen
                WHERE source.mate IS NULL AND source.cp IS NOT NULL
                  AND NOT EXISTS (SELECT 1 FROM old_fens AS old WHERE old.fen=source.fen)
                  AND (source.depth > final.depth OR
                       (source.depth = final.depth AND source.knodes > final.knodes))
                """
            ).fetchone()[0]
        )
        detach(connection)
        stats.append(
            {
                "database": str(database.resolve()),
                "source_name": source_name,
                "rows": source_rows,
                "mate_rows_excluded": mate_rows,
                "missing_cp_rows_excluded": missing_cp_rows,
                "eligible_cp_rows": eligible_rows,
                "old_fen_overlap_excluded": old_overlap,
                "duplicates_with_earlier_new_inputs": duplicate_new,
                "higher_quality_cross_input_updates": quality_updates,
                "new_unique_positions_added": after - before,
                "final_quality_order_violations": violations,
            }
        )
    audited = int(connection.execute("SELECT COUNT(*) FROM audit_new").fetchone()[0])
    stored = int(connection.execute("SELECT COUNT(*) FROM positions").fetchone()[0])
    if audited != stored:
        raise AssertionError(f"new-position audit {audited} != stored {stored}")
    connection.execute("DROP TABLE temp.audit_new")
    return stats


def rejection_reason(error: ValueError) -> str:
    message = str(error)
    for reason, marker in (
        ("malformed_side_to_move", "malformed FEN side-to-move"),
        ("malformed_board", "malformed FEN board"),
        ("malformed_piece_placement", "malformed FEN piece"),
        ("malformed_rank_width", "malformed FEN rank"),
        ("more_than_32_pieces", "too many pieces"),
        ("not_exactly_one_king_per_colour", "expected exactly one king"),
    ):
        if marker in message:
            return reason
    return "feature_encoding_error"


def filter_invalid_positions(connection: sqlite3.Connection) -> dict[str, Any]:
    state = connection.execute(
        "SELECT value FROM freeze_state WHERE key='validation_complete'"
    ).fetchone()
    if state is None:
        rejected_batch: list[tuple[str, str]] = []
        checked = 0
        for (fen,) in connection.execute("SELECT fen FROM positions WHERE valid=1 ORDER BY fen"):
            checked += 1
            try:
                encode_fen(fen)
            except ValueError as error:
                rejected_batch.append((fen, rejection_reason(error)))
            if len(rejected_batch) >= 1000:
                connection.executemany(
                    "INSERT OR REPLACE INTO rejected_positions(fen,reason) VALUES (?,?)",
                    rejected_batch,
                )
                connection.executemany(
                    "UPDATE positions SET valid=0 WHERE fen=?",
                    ((fen_value,) for fen_value, _ in rejected_batch),
                )
                connection.commit()
                rejected_batch.clear()
        if rejected_batch:
            connection.executemany(
                "INSERT OR REPLACE INTO rejected_positions(fen,reason) VALUES (?,?)",
                rejected_batch,
            )
            connection.executemany(
                "UPDATE positions SET valid=0 WHERE fen=?",
                ((fen_value,) for fen_value, _ in rejected_batch),
            )
        connection.execute(
            "INSERT OR REPLACE INTO freeze_state(key,value) VALUES ('validation_complete',?)",
            (str(checked),),
        )
        connection.commit()

    reasons = {
        str(reason): int(count)
        for reason, count in connection.execute(
            "SELECT reason,COUNT(*) FROM rejected_positions GROUP BY reason ORDER BY reason"
        )
    }
    raw_count = int(connection.execute("SELECT COUNT(*) FROM positions").fetchone()[0])
    valid_count = int(
        connection.execute("SELECT COUNT(*) FROM positions WHERE valid=1").fetchone()[0]
    )
    return {
        "raw_deduplicated_positions": raw_count,
        "valid_positions": valid_count,
        "invalid_positions_excluded": raw_count - valid_count,
        "reasons": reasons,
        "policy": "preserved in external rejection ledger with valid=0; absent from every split",
    }


def verify_manual_encoder(
    connection: sqlite3.Connection, count: int, repo_root: Path
) -> dict[str, Any]:
    import sys

    import chess

    sys.path.insert(0, str(repo_root))
    from nnue_lab.production.features import encode_board

    checked = 0
    for (fen,) in connection.execute(
        "SELECT fen FROM positions WHERE valid=1 ORDER BY fen LIMIT ?", (count,)
    ):
        manual, side = encode_fen(fen)
        board = chess.Board(fen if len(fen.split()) == 6 else f"{fen} 0 1")
        reference = encode_board(board)
        expected_side = 0 if board.turn == chess.WHITE else 1
        if side != expected_side or not np.array_equal(manual, reference):
            raise AssertionError(f"manual encoder disagrees with reference for {fen}")
        checked += 1
    return {"checked": checked, "mismatches": 0, "reference": "production.features.encode_board"}


@dataclass
class ExportStats:
    count: int = 0
    white_to_move: int = 0
    black_to_move: int = 0
    clipped_targets: int = 0
    target_sum: int = 0
    target_min: int | None = None
    target_max: int | None = None

    def add(self, side: int, target: int, clipped: bool) -> None:
        self.count += 1
        self.white_to_move += int(side == 0)
        self.black_to_move += int(side == 1)
        self.clipped_targets += int(clipped)
        self.target_sum += target
        self.target_min = target if self.target_min is None else min(self.target_min, target)
        self.target_max = target if self.target_max is None else max(self.target_max, target)

    def public(self, *, sealed: bool) -> dict[str, Any]:
        result: dict[str, Any] = {"count": self.count}
        if not sealed:
            result.update(
                {
                    "white_to_move": self.white_to_move,
                    "black_to_move": self.black_to_move,
                    "clipped_targets": self.clipped_targets,
                    "target_min_cp": self.target_min,
                    "target_max_cp": self.target_max,
                    "target_mean_cp": (
                        self.target_sum / self.count if self.count else None
                    ),
                }
            )
        return result


def save_npz(path: Path, indices: np.ndarray, sides: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    np.savez(path, indices=indices, sides=sides, targets=targets)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": int(indices.shape[0]),
    }


def export_split(
    connection: sqlite3.Connection,
    output_dir: Path,
    split_name: str,
    expected_count: int,
) -> dict[str, Any]:
    sealed = split_name == "pristine"
    stats = ExportStats()
    files: list[dict[str, Any]] = []
    fens_path: Path | None = None
    fens_handle: TextIO | None = None
    if split_name != "train":
        fens_path = output_dir / f"{split_name}_fens.jsonl.gz"
        fens_handle = gzip.open(fens_path, "wt", encoding="utf-8", newline="\n")

    shard_capacity = TRAIN_SHARD_ROWS if split_name == "train" else expected_count
    indices = np.empty((max(1, shard_capacity), 2, MAX_PIECES), dtype=np.uint16)
    sides = np.empty(max(1, shard_capacity), dtype=np.uint8)
    targets = np.empty(max(1, shard_capacity), dtype=np.int16)
    in_shard = 0
    shard_index = 0

    def flush() -> None:
        nonlocal in_shard, shard_index
        if in_shard == 0:
            return
        filename = (
            f"train_{shard_index:03d}.npz" if split_name == "train" else f"{split_name}.npz"
        )
        files.append(
            save_npz(
                output_dir / filename,
                indices[:in_shard],
                sides[:in_shard],
                targets[:in_shard],
            )
        )
        shard_index += 1
        in_shard = 0

    try:
        cursor = connection.execute(
            "SELECT fen,line,depth,knodes,cp,source_name,source_row FROM positions WHERE valid=1 ORDER BY fen"
        )
        for fen, pv_line, depth, knodes, cp, source_name, source_row in cursor:
            if split_for_fen(fen) != split_name:
                continue
            encoded, side = encode_fen(fen)
            target, clipped = side_to_move_target(cp, side)
            indices[in_shard] = encoded
            sides[in_shard] = side
            targets[in_shard] = target
            stats.add(side, target, clipped)
            if fens_handle is not None:
                first_move = str(pv_line).split()[0] if str(pv_line).split() else ""
                fens_handle.write(
                    json.dumps(
                        {
                            "row": stats.count - 1,
                            "fen": fen,
                            "depth": depth,
                            "knodes": knodes,
                            "first_move": first_move,
                            "source": source_name,
                            "source_row": source_row,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            in_shard += 1
            if in_shard == shard_capacity:
                flush()
        flush()
    finally:
        if fens_handle is not None:
            fens_handle.close()

    if stats.count != expected_count:
        raise AssertionError(
            f"{split_name} export count {stats.count} != expected {expected_count}"
        )
    result = stats.public(sealed=sealed)
    result["npz_files"] = files
    result["ordering"] = "lexicographic exact FEN"
    if fens_path is not None:
        result["fens"] = {
            "path": str(fens_path.resolve()),
            "bytes": fens_path.stat().st_size,
            "sha256": sha256_file(fens_path),
            "contains_teacher_targets": False,
        }
    if sealed:
        result["seal"] = (
            "count and cryptographic hashes only; no target distribution or model metric computed"
        )
    return result


def convert_bucketed_training_component(source: Path, output: Path) -> dict[str, Any]:
    with np.load(source) as data:
        if set(data.files) != {"indices", "sides", "targets"}:
            raise ValueError(f"unexpected arrays in {source}: {data.files}")
        old_indices = np.asarray(data["indices"], dtype=np.uint16)
        sides = np.asarray(data["sides"], dtype=np.uint8)
        targets = np.asarray(data["targets"], dtype=np.int16)
    if old_indices.ndim != 3 or old_indices.shape[1:] != (2, MAX_PIECES):
        raise ValueError(f"unexpected bucketed index shape: {old_indices.shape}")
    if sides.shape != (old_indices.shape[0],) or targets.shape != (old_indices.shape[0],):
        raise ValueError("bucketed component arrays have inconsistent lengths")
    if np.any(old_indices > 6144):
        raise ValueError("bucketed component contains an index above its padding row 6144")

    active = old_indices < 6144
    converted = np.full(old_indices.shape, PADDING_FEATURE, dtype=np.uint16)
    converted[active] = old_indices[active] % PADDING_FEATURE
    if np.any(converted[active] >= PADDING_FEATURE) or np.any(converted[~active] != 768):
        raise AssertionError("Chess768 conversion invariant failed")
    result = save_npz(output, converted, sides, targets)
    result.update(
        {
            "source_path": str(source.resolve()),
            "source_bytes": source.stat().st_size,
            "source_sha256": sha256_file(source),
            "conversion": "active bucketed row modulo 768; padding row 6144 maps to 768",
            "active_feature_entries": int(active.sum()),
            "padding_entries": int((~active).sum()),
            "quiet_count": 400_000,
            "general_nonquiet_count": 100_000,
            "composition_provenance": "existing final_mixed v0 manifest",
        }
    )
    return result


def input_inventory(
    json_paths: Iterable[Path], database_paths: Iterable[Path]
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for json_path, database_path in zip(json_paths, database_paths, strict=True):
        inventory.append(
            {
                "selected_jsonl": str(json_path.resolve()),
                "selected_jsonl_bytes": json_path.stat().st_size,
                "selected_jsonl_sha256": sha256_file(json_path),
                "operational_sqlite": str(database_path.resolve()),
                "operational_sqlite_bytes": database_path.stat().st_size,
                "operational_sqlite_sha256": sha256_file(database_path),
            }
        )
    return inventory


def split_membership(connection: sqlite3.Connection) -> tuple[dict[str, int], dict[str, str]]:
    counts: Counter[str] = Counter()
    digests = {name: hashlib.sha256() for name in SPLIT_NAMES}
    for (fen,) in connection.execute("SELECT fen FROM positions WHERE valid=1 ORDER BY fen"):
        name = split_for_fen(fen)
        counts[name] += 1
        canonical_fen_digest_update(digests[name], fen)
    return (
        {name: counts[name] for name in SPLIT_NAMES},
        {name: digests[name].hexdigest() for name in SPLIT_NAMES},
    )


def render_report(manifest: dict[str, Any]) -> str:
    counts = manifest["split_counts"]
    new_total = sum(counts.values())
    old = manifest["old_training_component"]
    datasets = manifest["datasets"]
    return f"""# Production data freeze

Frozen at `{manifest['created_utc']}` from the completed Stage-1 acquisitions only. This is a
**staging corpus, not the final acquisition ceiling**. No download or network operation is
implemented by the freeze script.

## Contents

- New unique, clean, non-mate FENs after old-data exclusion: **{new_total:,}**.
- Production training rows: **{counts['train']:,} new + {old['rows']:,} prior mixed = {counts['train'] + old['rows']:,}**.
- Validation: **{counts['validation']:,}**.
- Development: **{counts['development']:,}**.
- Pristine final test: **{counts['pristine']:,}**, sealed; no model metric or target-distribution
  statistic was computed during this freeze.
- Split membership is `BLAKE2b-64(exact FEN, key={SPLIT_SEED}) mod 10000`: 0-199 pristine,
  200-349 validation, 350-499 development, and 500-9999 train.

## Representation and target

- Shared Chess768 rows, fixed White and Black perspectives, maximum 32 pieces, padding row 768.
- Black perspective flips ranks (`square xor 56`) and swaps relative colours.
- The source `cp` is White-relative as established by the preserved v0 orientation proof.
- Training target is side-to-move relative and clipped to +/-{CLIP_CP} cp; mate rows are excluded.
- The prior 500k 80/20 quiet/general component was converted exactly by mapping active bucketed
  indices modulo 768 and mapping old padding row 6144 to new padding row 768.

## External artifacts

- Canonical FEN ledger: `{manifest['database']['path']}`
- Converted 500k component: `{old['path']}`
- New train shards: **{len(datasets['train']['npz_files'])}** files.
- Validation/development/pristine arrays and FEN sidecars live alongside those shards.

All external artifacts and exact inputs are SHA-256 pinned in `data_freeze_manifest.json`.
"""


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir.resolve()
    if output_dir.is_relative_to(repo_root):
        raise ValueError("bulky production data must remain outside the repository")
    database_path = output_dir / "frozen_positions.sqlite3"
    resumed = bool(args.resume)
    if output_dir.exists() and not resumed:
        raise FileExistsError(f"refusing to overwrite an existing freeze: {output_dir}")
    if resumed:
        if not database_path.is_file():
            raise FileNotFoundError(f"resume database not found: {database_path}")
        connection = connect_existing_database(database_path)
    else:
        output_dir.mkdir(parents=True)
        connection = connect_output_database(database_path)
    try:
        if not resumed:
            build_old_exclusion(connection, args.old_database)
            merge_new_positions(connection, args.new_database)
        old_merge, old_unique = audit_old_inputs(connection, args.old_database)
        new_merge = audit_new_inputs(connection, args.new_database)
        validation = filter_invalid_positions(connection)
        total_positions = int(
            connection.execute("SELECT COUNT(*) FROM positions WHERE valid=1").fetchone()[0]
        )
        if total_positions == 0:
            raise RuntimeError("no eligible new positions survived the freeze")
        encoder_check = verify_manual_encoder(connection, args.verify_encodings, repo_root)
        split_counts, split_hashes = split_membership(connection)

        old_component = convert_bucketed_training_component(
            args.bucketed_training_component,
            output_dir / "converted_baseline_train_500k.npz",
        )
        datasets: dict[str, dict[str, Any]] = {}
        for split_name in ("train", "validation", "development", "pristine"):
            datasets[split_name] = export_split(
                connection, output_dir, split_name, split_counts[split_name]
            )

        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise RuntimeError(f"frozen database integrity check failed: {integrity}")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()

    orientation_proof = repo_root / "nnue_lab" / "manifests" / "orientation_verification.json"
    baseline_preprocess = repo_root / "nnue_lab" / "manifests" / "preprocess.json"
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "stage": STAGE,
        "status": "staging corpus; deterministic append-compatible split, not final ceiling",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "network_used": False,
        "resumed_after_validation_edge_case": resumed,
        "source": {
            "dataset": "Lichess/chess-position-evaluations",
            "revision": "abb8f0b1251f89295a35b5ac801cb08a873812de",
            "license": "CC0-1.0",
            "old_exclusion_inputs": input_inventory(args.old_jsonl, args.old_database),
            "new_candidate_inputs": input_inventory(args.new_jsonl, args.new_database),
        },
        "selection": {
            "deduplication_key": "exact FEN string",
            "quality_order": "greatest depth, then greatest knodes; earlier input wins exact ties",
            "old_exclusion_unique_fens": old_unique,
            "old_merge": old_merge,
            "new_merge": new_merge,
            "feature_domain_validation": validation,
            "mate_policy": "excluded",
            "missing_cp_policy": "excluded",
        },
        "target": {
            "source_orientation": "White-relative cp",
            "runtime_orientation": "side-to-move relative",
            "formula": "clip(cp,-2000,2000) for White to move; negate after clipping for Black to move",
            "clip_cp": CLIP_CP,
            "orientation_proof": str(orientation_proof.resolve()),
            "orientation_proof_sha256": sha256_file(orientation_proof),
        },
        "features": {
            "vocabulary_rows": 768,
            "perspectives": ["fixed White", "fixed Black"],
            "shape": [2, 32],
            "dtype": "uint16",
            "padding_row": PADDING_FEATURE,
            "black_square_orientation": "rank flip: square xor 56",
            "relative_colour": "absolute colour xor perspective",
            "manual_encoder_check": encoder_check,
        },
        "split": {
            "algorithm": "BLAKE2b-64 keyed hash of exact FEN, little-endian integer mod 10000",
            "seed": SPLIT_SEED,
            "buckets": {
                "pristine": "0..199 (2.0%)",
                "validation": "200..349 (1.5%)",
                "development": "350..499 (1.5%)",
                "train": "500..9999 (95.0%)",
            },
            "membership_fen_sha256": split_hashes,
            "output_order": "lexicographic exact FEN",
        },
        "split_counts": split_counts,
        "old_training_component": old_component,
        "datasets": datasets,
        "database": {
            "path": str(database_path.resolve()),
            "bytes": database_path.stat().st_size,
            "sha256": sha256_file(database_path),
            "integrity_check": "ok",
            "role": "canonical ledger of excluded old FENs and deduplicated new FENs",
        },
        "baseline_preprocess_manifest": {
            "path": str(baseline_preprocess.resolve()),
            "sha256": sha256_file(baseline_preprocess),
        },
        "training_shuffle_seed_reserved": 20260911,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--old-database", type=Path, action="append", required=True)
    parser.add_argument("--new-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--new-database", type=Path, action="append", required=True)
    parser.add_argument("--bucketed-training-component", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--verify-encodings", type=int, default=10_000)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the newly-created external freeze ledger after an interrupted materialisation",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    lab_root = repo_root / "nnue_lab"
    if len(args.old_jsonl) != len(args.old_database):
        parser.error("old JSONL/database input counts differ")
    if len(args.new_jsonl) != len(args.new_database):
        parser.error("new JSONL/database input counts differ")
    for input_path in (
        args.old_jsonl
        + args.old_database
        + args.new_jsonl
        + args.new_database
        + [args.bucketed_training_component]
    ):
        if not input_path.is_file():
            parser.error(f"missing input: {input_path}")
    for small_output in (args.manifest, args.report):
        if not small_output.resolve().is_relative_to(lab_root):
            parser.error(f"small outputs must remain in nnue_lab: {small_output}")
    if args.output_dir.resolve().is_relative_to(repo_root):
        parser.error("large outputs must remain outside the repository")
    if args.verify_encodings <= 0:
        parser.error("--verify-encodings must be positive")
    return args


def main() -> None:
    args = parse_args()
    manifest = freeze(args)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(render_report(manifest), encoding="utf-8")
    print(json.dumps({"manifest": str(args.manifest), "split_counts": manifest["split_counts"], "elapsed_seconds": manifest["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
