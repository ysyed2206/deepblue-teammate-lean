"""Deduplicate, validate, quiet-filter, split, encode, and materialise NNUE data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import chess
import numpy as np

from nnue_lab.features import encode_board, normalize_fen, split_for_fen, stable_hash64, stm_target

HASH_SEED = 20260901
SAMPLE_SEED = 20260902


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(
        """
        DROP TABLE IF EXISTS raw_positions;
        DROP TABLE IF EXISTS clean_positions;
        CREATE TABLE raw_positions (
            fen TEXT PRIMARY KEY,
            line TEXT NOT NULL,
            depth INTEGER NOT NULL,
            knodes INTEGER NOT NULL,
            cp INTEGER,
            mate INTEGER
        );
        CREATE TABLE clean_positions (
            fen TEXT PRIMARY KEY,
            sample_hash TEXT NOT NULL,
            split TEXT NOT NULL,
            quiet INTEGER NOT NULL,
            side INTEGER NOT NULL,
            target INTEGER NOT NULL,
            encoded BLOB NOT NULL,
            depth INTEGER NOT NULL,
            knodes INTEGER NOT NULL,
            first_move TEXT NOT NULL
        );
        CREATE INDEX clean_split_hash ON clean_positions(split, sample_hash);
        CREATE INDEX clean_split_quiet_hash ON clean_positions(split, quiet, sample_hash);
        """
    )
    connection.commit()
    return connection


def merge_inputs(connection: sqlite3.Connection, input_paths: Iterable[Path]) -> dict[str, int]:
    rows = 0
    inserted = 0
    improved = 0
    for input_path in input_paths:
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                fen = str(record["fen"])
                depth = int(record["depth"])
                knodes = int(record["knodes"])
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO raw_positions VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        fen,
                        str(record["line"]),
                        depth,
                        knodes,
                        record.get("cp"),
                        record.get("mate"),
                    ),
                )
                if cursor.rowcount == 1:
                    inserted += 1
                else:
                    update = connection.execute(
                        """
                        UPDATE raw_positions SET line=?,depth=?,knodes=?,cp=?,mate=?
                        WHERE fen=? AND (depth < ? OR (depth = ? AND knodes < ?))
                        """,
                        (
                            str(record["line"]),
                            depth,
                            knodes,
                            record.get("cp"),
                            record.get("mate"),
                            fen,
                            depth,
                            depth,
                            knodes,
                        ),
                    )
                    improved += int(update.rowcount == 1)
                rows += 1
                if rows % 10_000 == 0:
                    connection.commit()
    connection.commit()
    return {"input_rows": rows, "unique_fens": inserted, "cross_input_quality_updates": improved}


def quiet_reasons(board: chess.Board, move: chess.Move) -> list[str]:
    reasons: list[str] = []
    if board.is_check():
        reasons.append("side_to_move_in_check")
    if board.is_capture(move):
        reasons.append("best_move_capture")
    if move.promotion is not None:
        reasons.append("best_move_promotion")
    if board.gives_check(move):
        reasons.append("best_move_gives_check")
    return reasons


def validate_and_encode(
    connection: sqlite3.Connection, clip_cp: int
) -> tuple[Counter[str], Counter[str], Counter[str]]:
    rejected: Counter[str] = Counter()
    quiet_all_reasons: Counter[str] = Counter()
    quiet_primary_reasons: Counter[str] = Counter()
    processed = 0
    cursor = connection.execute("SELECT fen,line,depth,knodes,cp,mate FROM raw_positions")
    for fen, pv_line, depth, knodes, cp, mate in cursor:
        processed += 1
        if mate is not None:
            rejected["mate_teacher"] += 1
            continue
        if cp is None:
            rejected["missing_cp"] += 1
            continue
        try:
            board = chess.Board(normalize_fen(fen))
        except ValueError:
            rejected["invalid_fen_syntax"] += 1
            continue
        if not board.is_valid():
            rejected[f"invalid_board_status_{int(board.status())}"] += 1
            continue
        tokens = str(pv_line).split()
        if not tokens:
            rejected["empty_pv"] += 1
            continue
        try:
            # Lichess uses UCI_Chess960-compatible castling (e1h1/e1a1).  parse_uci
            # accepts both that form and ordinary king-destination UCI.
            move = board.parse_uci(tokens[0])
        except ValueError:
            rejected["illegal_or_malformed_pv_first_move"] += 1
            continue
        try:
            encoded = encode_board(board)
        except ValueError:
            rejected["feature_encoding_failure"] += 1
            continue
        reasons = quiet_reasons(board, move)
        quiet = not reasons
        if quiet:
            quiet_primary_reasons["accepted_quiet"] += 1
        else:
            quiet_primary_reasons[reasons[0]] += 1
            quiet_all_reasons.update(reasons)
        unclipped_stm = int(cp) if board.turn == chess.WHITE else -int(cp)
        target = stm_target(int(cp), board, clip_cp=clip_cp)
        if target != unclipped_stm:
            rejected["cp_clipped_not_rejected"] += 1
        side = 0 if board.turn == chess.WHITE else 1
        split = split_for_fen(fen, HASH_SEED)
        sample_hash = f"{stable_hash64(fen, SAMPLE_SEED):016x}"
        connection.execute(
            "INSERT INTO clean_positions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                fen,
                sample_hash,
                split,
                int(quiet),
                side,
                target,
                encoded.tobytes(order="C"),
                depth,
                knodes,
                tokens[0],
            ),
        )
        if processed % 5000 == 0:
            connection.commit()
    connection.commit()
    return rejected, quiet_all_reasons, quiet_primary_reasons


def query_count(connection: sqlite3.Connection, query: str) -> int:
    return int(connection.execute(f"SELECT COUNT(*) FROM ({query})").fetchone()[0])


def export_npz(
    connection: sqlite3.Connection,
    output_path: Path,
    query: str,
    *,
    fens_path: Path | None = None,
) -> dict[str, Any]:
    count = query_count(connection, query)
    indices = np.empty((count, 2, 32), dtype=np.uint16)
    sides = np.empty(count, dtype=np.uint8)
    targets = np.empty(count, dtype=np.int16)
    quiet_flags = np.empty(count, dtype=np.uint8)
    fen_handle = None if fens_path is None else fens_path.open("w", encoding="utf-8", newline="\n")
    try:
        for index, row in enumerate(connection.execute(query)):
            fen, encoded, side, target, quiet, depth, knodes, first_move = row
            indices[index] = np.frombuffer(encoded, dtype=np.uint16).reshape(2, 32)
            sides[index] = side
            targets[index] = target
            quiet_flags[index] = quiet
            if fen_handle is not None:
                fen_handle.write(
                    json.dumps(
                        {
                            "fen": fen,
                            "target_cp": target,
                            "side": side,
                            "quiet": bool(quiet),
                            "depth": depth,
                            "knodes": knodes,
                            "first_move": first_move,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    finally:
        if fen_handle is not None:
            fen_handle.close()
    np.savez(output_path, indices=indices, sides=sides, targets=targets)
    result: dict[str, Any] = {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "bytes": output_path.stat().st_size,
        "count": count,
        "quiet_count": int(quiet_flags.sum()),
        "general_nonquiet_count": int(count - quiet_flags.sum()),
        "target_min_cp": int(targets.min()) if count else None,
        "target_max_cp": int(targets.max()) if count else None,
        "target_mean_cp": float(targets.mean()) if count else None,
    }
    if fens_path is not None:
        result["fens_path"] = str(fens_path)
        result["fens_sha256"] = sha256_file(fens_path)
    return result


def limited_query(where: str, limit: int) -> str:
    return (
        "SELECT fen,encoded,side,target,quiet,depth,knodes,first_move "
        f"FROM clean_positions WHERE {where} ORDER BY sample_hash LIMIT {int(limit)}"
    )


def mixed_query(total: int) -> str:
    quiet_count = (total * 4) // 5
    general_count = total - quiet_count
    return f"""
        SELECT fen,encoded,side,target,quiet,depth,knodes,first_move FROM (
            SELECT * FROM ({limited_query("split='train' AND quiet=1", quiet_count)})
            UNION ALL
            SELECT * FROM ({limited_query("split='train' AND quiet=0", general_count)})
        ) ORDER BY fen
    """


def available_mixed_count(connection: sqlite3.Connection, requested: int) -> int:
    quiet = int(
        connection.execute(
            "SELECT COUNT(*) FROM clean_positions WHERE split='train' AND quiet=1"
        ).fetchone()[0]
    )
    nonquiet = int(
        connection.execute(
            "SELECT COUNT(*) FROM clean_positions WHERE split='train' AND quiet=0"
        ).fetchone()[0]
    )
    possible = min(requested, (quiet * 5) // 4, nonquiet * 5)
    return (possible // 5) * 5


def preprocess(args: argparse.Namespace) -> dict[str, Any]:
    proof = json.loads(args.orientation_proof.read_text(encoding="utf-8"))
    if proof.get("conclusion") != "white-relative":
        raise RuntimeError("orientation proof does not establish White-relative cp")
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    database_path = args.output_dir / "processed.sqlite3"
    connection = build_database(database_path)
    merge_stats = merge_inputs(connection, args.input)
    rejected, quiet_all, quiet_primary = validate_and_encode(connection, args.clip_cp)
    counts = {
        split: int(
            connection.execute(
                "SELECT COUNT(*) FROM clean_positions WHERE split=?", (split,)
            ).fetchone()[0]
        )
        for split in ("train", "validation", "test")
    }
    counts["clean_total"] = sum(counts.values())
    counts["quiet_total"] = int(
        connection.execute("SELECT COUNT(*) FROM clean_positions WHERE quiet=1").fetchone()[0]
    )

    pilot_general_n = min(args.pilot_train, counts["train"])
    pilot_mixed_n = available_mixed_count(connection, args.pilot_train)
    final_general_n = min(args.final_train, counts["train"])
    final_mixed_n = available_mixed_count(connection, args.final_train)
    validation_n = min(args.heldout, counts["validation"])
    test_n = min(args.heldout, counts["test"])

    datasets: dict[str, dict[str, Any]] = {}
    datasets["pilot_general"] = export_npz(
        connection,
        args.output_dir / "pilot_general.npz",
        limited_query("split='train'", pilot_general_n),
    )
    datasets["pilot_mixed"] = export_npz(
        connection, args.output_dir / "pilot_mixed.npz", mixed_query(pilot_mixed_n)
    )
    datasets["final_general"] = export_npz(
        connection,
        args.output_dir / "final_general.npz",
        limited_query("split='train'", final_general_n),
    )
    datasets["final_mixed"] = export_npz(
        connection, args.output_dir / "final_mixed.npz", mixed_query(final_mixed_n)
    )
    datasets["validation"] = export_npz(
        connection,
        args.output_dir / "validation.npz",
        limited_query("split='validation'", validation_n),
        fens_path=args.output_dir / "validation_fens.jsonl",
    )
    datasets["test"] = export_npz(
        connection,
        args.output_dir / "test.npz",
        limited_query("split='test'", test_n),
        fens_path=args.output_dir / "test_fens.jsonl",
    )
    connection.close()
    elapsed = time.perf_counter() - started
    return {
        "format": "deepblue-nnue-preprocess-manifest-v0",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_inputs": [str(path.resolve()) for path in args.input],
        "source_input_sha256": {
            str(path.resolve()): sha256_file(path) for path in args.input
        },
        "orientation_proof": str(args.orientation_proof.resolve()),
        "orientation_proof_sha256": sha256_file(args.orientation_proof),
        "target": "side-to-move relative Stockfish cp, clipped before sign conversion",
        "clip_cp": args.clip_cp,
        "split": "BLAKE2b-64(FEN,key=20260901): 95% train, 2.5% validation, 2.5% test",
        "sample_order": "BLAKE2b-64(FEN,key=20260902)",
        "merge": merge_stats,
        "counts": counts,
        "clean_rejections_and_notes": dict(rejected),
        "quiet_filter": {
            "definition": (
                "not in check; legal PV first move; first move is not capture, promotion, or check"
            ),
            "all_disqualifying_reasons_overlapping": dict(quiet_all),
            "primary_disqualifying_reason_priority_order": dict(quiet_primary),
        },
        "datasets": datasets,
        "database": str(database_path),
        "elapsed_seconds": elapsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--orientation-proof", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pilot-train", type=int, default=200_000)
    parser.add_argument("--final-train", type=int, default=500_000)
    parser.add_argument("--heldout", type=int, default=20_000)
    parser.add_argument("--clip-cp", type=int, default=2000)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    if args.output_dir.resolve().is_relative_to(repo_root):
        parser.error("derived datasets must be written outside the repository")
    if not args.manifest.resolve().is_relative_to(repo_root / "nnue_lab"):
        parser.error("small manifest must stay inside nnue_lab")
    sizes = (args.pilot_train, args.final_train, args.heldout, args.clip_cp)
    if any(value <= 0 for value in sizes):
        parser.error("dataset sizes and clip must be positive")
    return args


def main() -> None:
    args = parse_args()
    manifest = preprocess(args)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
