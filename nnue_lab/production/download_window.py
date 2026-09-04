"""Acquire a bounded, disjoint later row-group window from pinned CC0 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    # The acquisition environment intentionally does not install Torch, which
    # production/__init__.py imports. Run this file directly, not with -m.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nnue_lab.download_data import (  # noqa: E402
    DATASET_PAGE,
    DATASET_REVISION,
    HARD_MAX_DOWNLOAD_BYTES,
    HARD_MAX_UNIQUE,
    LICENSE,
    DownloadCapExceeded,
    HttpRangeReader,
    OrderingAudit,
    Row,
    best_row,
    build_database,
    export_jsonl,
    insert_best,
    sha256_file,
)

FORMAT = "deepblue-nnue-data-row-group-window-v1"
DATA_ROOT = Path("C:/Users/mohib/AppData/Local/Temp/deepblue-nnue-data")
COLUMNS = ("fen", "line", "depth", "knodes", "cp", "mate")


def read_prior_chain(path: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Read only completed acquisition manifests, newest first."""
    result: list[tuple[Path, dict[str, Any]]] = []
    seen: set[Path] = set()
    source_file: str | None = None
    while True:
        path = path.resolve()
        if path in seen:
            raise ValueError("prior-manifest chain contains a cycle")
        seen.add(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("source_revision") != DATASET_REVISION:
            raise ValueError(f"unpinned or different source revision: {path}")
        if manifest.get("license") != LICENSE or manifest.get("source") != DATASET_PAGE:
            raise ValueError(f"unexpected source or license: {path}")
        url = str(manifest["source_file"])
        expected_prefix = f"{DATASET_PAGE}/resolve/{DATASET_REVISION}/data/data_"
        shard = url.removeprefix(expected_prefix).removesuffix(".parquet?download=true")
        if not url.startswith(expected_prefix) or len(shard) != 4 or not shard.isdigit():
            raise ValueError(f"unexpected pinned Parquet URL: {url}")
        if not 0 <= int(shard) < 20 or (source_file is not None and source_file != url):
            raise ValueError("prior chain must refer to one physical Lichess shard")
        source_file = url
        if int(manifest["unique_fens"]) != int(manifest["requested_unique_fens"]):
            raise ValueError(f"prior acquisition did not reach its requested boundary: {path}")
        if int(manifest.get("reappearing_fen_groups", -1)) != 0:
            raise ValueError(f"prior acquisition has no clean FEN-ordering audit: {path}")
        if not Path(manifest["database"]).is_file():
            raise FileNotFoundError(manifest["database"])
        result.append((path, manifest))
        previous = manifest.get("prior_manifest")
        if previous is None:
            break
        path = Path(previous["path"])
        if sha256_file(path) != previous["sha256"]:
            raise ValueError(f"prior manifest changed since acquisition: {path}")
    return result


def prior_end(manifest: dict[str, Any]) -> int:
    interval = manifest.get("source_row_interval")
    end = int(interval["end_exclusive"] if interval else manifest["raw_rows_scanned"])
    if end <= 0:
        raise ValueError("prior acquisition has no positive consumed row interval")
    return end


def row_group_layout(metadata: Any) -> list[dict[str, int]]:
    start = 0
    groups = []
    for index in range(metadata.num_row_groups):
        rows = int(metadata.row_group(index).num_rows)
        groups.append({"index": index, "start_inclusive": start, "end_exclusive": start + rows})
        start += rows
    if start != int(metadata.num_rows):
        raise ValueError("Parquet row-group counts do not match total rows")
    return groups


def next_row_group(groups: list[dict[str, int]], consumed_end: int) -> int:
    for group in groups:
        if group["start_inclusive"] >= consumed_end:
            return group["index"]
    raise ValueError("no entirely unconsumed row group remains after the prior window")


def select_window(
    connection: sqlite3.Connection,
    parquet_file: Any,
    groups: list[dict[str, int]],
    start_group: int,
    unique_limit: int,
    batch_rows: int,
) -> dict[str, Any]:
    """Select complete FEN runs, retaining absolute Parquet source row numbers."""
    unique_count = 0
    raw_rows = 0
    discarded_rows = 0
    reappearing = 0
    completed_groups = 0
    current_fen: str | None = None
    current_by_quality: dict[tuple[int, int], list[Row]] = defaultdict(list)
    first_run = True
    audit = OrderingAudit()
    requested_groups: list[int] = []
    lookahead_row: int | None = None
    start = groups[start_group]["start_inclusive"]
    stop = False

    def finish_group() -> None:
        nonlocal unique_count, discarded_rows, reappearing, completed_groups, first_run
        if current_fen is None:
            return
        if first_run:
            # The group boundary can cut a FEN run started in an earlier group.
            discarded_rows = sum(len(rows) for rows in current_by_quality.values())
            first_run = False
            return
        audit.observe(current_fen, current_by_quality)
        if insert_best(connection, best_row(current_by_quality)):
            unique_count += 1
        else:
            reappearing += 1
        completed_groups += 1
        if completed_groups % 5000 == 0:
            connection.commit()

    for group in groups[start_group:]:
        requested_groups.append(group["index"])
        group_offset = 0
        batches = parquet_file.iter_batches(
            batch_size=batch_rows, columns=COLUMNS, row_groups=[group["index"]]
        )
        for batch in batches:
            values = {name: batch.column(name).to_pylist() for name in COLUMNS}
            for offset in range(batch.num_rows):
                absolute_row = group["start_inclusive"] + group_offset
                fen = str(values["fen"][offset])
                if current_fen is not None and fen != current_fen:
                    finish_group()
                    current_fen = None
                    current_by_quality = defaultdict(list)
                    if unique_count >= unique_limit:
                        lookahead_row = absolute_row
                        stop = True
                        break
                current_fen = fen
                row = Row(
                    row_index=absolute_row,
                    fen=fen,
                    line=str(values["line"][offset]),
                    depth=int(values["depth"][offset]),
                    knodes=int(values["knodes"][offset]),
                    cp=None if values["cp"][offset] is None else int(values["cp"][offset]),
                    mate=None if values["mate"][offset] is None else int(values["mate"][offset]),
                )
                current_by_quality[row.quality].append(row)
                raw_rows += 1
                group_offset += 1
            if stop:
                break
        if stop:
            break
        if group_offset != group["end_exclusive"] - group["start_inclusive"]:
            raise RuntimeError("Parquet iterator did not yield its complete row group")
    if not stop:
        # Physical shard EOF may cut a FEN run continued in the next shard.
        # Never finalize that unproven final run.
        raise RuntimeError(f"shard ended before {unique_limit} complete FEN runs were selected")
    connection.commit()
    if reappearing:
        raise RuntimeError(f"observed {reappearing} non-contiguous FEN groups; ordering not safe")
    return {
        "raw_rows_scanned": raw_rows,
        "unique_fens": unique_count,
        "reappearing_fen_groups": reappearing,
        "discarded_incomplete_boundary_groups": 1,
        "discarded_first_fen_rows": discarded_rows,
        "source_row_interval": {"start_inclusive": start, "end_exclusive": start + raw_rows},
        "boundary_lookahead_row": lookahead_row,
        "parquet_row_groups_requested": requested_groups,
        "stopped_at_complete_fen_boundary": True,
        "ordering_audit": audit.as_dict(),
    }


def overlap_with_prior(database: Path, prior_database: Path) -> int:
    connection = sqlite3.connect(prior_database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("ATTACH DATABASE ? AS new_window", (database.resolve().as_uri() + "?mode=ro",))
        return int(connection.execute(
            "SELECT COUNT(*) FROM new_window.positions AS current "
            "JOIN main.positions AS prior ON prior.fen=current.fen"
        ).fetchone()[0])
    finally:
        connection.close()


def acquire(args: argparse.Namespace) -> dict[str, Any]:
    import pyarrow.parquet as parquet  # type: ignore[import-not-found]

    chain = read_prior_chain(args.prior_manifest)
    previous_path, previous = chain[0]
    output_dir = args.output_dir.resolve()
    if output_dir.exists() or args.manifest.exists():
        raise FileExistsError("refusing to overwrite an existing output directory or manifest")
    reader = HttpRangeReader(
        previous["source_file"], max_download_bytes=args.max_download_bytes,
        block_size=args.block_mib * 1024 * 1024, cached_blocks=args.cached_blocks,
    )
    if reader.size != int(previous["remote_file_bytes"]):
        raise ValueError("pinned remote file size differs from prior acquisition")
    if previous.get("source_etag") and reader.etag != previous["source_etag"]:
        raise ValueError("pinned remote ETag differs from prior acquisition")
    parquet_file = parquet.ParquetFile(reader)
    groups = row_group_layout(parquet_file.metadata)
    consumed_end = prior_end(previous)
    start_group = next_row_group(groups, consumed_end)
    layout_json = json.dumps(groups, sort_keys=True, separators=(",", ":")).encode()
    print(json.dumps({"event": "window_start", "prior_end_exclusive": consumed_end,
                      "start_row_group": start_group, **groups[start_group]}), flush=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    database = output_dir / "selected.sqlite3"
    jsonl = output_dir / "selected.jsonl"
    connection = build_database(database)
    started = time.perf_counter()
    try:
        selection = select_window(connection, parquet_file, groups, start_group,
                                  args.unique_fens, args.batch_rows)
        audit = selection["ordering_audit"]
        if (int(audit["multipv_groups_checked"]) < min(100, args.unique_fens // 2)
                or float(audit["first_is_stm_best_rate"]) < 0.995
                or float(audit["fully_monotonic_rate"]) < 0.98):
            raise RuntimeError(f"MultiPV ordering audit failed: {audit}")
        exported = export_jsonl(connection, jsonl)
        if exported != args.unique_fens or exported != selection["unique_fens"]:
            raise RuntimeError("selected/exported FEN count does not match requested count")
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"output SQLite integrity failure: {integrity}")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
        reader.session.close()
    overlaps = []
    for manifest_path, prior in chain:
        overlap = overlap_with_prior(database, Path(prior["database"]))
        overlaps.append({"manifest": str(manifest_path), "database": prior["database"],
                         "overlapping_fens": overlap})
        if overlap:
            raise RuntimeError(f"window overlaps prior acquisition by {overlap} FENs: {manifest_path}")
    elapsed = time.perf_counter() - started
    return {
        "format": FORMAT, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": DATASET_PAGE, "source_revision": DATASET_REVISION, "license": LICENSE,
        "source_file": previous["source_file"], "remote_file_bytes": reader.size,
        "source_etag": reader.etag, "source_last_modified": reader.last_modified,
        "downloaded_bytes": reader.downloaded_bytes, "http_range_requests": reader.http_requests,
        "download_cap_bytes": args.max_download_bytes,
        "prior_manifest": {"path": str(previous_path), "sha256": sha256_file(previous_path)},
        "prior_end_exclusive": consumed_end,
        "skipped_rows_before_window": groups[start_group]["start_inclusive"] - consumed_end,
        "parquet_total_rows": int(parquet_file.metadata.num_rows),
        "parquet_total_row_groups": len(groups), "row_group_layout_sha256": hashlib.sha256(layout_json).hexdigest(),
        "requested_row_group_intervals": [groups[index] for index in selection["parquet_row_groups_requested"]],
        **selection, "requested_unique_fens": args.unique_fens,
        "selection": "max(depth, then knodes), first row within complete analysis/MultiPV group",
        "prior_fen_overlap_checks": overlaps,
        "selected_jsonl": str(jsonl), "selected_jsonl_bytes": jsonl.stat().st_size,
        "selected_jsonl_sha256": sha256_file(jsonl), "database": str(database),
        "database_bytes": database.stat().st_size, "database_sha256": sha256_file(database),
        "database_integrity_check": "ok", "elapsed_seconds": elapsed,
        "rows_per_second": selection["raw_rows_scanned"] / max(elapsed, 1e-9),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--unique-fens", type=int, default=1_000_000)
    parser.add_argument("--max-download-bytes", type=int, default=900_000_000)
    parser.add_argument("--block-mib", type=int, default=8)
    parser.add_argument("--cached-blocks", type=int, default=16)
    parser.add_argument("--batch-rows", type=int, default=65_536)
    args = parser.parse_args()
    if not 1 <= args.unique_fens <= HARD_MAX_UNIQUE:
        parser.error(f"--unique-fens must be in [1,{HARD_MAX_UNIQUE}]")
    if not 1 <= args.max_download_bytes <= HARD_MAX_DOWNLOAD_BYTES:
        parser.error(f"--max-download-bytes must be in [1,{HARD_MAX_DOWNLOAD_BYTES}]")
    if min(args.block_mib, args.cached_blocks, args.batch_rows) < 1:
        parser.error("block, cache and batch sizes must be positive")
    output = args.output_dir.resolve()
    if output == DATA_ROOT.resolve() or not output.is_relative_to(DATA_ROOT.resolve()):
        parser.error(f"data output must be a new child directory of {DATA_ROOT}")
    lab = Path(__file__).resolve().parents[1]
    if not args.manifest.resolve().is_relative_to(lab):
        parser.error("small manifest must stay inside nnue_lab")
    return args


def main() -> int:
    args = parse_args()
    try:
        manifest = acquire(args)
    except DownloadCapExceeded as error:
        print(f"download cap reached safely: {error}", file=sys.stderr)
        return 2
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
