"""Range-read a capped prefix of the Lichess evaluations Parquet dataset.

This script is intentionally run from a temporary data-processing environment
containing ``requests`` and ``pyarrow``.  It never downloads a complete shard.
HTTP byte ranges are counted at the reader boundary and a hard cap aborts the
run before another block would exceed the allowance.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sqlite3
import sys
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

DATASET_URL = (
    "https://huggingface.co/datasets/Lichess/chess-position-evaluations/"
    "resolve/abb8f0b1251f89295a35b5ac801cb08a873812de/"
    "data/data_0000.parquet?download=true"
)
DATASET_PAGE = "https://huggingface.co/datasets/Lichess/chess-position-evaluations"
DATASET_REVISION = "abb8f0b1251f89295a35b5ac801cb08a873812de"
LICENSE = "CC0-1.0"
HARD_MAX_UNIQUE = 1_000_000
HARD_MAX_DOWNLOAD_BYTES = 2_000_000_000


class DownloadCapExceeded(RuntimeError):
    """Raised before the next HTTP range would cross the configured cap."""


class HttpRangeReader(io.RawIOBase):
    """Seekable, block-cached HTTP reader with exact transferred-byte accounting."""

    def __init__(
        self,
        url: str,
        *,
        max_download_bytes: int,
        block_size: int = 8 * 1024 * 1024,
        cached_blocks: int = 16,
    ) -> None:
        import requests  # type: ignore[import-untyped]

        super().__init__()
        self.url = url
        self.max_download_bytes = max_download_bytes
        self.block_size = block_size
        self.cached_blocks = cached_blocks
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "DeepBlue-NNUE-lab/0 (bounded research range reader)",
                "Accept-Encoding": "identity",
            }
        )
        self.size, self.etag, self.last_modified = self._discover_metadata()
        self.position = 0
        self.downloaded_bytes = 0
        self.http_requests = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()

    def _discover_metadata(self) -> tuple[int, str | None, str | None]:
        response = self.session.head(self.url, allow_redirects=True, timeout=60)
        response.raise_for_status()
        length = response.headers.get("Content-Length")
        etag = response.headers.get("ETag")
        modified = response.headers.get("Last-Modified")
        if length is not None and int(length) > 1:
            return int(length), etag, modified
        response.close()
        probe = self.session.get(
            self.url,
            headers={"Range": "bytes=0-0"},
            allow_redirects=True,
            stream=True,
            timeout=60,
        )
        if probe.status_code != 206:
            probe.close()
            raise RuntimeError("server did not honour a one-byte HTTP Range probe")
        content_range = probe.headers.get("Content-Range", "")
        probe.close()
        if "/" not in content_range:
            raise RuntimeError("range response omitted total size")
        return int(content_range.rsplit("/", 1)[1]), etag, modified

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = min(position, self.size)
        return self.position

    def _fetch_block(self, block_number: int) -> bytes:
        cached = self.cache.pop(block_number, None)
        if cached is not None:
            self.cache[block_number] = cached
            return cached
        start = block_number * self.block_size
        end = min(start + self.block_size, self.size) - 1
        requested = end - start + 1
        if self.downloaded_bytes + requested > self.max_download_bytes:
            raise DownloadCapExceeded(
                f"next {requested}-byte range would exceed {self.max_download_bytes} bytes"
            )
        response = self.session.get(
            self.url,
            headers={"Range": f"bytes={start}-{end}"},
            allow_redirects=True,
            stream=True,
            timeout=120,
        )
        self.http_requests += 1
        if response.status_code != 206:
            response.close()
            raise RuntimeError(
                f"server returned HTTP {response.status_code}; refusing an unbounded full response"
            )
        data = bytes(response.content)
        response.close()
        if len(data) != requested:
            raise RuntimeError(f"short range read: requested {requested}, received {len(data)}")
        self.downloaded_bytes += len(data)
        self.cache[block_number] = data
        while len(self.cache) > self.cached_blocks:
            self.cache.popitem(last=False)
        return data

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        result = bytearray()
        remaining = size
        while remaining:
            block_number = self.position // self.block_size
            within = self.position % self.block_size
            block = self._fetch_block(block_number)
            take = min(remaining, len(block) - within)
            result.extend(block[within : within + take])
            self.position += take
            remaining -= take
        return bytes(result)

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


@dataclass
class Row:
    row_index: int
    fen: str
    line: str
    depth: int
    knodes: int
    cp: int | None
    mate: int | None

    @property
    def quality(self) -> tuple[int, int]:
        return self.depth, self.knodes


class OrderingAudit:
    def __init__(self) -> None:
        self.multipv_groups = 0
        self.first_is_best = 0
        self.monotonic = 0
        self.white_groups = 0
        self.black_groups = 0

    def observe(self, fen: str, rows_by_quality: dict[tuple[int, int], list[Row]]) -> None:
        white_to_move = fen.split()[1] == "w"
        for rows in rows_by_quality.values():
            cp_values = [row.cp for row in rows]
            if len(cp_values) < 2 or any(value is None for value in cp_values):
                continue
            values = [int(value) for value in cp_values if value is not None]
            self.multipv_groups += 1
            if white_to_move:
                self.white_groups += 1
                if values[0] == max(values):
                    self.first_is_best += 1
                if all(left >= right for left, right in pairwise(values)):
                    self.monotonic += 1
            else:
                self.black_groups += 1
                if values[0] == min(values):
                    self.first_is_best += 1
                if all(left <= right for left, right in pairwise(values)):
                    self.monotonic += 1

    def as_dict(self) -> dict[str, int | float]:
        denominator = max(1, self.multipv_groups)
        return {
            "multipv_groups_checked": self.multipv_groups,
            "white_groups": self.white_groups,
            "black_groups": self.black_groups,
            "first_is_stm_best_count": self.first_is_best,
            "first_is_stm_best_rate": self.first_is_best / denominator,
            "fully_monotonic_count": self.monotonic,
            "fully_monotonic_rate": self.monotonic / denominator,
        }


def best_row(rows_by_quality: dict[tuple[int, int], list[Row]]) -> Row:
    """Deepest analysis, then largest node count, then first/PV1 row."""
    best_quality = max(rows_by_quality)
    return rows_by_quality[best_quality][0]


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
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS positions (
            fen TEXT PRIMARY KEY,
            line TEXT NOT NULL,
            depth INTEGER NOT NULL,
            knodes INTEGER NOT NULL,
            cp INTEGER,
            mate INTEGER,
            first_row INTEGER NOT NULL
        )
        """
    )
    connection.execute("DELETE FROM positions")
    connection.commit()
    return connection


def insert_best(connection: sqlite3.Connection, row: Row) -> bool:
    cursor = connection.execute(
        "INSERT OR IGNORE INTO positions VALUES (?, ?, ?, ?, ?, ?, ?)",
        (row.fen, row.line, row.depth, row.knodes, row.cp, row.mate, row.row_index),
    )
    inserted = cursor.rowcount == 1
    if not inserted:
        connection.execute(
            """
            UPDATE positions
            SET line=?, depth=?, knodes=?, cp=?, mate=?, first_row=?
            WHERE fen=? AND (depth < ? OR (depth = ? AND knodes < ?))
            """,
            (
                row.line,
                row.depth,
                row.knodes,
                row.cp,
                row.mate,
                row.row_index,
                row.fen,
                row.depth,
                row.depth,
                row.knodes,
            ),
        )
    return inserted


def export_jsonl(connection: sqlite3.Connection, output_path: Path) -> int:
    count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        cursor = connection.execute(
            "SELECT fen,line,depth,knodes,cp,mate,first_row FROM positions ORDER BY first_row"
        )
        for fen, line, depth, knodes, cp, mate, first_row in cursor:
            handle.write(
                json.dumps(
                    {
                        "fen": fen,
                        "line": line,
                        "depth": depth,
                        "knodes": knodes,
                        "cp": cp,
                        "mate": mate,
                        "source_row": first_row,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
    return count


def acquire(args: argparse.Namespace) -> dict[str, Any]:
    import pyarrow.parquet as parquet  # type: ignore[import-not-found]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "selected.sqlite3"
    jsonl_path = output_dir / "selected.jsonl"
    connection = build_database(database_path)
    reader = HttpRangeReader(
        args.url,
        max_download_bytes=args.max_download_bytes,
        block_size=args.block_mib * 1024 * 1024,
        cached_blocks=args.cached_blocks,
    )
    parquet_file = parquet.ParquetFile(reader)
    columns = ("fen", "line", "depth", "knodes", "cp", "mate")
    unique_count = 0
    raw_rows = 0
    reappearing_groups = 0
    discarded_boundary_groups = 0
    commit_counter = 0
    audit = OrderingAudit()
    current_fen: str | None = None
    current_by_quality: dict[tuple[int, int], list[Row]] = defaultdict(list)
    started = time.perf_counter()
    stop = False

    def finish_group() -> None:
        nonlocal unique_count, reappearing_groups, commit_counter, discarded_boundary_groups
        if current_fen is None or not current_by_quality:
            return
        if args.discard_first_fen and discarded_boundary_groups == 0 and unique_count == 0:
            # A Parquet shard may begin in the middle of a FEN run continued
            # from the previous shard.  Never train on that incomplete run.
            discarded_boundary_groups = 1
            return
        audit.observe(current_fen, current_by_quality)
        selected = best_row(current_by_quality)
        if insert_best(connection, selected):
            unique_count += 1
        else:
            reappearing_groups += 1
        commit_counter += 1
        if commit_counter >= 5000:
            connection.commit()
            commit_counter = 0

    try:
        batches = parquet_file.iter_batches(batch_size=args.batch_rows, columns=columns)
        for batch in batches:
            values = {name: batch.column(name).to_pylist() for name in columns}
            for offset in range(batch.num_rows):
                fen = str(values["fen"][offset])
                if current_fen is not None and fen != current_fen:
                    finish_group()
                    current_fen = None
                    current_by_quality = defaultdict(list)
                    if unique_count >= args.unique_fens:
                        stop = True
                        break
                current_fen = fen
                row = Row(
                    row_index=raw_rows,
                    fen=fen,
                    line=str(values["line"][offset]),
                    depth=int(values["depth"][offset]),
                    knodes=int(values["knodes"][offset]),
                    cp=None if values["cp"][offset] is None else int(values["cp"][offset]),
                    mate=(
                        None if values["mate"][offset] is None else int(values["mate"][offset])
                    ),
                )
                current_by_quality[row.quality].append(row)
                raw_rows += 1
                if args.raw_row_limit and raw_rows >= args.raw_row_limit:
                    stop = True
                    break
            if stop:
                break
        finish_group()
    finally:
        connection.commit()

    ordering_audit = audit.as_dict()
    if reappearing_groups != 0:
        connection.close()
        raise RuntimeError(
            f"observed {reappearing_groups} non-contiguous FEN groups; ordering not safe"
        )
    if (
        int(ordering_audit["multipv_groups_checked"]) < min(100, unique_count // 2)
        or float(ordering_audit["first_is_stm_best_rate"]) < 0.995
        or float(ordering_audit["fully_monotonic_rate"]) < 0.98
    ):
        connection.close()
        raise RuntimeError(f"MultiPV ordering audit failed: {ordering_audit}")
    exported_count = export_jsonl(connection, jsonl_path)
    connection.close()
    elapsed = time.perf_counter() - started
    manifest: dict[str, Any] = {
        "format": "deepblue-nnue-data-manifest-v0",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": DATASET_PAGE,
        "source_revision": DATASET_REVISION,
        "source_file": args.url,
        "license": LICENSE,
        "remote_file_bytes": reader.size,
        "source_etag": reader.etag,
        "source_last_modified": reader.last_modified,
        "downloaded_bytes": reader.downloaded_bytes,
        "http_range_requests": reader.http_requests,
        "download_cap_bytes": args.max_download_bytes,
        "raw_rows_scanned": raw_rows,
        "unique_fens": exported_count,
        "requested_unique_fens": args.unique_fens,
        "reappearing_fen_groups": reappearing_groups,
        "discarded_incomplete_boundary_groups": discarded_boundary_groups,
        "selection": "max(depth, then knodes), first row within that analysis/MultiPV group",
        "ordering_audit": ordering_audit,
        "elapsed_seconds": elapsed,
        "rows_per_second": raw_rows / max(elapsed, 1e-9),
        "selected_jsonl": str(jsonl_path),
        "selected_jsonl_sha256": sha256_file(jsonl_path),
        "database": str(database_path),
    }
    if exported_count > HARD_MAX_UNIQUE:
        raise RuntimeError("hard unique-FEN cap violated")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--url", default=DATASET_URL)
    parser.add_argument("--unique-fens", type=int, default=800_000)
    parser.add_argument("--max-download-bytes", type=int, default=1_500_000_000)
    parser.add_argument("--block-mib", type=int, default=8)
    parser.add_argument("--cached-blocks", type=int, default=16)
    parser.add_argument("--batch-rows", type=int, default=65_536)
    parser.add_argument("--raw-row-limit", type=int, default=0)
    parser.add_argument("--discard-first-fen", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.unique_fens <= HARD_MAX_UNIQUE:
        parser.error(f"--unique-fens must be in [1,{HARD_MAX_UNIQUE}]")
    if not 1 <= args.max_download_bytes <= HARD_MAX_DOWNLOAD_BYTES:
        parser.error(f"--max-download-bytes must be in [1,{HARD_MAX_DOWNLOAD_BYTES}]")
    repo_root = Path(__file__).resolve().parent.parent
    if args.output_dir.resolve().is_relative_to(repo_root):
        parser.error("large/derived data output must be outside the repository")
    if not args.manifest.resolve().is_relative_to(repo_root / "nnue_lab"):
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
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
