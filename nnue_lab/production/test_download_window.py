"""Small offline boundary tests for later Parquet acquisition windows."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nnue_lab.download_data import build_database  # noqa: E402
from download_window import (  # noqa: E402
    acquire,
    next_row_group,
    prior_end,
    row_group_layout,
    select_window,
)


def row(fen: str, depth: int = 10, cp: int = 20) -> dict[str, Any]:
    return {"fen": fen + " w", "line": "a2a3", "depth": depth, "knodes": 10,
            "cp": cp, "mate": None}


class FakeBatch:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.num_rows = len(rows)

    def column(self, name: str) -> Any:
        return SimpleNamespace(to_pylist=lambda: [item[name] for item in self.rows])


class FakeParquet:
    def __init__(self, groups: list[list[dict[str, Any]]]) -> None:
        self.groups = groups
        self.metadata = SimpleNamespace(
            num_row_groups=len(groups), num_rows=sum(map(len, groups)),
            row_group=lambda index: SimpleNamespace(num_rows=len(groups[index])),
        )

    def iter_batches(self, *, batch_size: int, columns: Any, row_groups: list[int]) -> Any:
        del columns
        for index in row_groups:
            rows = self.groups[index]
            for start in range(0, len(rows), batch_size):
                yield FakeBatch(rows[start:start + batch_size])


class WindowBoundaryTests(unittest.TestCase):
    def test_absolute_end_is_reused_for_chained_windows(self) -> None:
        self.assertEqual(prior_end({"raw_rows_scanned": 123}), 123)
        self.assertEqual(prior_end({
            "raw_rows_scanned": 123,
            "source_row_interval": {"start_inclusive": 1000, "end_exclusive": 1123},
        }), 1123)

    def test_existing_output_fails_before_any_network_request(self) -> None:
        args = SimpleNamespace(prior_manifest=Path("unused.json"),
                               output_dir=Path(__file__).parent,
                               manifest=Path(__file__).parent / "unused.json")
        with patch("download_window.read_prior_chain", return_value=[(Path("unused"), {})]):
            with patch("download_window.HttpRangeReader") as reader:
                with self.assertRaises(FileExistsError):
                    acquire(args)
                reader.assert_not_called()

    def test_next_group_never_reuses_partially_consumed_group(self) -> None:
        parquet = FakeParquet([[row("A")] * 4, [row("B")] * 6, [row("C")] * 3])
        groups = row_group_layout(parquet.metadata)
        self.assertEqual(next_row_group(groups, 3), 1)
        self.assertEqual(next_row_group(groups, 4), 1)
        self.assertEqual(next_row_group(groups, 5), 2)
        with self.assertRaises(ValueError):
            next_row_group(groups, 11)

    def test_first_run_discard_and_last_run_complete_across_groups(self) -> None:
        parquet = FakeParquet([
            [row("old")] * 4,
            [row("boundary"), row("boundary"), row("A", 10)],
            [row("A", 20), row("B", 10), row("B", 20)],
            [row("C"), row("D")],
        ])
        groups = row_group_layout(parquet.metadata)
        connection = build_database(Path(":memory:"))
        try:
            result = select_window(connection, parquet, groups, 1, 2, 1)
            selected = connection.execute(
                "SELECT fen,depth,first_row FROM positions ORDER BY first_row"
            ).fetchall()
            self.assertEqual(selected, [("A w", 20, 7), ("B w", 20, 9)])
            self.assertEqual(result["discarded_first_fen_rows"], 2)
            self.assertEqual(result["source_row_interval"], {"start_inclusive": 4, "end_exclusive": 10})
            self.assertEqual(result["boundary_lookahead_row"], 10)
            self.assertEqual(result["parquet_row_groups_requested"], [1, 2, 3])
            self.assertEqual(result["raw_rows_scanned"], 6)
        finally:
            connection.close()

    def test_first_discarded_fen_can_span_row_groups(self) -> None:
        parquet = FakeParquet([[row("boundary")], [row("boundary"), row("A"), row("B")]])
        connection = build_database(Path(":memory:"))
        try:
            result = select_window(connection, parquet, row_group_layout(parquet.metadata), 0, 1, 2)
            self.assertEqual(result["discarded_first_fen_rows"], 2)
            self.assertEqual(connection.execute("SELECT fen FROM positions").fetchall(), [("A w",)])
        finally:
            connection.close()

    def test_physical_eof_is_not_treated_as_complete_fen_boundary(self) -> None:
        parquet = FakeParquet([[row("boundary"), row("A"), row("A", 20)]])
        connection = build_database(Path(":memory:"))
        try:
            with self.assertRaisesRegex(RuntimeError, "shard ended"):
                select_window(connection, parquet, row_group_layout(parquet.metadata), 0, 1, 2)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM positions").fetchone()[0], 0)
        finally:
            connection.close()

    def test_noncontiguous_fen_run_rejected(self) -> None:
        parquet = FakeParquet([[row("boundary"), row("A"), row("B"), row("A"), row("C"), row("D")]])
        connection = build_database(Path(":memory:"))
        try:
            with self.assertRaisesRegex(RuntimeError, "non-contiguous"):
                select_window(connection, parquet, row_group_layout(parquet.metadata), 0, 3, 2)
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
