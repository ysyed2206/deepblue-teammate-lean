"""Small synthetic checks for append-only materialisation, without model metrics."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np

from nnue_lab.production.prepare_data import (
    encode_fen,
    export_split,
    filter_invalid_positions,
    sha256_file,
    side_to_move_target,
    split_for_fen,
)
from nnue_lab.production.prepare_extension import (
    DATA_ROOT,
    connect_extension_database,
    exclude_ledger,
    merge_train_source,
)


class ExtensionTest(unittest.TestCase):
    def test_exclusion_quality_domain_and_train_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="extension-test-", dir=DATA_ROOT) as directory:
            root = Path(directory)
            fens = [
                f"4k3/8/8/8/8/8/8/4K3 {'w' if index % 2 else 'b'} - - {index} 1"
                for index in range(1, 80)
            ]
            train = [fen for fen in fens if split_for_fen(fen) == "train"]
            nontrain = next(fen for fen in fens if split_for_fen(fen) != "train")
            old = root / "old.sqlite3"
            with closing(sqlite3.connect(old)) as db, db:
                db.executescript("CREATE TABLE positions(fen TEXT PRIMARY KEY);"
                                 "CREATE TABLE old_fens(fen TEXT PRIMARY KEY);")
                db.execute("INSERT INTO positions VALUES (?)", (train[0],))
                db.execute("INSERT INTO old_fens VALUES (?)", (train[1],))
            old_hash = sha256_file(old)
            later = root / "prior.sqlite3"
            with closing(sqlite3.connect(later)) as db, db:
                db.execute("CREATE TABLE positions(fen TEXT PRIMARY KEY)")
                db.execute("INSERT INTO positions VALUES (?)", (train[2],))
            invalid = next(
                f"8/8/8/8/8/8/8/4K3 w - - {index} 1"
                for index in range(100)
                if split_for_fen(f"8/8/8/8/8/8/8/4K3 w - - {index} 1") == "train"
            )
            source1 = root / "first" / "selected.sqlite3"
            source2 = root / "second" / "selected.sqlite3"
            rows1: list[tuple[str, str, int, int, int | None, int | None, int]] = [
                (fen, "e1e2", 10, 100, 20, None, index)
                for index, fen in enumerate(train[:7])
            ]
            rows1 += [(nontrain, "e1e2", 10, 100, 20, None, 80),
                      (train[7], "e1e2", 10, 100, 20, 3, 81),
                      (train[8], "e1e2", 10, 100, None, None, 82),
                      (invalid, "e1e2", 10, 100, 20, None, 83)]
            rows2 = [(train[3], "e1e2", 11, 1, 3000, None, 0),
                     (train[4], "e1e2", 10, 101, -3000, None, 1),
                     (train[5], "e1e2", 10, 100, 999, None, 2),
                     (train[6], "e1e2", 9, 1000, 999, None, 3)]
            for path, rows in ((source1, rows1), (source2, rows2)):
                path.parent.mkdir()
                with closing(sqlite3.connect(path)) as db, db:
                    db.execute(
                        "CREATE TABLE positions(fen TEXT PRIMARY KEY,line TEXT,"
                        "depth INTEGER,knodes INTEGER,cp INTEGER,mate INTEGER,first_row INTEGER)"
                    )
                    db.executemany("INSERT INTO positions VALUES (?,?,?,?,?,?,?)", rows)
            hashes = [sha256_file(path) for path in (old, later, source1, source2)]
            out = root / "out"
            out.mkdir()
            connection = connect_extension_database(out / "extension_positions.sqlite3")
            try:
                exclude_ledger(connection, old)
                exclude_ledger(connection, later)
                merge_train_source(connection, source1)
                update = merge_train_source(connection, source2)
                self.assertEqual(update["higher_quality_updates"], 2)
                validation = filter_invalid_positions(connection)
                self.assertEqual(validation["invalid_positions_excluded"], 1)
                connection.execute("DELETE FROM positions WHERE valid=0")
                connection.commit()
                actual = dict(connection.execute("SELECT fen,cp FROM positions"))
                self.assertEqual(actual, {train[3]: 3000, train[4]: -3000,
                                          train[5]: 20, train[6]: 20})
                exported = export_split(connection, out, "train", 4)
                self.assertEqual(len(exported["npz_files"]), 1)
                with np.load(exported["npz_files"][0]["path"]) as data:
                    self.assertEqual(data["indices"].dtype, np.uint16)
                    self.assertEqual(data["sides"].dtype, np.uint8)
                    self.assertEqual(data["targets"].dtype, np.int16)
                    for index, fen in enumerate(sorted(actual)):
                        features, side = encode_fen(fen)
                        np.testing.assert_array_equal(data["indices"][index], features)
                        self.assertEqual(data["sides"][index], side)
                        self.assertEqual(
                            data["targets"][index], side_to_move_target(actual[fen], side)[0]
                        )
            finally:
                connection.close()
            self.assertEqual(old_hash, sha256_file(old))
            self.assertEqual(hashes, [sha256_file(path) for path in (old, later, source1, source2)])


if __name__ == "__main__":
    unittest.main()
