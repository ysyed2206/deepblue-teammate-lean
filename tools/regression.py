"""Run the S0 regression corpus in tests/regression_fens.txt.

Each line asserts what the engine must conclude about a position. These are
the bugs we have already been bitten by; nothing gets removed from the corpus,
only added to.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepblue.constants import MATE_THRESHOLD  # noqa: E402
from deepblue.reference import Engine  # noqa: E402

CORPUS = Path(__file__).resolve().parent.parent / "tests" / "regression_fens.txt"
# Wide enough to admit a deliberate contempt score on a terminal draw
# (fastsearch144 returns -20 rather than 0 so the search steers away from
# drawn endings), still tight enough to catch a draw scored as +/-900.
DRAW_TOLERANCE = 25


class _Reference:
    """The S0 engine behind a uniform interface."""

    name = "reference"

    def search(self, fen: str, movetime_ms: int):
        board = chess.Board(fen)
        engine = Engine()
        engine.record_game_position(board)
        result = engine.search(board, movetime_ms, movetime_ms * 1.5)
        return (result.move.uci() if result.move else None), result.score, result.depth

    def insufficient(self, fen: str) -> bool:
        board = chess.Board(fen)
        return board.is_insufficient_material()


class _Fast1:
    """S1-search1: Zobrist + transposition table + ordering + repetition."""

    name = "fast1"

    def __init__(self) -> None:
        from deepblue import fastsearch as fs
        from deepblue import fastsearch1 as fs1
        from deepblue.fastcore import from_fen

        fs1.warm_up()
        self._fs = fs
        self._from_fen = from_fen
        self._engine = fs1.FastEngine1()

    def search(self, fen: str, movetime_ms: int):
        move, score, depth, _, _ = self._engine.search(
            self._from_fen(fen), movetime_ms, movetime_ms * 1.5
        )
        return move, score, depth

    def insufficient(self, fen: str) -> bool:
        bb, occ, _, _ = self._from_fen(fen)
        return bool(self._fs.insufficient_material(bb, occ))


class _Fast:
    """The S1 compiled engine behind the same interface."""

    name = "fast"

    def __init__(self) -> None:
        from deepblue import fastsearch as fs
        from deepblue.fastcore import from_fen

        fs.warm_up()
        self._fs = fs
        self._from_fen = from_fen
        self._engine = fs.FastEngine()

    def search(self, fen: str, movetime_ms: int):
        move, score, depth, _, _ = self._engine.search(
            self._from_fen(fen), movetime_ms, movetime_ms * 1.5
        )
        return move, score, depth

    def insufficient(self, fen: str) -> bool:
        bb, occ, _, _ = self._from_fen(fen)
        return bool(self._fs.insufficient_material(bb, occ))


class _Variant:
    """Any deepblue.fastsearchNNN module, behind the same interface.

    _Fast is pinned to deepblue.fastsearch -- the ORIGINAL engine, not the one
    that actually plays. So every corpus run has been validating a module the
    submission does not use. This makes the corpus point at whichever variant
    is being considered, which is the only way an entry about the champion's
    behaviour can mean anything.
    """

    def __init__(self, module_name: str) -> None:
        import importlib

        from deepblue.fastcore import from_fen

        module = importlib.import_module(f"deepblue.{module_name}")
        suffix = module_name.replace("fastsearch", "")
        self.name = module_name
        self._fs = module
        self._from_fen = from_fen
        self._engine = getattr(module, f"FastEngine{suffix}")()
        # Pay the JIT cost once, before anything is timed.
        self._engine.search(from_fen(chess.STARTING_FEN), 200, 400)

    def search(self, fen: str, movetime_ms: int):
        move, score, depth, _, _ = self._engine.search(
            self._from_fen(fen), movetime_ms, movetime_ms * 1.5
        )
        return move, score, depth

    def insufficient(self, fen: str) -> bool:
        bb, occ, _, _ = self._from_fen(fen)
        return bool(self._fs.insufficient_material(bb, occ))


def check(engine, tag: str, fen: str, note: str, movetime_ms: int) -> tuple[bool, str]:
    board = chess.Board(fen)

    # Rule-predicate tags do not run a search. Their expected value is taken
    # from python-chess, never from the handwritten label.
    if tag in ("insufficient", "not_insufficient"):
        truth = board.is_insufficient_material()
        expected = tag == "insufficient"
        if truth != expected:
            return False, f"CORPUS LABEL WRONG: python-chess says insufficient={truth}"
        ours = engine.insufficient(fen)
        if ours != truth:
            return False, f"engine says {ours}, python-chess says {truth}"
        return True, f"insufficient={truth}, agrees with python-chess"

    move, score, depth = engine.search(fen, movetime_ms)
    result = type("R", (), {"move": move, "score": score, "depth": depth})()

    has_moves = any(board.legal_moves)
    if has_moves and (result.move is None or chess.Move.from_uci(result.move) not in board.legal_moves):
        return False, f"returned {result.move!r}, not a legal move"

    if tag == "draw":
        # Cross-check the label: the referee must actually agree this is drawn.
        outcome = board.outcome(claim_draw=True)
        if outcome is not None and outcome.winner is not None:
            return False, f"CORPUS LABEL WRONG: referee says {outcome.termination}"
        if not has_moves:
            # A terminal draw: the search cannot move, which is itself correct.
            # Verify the referee agrees it is drawn rather than lost.
            outcome = board.outcome(claim_draw=True)
            if outcome is None or outcome.winner is not None:
                return False, f"referee says {outcome}"
            return True, "terminal draw, referee agrees"
        if abs(result.score) > DRAW_TOLERANCE:
            return False, f"scored {result.score:+d}, expected a draw"
        return True, f"score {result.score:+d}"
    if tag == "mate_stm":
        if result.score <= MATE_THRESHOLD:
            return False, f"scored {result.score:+d}, expected a forced mate"
        return True, f"mate score {result.score:+d} via {result.move}"
    if tag == "mated_stm":
        if result.score >= -MATE_THRESHOLD:
            return False, f"scored {result.score:+d}, expected to be getting mated"
        return True, f"score {result.score:+d}"
    if tag == "no_draw":
        if result.score <= DRAW_TOLERANCE:
            return False, f"scored {result.score:+d}, settled for a draw"
        return True, f"score {result.score:+d}"
    if tag == "legal":
        return True, f"{result.move} score {result.score:+d} depth {result.depth}"
    if tag.startswith("known_fail:"):
        banned = tag.split(":", 1)[1]
        if result.move == banned:
            return True, f"STILL BROKEN (expected): plays {banned} ({result.score:+d})"
        return True, f"NOW FIXED: plays {result.move} instead of {banned}"
    if tag.startswith("avoid:"):
        banned = tag.split(":", 1)[1]
        if result.move == banned:
            return False, f"played the banned move {banned} (score {result.score:+d})"
        return True, f"{result.move} score {result.score:+d} depth {result.depth}"
    return False, f"unknown tag {tag!r}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--movetime-ms", type=int, default=400)
    parser.add_argument(
        "--engine", choices=("reference", "fast", "fast1"), default="reference"
    )
    parser.add_argument(
        "--module",
        help="test a specific variant, e.g. fastsearch126 (overrides --engine)",
    )
    arguments = parser.parse_args()

    if arguments.module:
        engine = _Variant(arguments.module)
    else:
        engine = {"reference": _Reference, "fast": _Fast, "fast1": _Fast1}[arguments.engine]()
    print(f"engine: {engine.name}\n")
    failures = 0
    total = 0
    for line in CORPUS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tag, fen, note = (part.strip() for part in line.split("|", 2))
        total += 1
        passed, detail = check(engine, tag, fen, note, arguments.movetime_ms)
        failures += 0 if passed else 1
        mark = "pass" if passed else "FAIL"
        print(f"  {mark}  {tag:9s} {detail}")
        if not passed:
            print(f"        fen  {fen}")
            print(f"        note {note}")
    print(f"\n{total - failures}/{total} regression positions passed")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
