"""Empirically verify whether dataset cp labels are White-relative."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import chess
import chess.engine
import numpy as np

from nnue_lab.features import normalize_fen, stable_hash64

OFFICIAL_SCHEMA = (
    "https://raw.githubusercontent.com/lichess-org/api/master/doc/specs/schemas/CloudEval.yaml"
)


def load_candidates(paths: list[Path]) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("mate") is not None or row.get("cp") is None:
                    continue
                fen = str(row["fen"])
                current = candidates.get(fen)
                quality = (int(row["depth"]), int(row["knodes"]))
                if current is None or quality > (int(current["depth"]), int(current["knodes"])):
                    candidates[fen] = row
    return candidates


def balanced_sample(candidates: dict[str, dict[str, Any]], count: int) -> list[dict[str, Any]]:
    per_side = count // 2
    selected: list[dict[str, Any]] = []
    for active_colour in ("w", "b"):
        pool = [
            row
            for row in candidates.values()
            if str(row["fen"]).split()[1] == active_colour
            and 300 <= abs(int(row["cp"])) <= 1800
        ]
        if len(pool) < per_side:
            pool = [
                row
                for row in candidates.values()
                if str(row["fen"]).split()[1] == active_colour
                and 100 <= abs(int(row["cp"])) <= 2000
            ]
        pool.sort(key=lambda row: stable_hash64(str(row["fen"]), 20260903))
        selected.extend(pool[:per_side])
    return selected


def sign(value: int) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def correlation(left: list[int], right: list[int]) -> float:
    if len(left) < 2:
        return math.nan
    return float(np.corrcoef(np.asarray(left, dtype=np.float64), np.asarray(right))[0, 1])


def verify(args: argparse.Namespace) -> dict[str, Any]:
    candidates = load_candidates(args.input)
    sample = balanced_sample(candidates, args.positions)
    if len(sample) < max(20, args.positions // 2):
        raise RuntimeError(f"only {len(sample)} balanced high-signal candidates were available")
    engine = chess.engine.SimpleEngine.popen_uci(str(args.stockfish))
    evidence: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        engine.configure({"Threads": 1, "Hash": args.hash_mib})
        engine_id = dict(engine.id)
        for row in sample:
            board = chess.Board(normalize_fen(str(row["fen"])))
            info = engine.analyse(board, chess.engine.Limit(depth=args.depth))
            score = info["score"]
            white_score = score.pov(chess.WHITE).score()
            stm_score = score.pov(board.turn).score()
            if white_score is None or stm_score is None:
                continue
            evidence.append(
                {
                    "fen": str(row["fen"]),
                    "dataset_cp": int(row["cp"]),
                    "dataset_depth": int(row["depth"]),
                    "dataset_knodes": int(row["knodes"]),
                    "stockfish_white_cp": int(white_score),
                    "stockfish_stm_cp": int(stm_score),
                    "stockfish_depth": int(info.get("depth", 0)),
                    "stockfish_nodes": int(info.get("nodes", 0)),
                }
            )
    finally:
        engine.quit()
    dataset = [int(item["dataset_cp"]) for item in evidence]
    sf_white = [int(item["stockfish_white_cp"]) for item in evidence]
    sf_stm = [int(item["stockfish_stm_cp"]) for item in evidence]
    decisive = [index for index, value in enumerate(sf_white) if abs(value) >= args.sign_floor_cp]
    white_matches = sum(
        sign(dataset[index]) == sign(sf_white[index]) for index in decisive
    )
    white_agreement = white_matches / max(1, len(decisive))
    stm_agreement = sum(sign(dataset[index]) == sign(sf_stm[index]) for index in decisive) / max(
        1, len(decisive)
    )
    white_turns = sum(str(item["fen"]).split()[1] == "w" for item in evidence)
    black_turns = len(evidence) - white_turns
    conclusion = "inconclusive"
    if (
        len(decisive) >= 20
        and min(white_turns, black_turns) >= 8
        and white_agreement >= 0.75
        and white_agreement >= stm_agreement + 0.20
    ):
        conclusion = "white-relative"
    return {
        "format": "deepblue-nnue-orientation-proof-v0",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "conclusion": conclusion,
        "method": (
            "balanced high-|cp| non-mate sample; current Stockfish re-analysis; "
            "compare raw dataset "
            "sign/correlation against White POV and side-to-move POV"
        ),
        "official_schema_support": OFFICIAL_SCHEMA,
        "stockfish_path_external_not_shipped": str(args.stockfish.resolve()),
        "stockfish_id": engine_id,
        "analysis_depth": args.depth,
        "positions_requested": args.positions,
        "positions_scored_nonmate": len(evidence),
        "white_to_move": white_turns,
        "black_to_move": black_turns,
        "decisive_for_sign_test": len(decisive),
        "sign_floor_cp": args.sign_floor_cp,
        "white_relative_sign_agreement": white_agreement,
        "stm_relative_sign_agreement": stm_agreement,
        "white_relative_pearson_correlation": correlation(dataset, sf_white),
        "stm_relative_pearson_correlation": correlation(dataset, sf_stm),
        "elapsed_seconds": time.perf_counter() - started,
        "evidence": evidence,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=40)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--hash-mib", type=int, default=64)
    parser.add_argument("--sign-floor-cp", type=int, default=80)
    args = parser.parse_args()
    repo_lab = Path(__file__).resolve().parent
    if not args.output.resolve().is_relative_to(repo_lab):
        parser.error("orientation proof must be written inside nnue_lab")
    return args


def main() -> None:
    args = parse_args()
    result = verify(args)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "evidence"}, indent=2))
    if result["conclusion"] != "white-relative":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
