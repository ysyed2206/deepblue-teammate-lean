"""Point agent.py at a candidate, validate it, and package it.

Everything that has to be true before a build is uploaded, in one place and in
the order that catches problems cheapest-first:

  1. the engine module imports and returns a legal move
  2. the regression corpus passes (known_fail entries are reported, not counted)
  3. package_lean builds the trimmed tree, which itself measures TRUE cold
     start against the 90s platform budget and refuses to package over 75s

Every one of these was learned the hard way this session. The cold-start gate
exists because init time went unmeasured for weeks and reached 62.6s against a
65s stop line. The regression run exists because a candidate that fixed one
position (the graded king shield) passed 33/33 and still cost 66 Elo.

    python tools/ship.py fastsearch135 --out submission_135.zip
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(label: str, args: list[str]) -> str:
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout + result.stderr)
        raise SystemExit(f"FAILED: {label}")
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("module")
    parser.add_argument("--out", default=None)
    parser.add_argument("--movetime-ms", type=int, default=2000)
    args = parser.parse_args()

    module = args.module
    suffix = module.replace("fastsearch", "")
    out = Path(args.out or f"submission_{suffix}.zip")

    agent = ROOT / "agent.py"
    backup = ROOT / f".agent_backup_{suffix}.py"
    shutil.copy2(agent, backup)
    try:
        text = agent.read_text(encoding="utf-8")
        current = None
        for line in text.splitlines():
            if line.startswith("from deepblue import fastsearch"):
                current = line.split()[-1]
                break
        if current is None:
            raise SystemExit("could not find the engine import in agent.py")
        if current != module:
            old_suffix = current.replace("fastsearch", "")
            text = text.replace(current, module)
            text = text.replace(f"FastEngine{old_suffix}", f"FastEngine{suffix}")
            agent.write_text(text, encoding="utf-8")
        print(f"agent.py -> {module}")

        print("\n[1/3] engine returns a legal move")
        print(run("smoke", [sys.executable, "-c",
                            "import agent, chess;"
                            "m = agent.get_move(chess.Board().fen(), 3000);"
                            "assert chess.Move.from_uci(m) in chess.Board().legal_moves;"
                            "print('   ok:', m)"]).strip())

        print("\n[2/3] regression corpus")
        corpus = run("regression", [sys.executable, "tools/regression.py",
                                    "--module", module,
                                    "--movetime-ms", str(args.movetime_ms)])
        for line in corpus.splitlines():
            if "passed" in line or "FAIL" in line or "STILL BROKEN" in line or "NOW FIXED" in line:
                print("   " + line.strip())

        print("\n[3/3] package (measures true cold start, gates on the 90s budget)")
        print(run("package", [sys.executable, "tools/package_lean.py",
                              "--out", str(out.resolve())]))
    finally:
        shutil.copy2(backup, agent)
        backup.unlink()
        print(f"agent.py restored; built {out.name}")


if __name__ == "__main__":
    main()
