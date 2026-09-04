"""Build the two submission artifacts, explicitly and unambiguously.

There is exactly one champion and it is generated only from the frozen
champion source. A candidate packaging successfully must never overwrite it.
Having a single `submission.zip` that sometimes held the champion and
sometimes the candidate is how the wrong build gets uploaded at 10:45 on
submission day.

    submission_champion.zip   built from champion/   - the proven fallback
    submission_candidate.zip  built from the repo    - the current candidate
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAMPION = ROOT / "champion"
CHAMPION_ZIP = ROOT / "submission_champion.zip"
CANDIDATE_ZIP = ROOT / "submission_candidate.zip"


def build(source: Path, destination: Path) -> list[str]:
    """Package `source` using the harness's own packager, bare - no --include.

    Running bare is deliberate: it proves `deepblue/` is a default include, so
    a regression in that default fails here instead of on the platform.
    """
    completed = subprocess.run(
        [sys.executable, "-m", "harness.package", "--out", str(destination)],
        cwd=source,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
    )
    if completed.returncode != 0:
        print(completed.stdout + completed.stderr)
        raise SystemExit(f"packaging {source} failed")
    with zipfile.ZipFile(destination) as archive:
        names = sorted(archive.namelist())
    if "agent.py" not in names:
        raise SystemExit(f"{destination.name} has no agent.py at its root")
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--what", choices=("champion", "candidate", "both"), default="candidate"
    )
    arguments = parser.parse_args()

    if arguments.what in ("candidate", "both"):
        names = build(ROOT, CANDIDATE_ZIP)
        size = CANDIDATE_ZIP.stat().st_size
        print(f"{CANDIDATE_ZIP.name}: {size:,} bytes, {len(names)} files")
        for name in names:
            print(f"    {name}")

    if arguments.what in ("champion", "both"):
        if not (CHAMPION / "agent.py").exists():
            raise SystemExit("champion/agent.py is missing; the champion is not frozen")
        names = build(CHAMPION, CHAMPION_ZIP)
        size = CHAMPION_ZIP.stat().st_size
        print(f"{CHAMPION_ZIP.name}: {size:,} bytes, {len(names)} files")
        for name in names:
            print(f"    {name}")

    print("\nThe champion artifact is only ever built from champion/.")
    print("Packaging a candidate successfully does not promote it.")


if __name__ == "__main__":
    main()
