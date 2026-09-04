"""Create a complete, deterministic inventory of the isolated lab directory."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "FILE_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = sorted(
        (path for path in ROOT.rglob("*") if path.is_file() and path != OUTPUT),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )
    entries = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    manifest = {
        "format": "deepblue-nnue-file-manifest-v0",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "root": "nnue_lab",
        "manifest_file": "FILE_MANIFEST.json",
        "note": "The manifest names itself here; all other files have size and SHA-256 entries.",
        "files_excluding_manifest": len(entries),
        "total_files_including_manifest": len(entries) + 1,
        "files": entries,
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "total_files": len(entries) + 1}, indent=2))


if __name__ == "__main__":
    main()
