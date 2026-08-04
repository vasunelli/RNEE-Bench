from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def ordered_sha256(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def artifact_manifest(paths: Iterable[Path], **metadata: Any) -> dict[str, Any]:
    unique = sorted(set(paths), key=lambda value: str(value).lower())
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifacts": [
            {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in unique
        ],
    }
    result.update(metadata)
    return result


def verify_manifest(
    manifest: dict[str, Any], base_directory: Path | None = None
) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    for item in manifest.get("artifacts", []):
        path = Path(item["path"])
        if not path.is_absolute() and base_directory is not None:
            path = base_directory / path
        if (
            not path.exists()
            or int(path.stat().st_size) != int(item["bytes"])
            or sha256_file(path) != item["sha256"]
        ):
            mismatches.append(str(path))
    return not mismatches, mismatches
