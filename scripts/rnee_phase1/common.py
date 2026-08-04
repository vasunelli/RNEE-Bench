from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(value.rstrip() + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(base: int, *parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return (int(base) + int.from_bytes(digest[:4], "little")) % (2**31 - 1)


def ordered_sha256(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def windows_to_local_path(value: str | Path) -> Path:
    """Normalize Windows-style manifest paths under either Windows or WSL."""
    text = str(value)
    if os.name == "nt":
        return Path(text)
    if len(text) >= 3 and text[1:3] == ":\\":
        drive = text[0].lower()
        return Path(f"/mnt/{drive}/{text[3:].replace(chr(92), '/')}")
    return Path(text)


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    residual = y - prediction
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    medae = float(np.median(np.abs(residual)))
    denominator = float(np.sum(np.square(y - np.mean(y))))
    r2 = float(1.0 - np.sum(np.square(residual)) / denominator) if denominator > 0 else np.nan
    return {"mae": mae, "rmse": rmse, "medae": medae, "r2": r2}


def artifact_manifest(paths: Iterable[Path], **metadata: Any) -> dict[str, Any]:
    unique = sorted(set(paths), key=lambda path: str(path).lower())
    payload: dict[str, Any] = {
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
    payload.update(metadata)
    return payload


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "_No eligible rows._"
    rendered = frame.copy()
    for column in rendered.columns:
        if pd.api.types.is_float_dtype(rendered[column]):
            rendered[column] = rendered[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
            )
    columns = [str(column) for column in rendered.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rendered.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)
