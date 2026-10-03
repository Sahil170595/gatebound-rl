"""Content identities and coverage checks for reproducible experiments."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

# Both produce the same UTC/column contract; v3 adds retained-status counters.
SUPPORTED_NORMALIZATION_VERSIONS = frozenset({2, 3})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(root: Path) -> dict:
    """Hash working source with CRLF normalized to LF, including uncommitted edits.

    Dataset hashes remain byte-exact. Only source/configuration line endings are
    normalized so Windows Git checkout conversion does not change their identity.
    """
    paths = [
        *root.joinpath("src").rglob("*.py"),
        *root.joinpath("src").rglob("*.html"),
        *root.joinpath("scripts").rglob("*.py"),
    ]
    paths += [
        root / "pyproject.toml",
        root / "constraints-tested.txt",
        root / "examples" / "decision_lab_cases.json",
    ]
    hashes = {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for path in sorted(paths)
        if path.is_file()
    }
    encoded = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        head = None
    return {
        "base_head": head,
        "source_content_sha256": hashlib.sha256(encoded).hexdigest(),
        "files_sha256": hashes,
        "source_hash_format": "sha256-crlf-to-lf-v1",
        "identity_scope": "Actual Python/HTML source, scripts, decision-lab case catalog and "
        "direct dependency configuration; "
        "CRLF is normalized to LF; all other bytes, including working-tree changes beyond "
        "HEAD, are hashed. Dataset hashes remain byte-exact.",
    }


def _months(start: str, end: str) -> list[str]:
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if last < first:
        raise ValueError("fit_end must not precede fit_start")
    year, month = first.year, first.month
    result = []
    while (year, month) <= (last.year, last.month):
        result.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return result


def dataset_identity(path: Path, fit_start: str, fit_end: str) -> dict:
    """Check month coverage and every listed partition's actual content hash.

    This establishes local content identity, not authenticity of a third-party source.
    Requested date bounds can select partial boundary months even when archives cover full months.
    """
    root = path.resolve()
    manifest = root / "_preprocess_manifest.json"
    if not manifest.is_file():
        raise ValueError(f"Missing preprocessing lineage manifest: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    entries = payload.get("months", [])
    if payload.get("schema_version") != 1 or not isinstance(entries, list) or not entries:
        raise ValueError("Unsupported or empty preprocessing manifest")
    by_month = {entry["month"]: entry for entry in entries}
    if len(by_month) != len(entries):
        raise ValueError("Duplicate months in preprocessing manifest")
    expected = _months(fit_start, fit_end)
    missing = sorted(set(expected) - set(by_month))
    if missing:
        raise ValueError(f"Requested fit months are missing: {', '.join(missing)}")
    version_values = [entry.get("normalization_version") for entry in entries]
    if any(
        type(value) is not int or value not in SUPPORTED_NORMALIZATION_VERSIONS
        for value in version_values
    ):
        raise ValueError("Unsupported normalization version")
    versions = set(version_values)
    scopes = {json.dumps(entry.get("airports"), sort_keys=True) for entry in entries}
    if len(versions) != 1 or len(scopes) != 1:
        raise ValueError("Incompatible normalization versions or airport scopes")
    partitions = set()
    for entry in entries:
        partition = (root / entry["parquet_path"]).resolve()
        if not partition.is_relative_to(root):
            raise ValueError("Partition path escapes the dataset directory")
        if partition in partitions:
            raise ValueError("Duplicate partition paths in manifest")
        partitions.add(partition)
        if (
            not partition.is_file()
            or partition.stat().st_size != entry["parquet_bytes"]
            or file_sha256(partition) != entry["parquet_sha256"]
        ):
            raise ValueError(f"Partition content does not match lineage: {partition}")
    actual = {p.resolve() for p in root.rglob("*.parquet")}
    if actual != partitions:
        raise ValueError("Parquet inventory differs from the lineage manifest")
    selected = [by_month[month] for month in expected]
    five_year_months = _months("2020-01-01", "2024-12-31")
    return {
        "manifest_sha256": file_sha256(manifest),
        "verified_partition_count": len(partitions),
        "available_months": sorted(by_month),
        "requested_fit_months": expected,
        "complete_requested_month_coverage": True,
        "all_2020_2024_source_months_present": set(five_year_months).issubset(by_month),
        "normalization_version": next(iter(versions)),
        "airport_scope": entries[0].get("airports"),
        "fit_source_raw_rows": sum(x["quality"]["raw_rows"] for x in selected),
        "fit_source_retained_rows": sum(x["quality"]["retained_rows"] for x in selected),
        "fit_source_hashes": {x["month"]: x["source_sha256"] for x in selected},
        "scope": "Full source-month counts before the loader's exact date/airport filters; "
        "local lineage and content identity, not independent source authentication",
    }
