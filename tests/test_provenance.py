from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from flight_rl.provenance import dataset_identity, file_sha256, source_identity


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _partition_entry(
    root: Path,
    month: str,
    *,
    content: bytes | None = None,
    normalization_version: int = 2,
    source_sha256: str | None = None,
) -> tuple[dict[str, object], Path]:
    year, month_number = month.split("-")
    partition = root / f"year={year}" / f"month={month_number}" / "flights.parquet"
    partition.parent.mkdir(parents=True, exist_ok=True)
    payload = content if content is not None else f"parquet-{month}".encode()
    partition.write_bytes(payload)
    entry: dict[str, object] = {
        "month": month,
        "normalization_version": normalization_version,
        "airports": ["AAA", "BBB"],
        "parquet_path": partition.relative_to(root).as_posix(),
        "parquet_bytes": len(payload),
        "parquet_sha256": _sha256(partition),
        "source_sha256": source_sha256 or hashlib.sha256(f"raw-{month}".encode()).hexdigest(),
        "quality": {"raw_rows": 10, "retained_rows": 8},
    }
    return entry, partition


def _write_manifest(root: Path, entries: list[dict[str, object]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "_preprocess_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "test normalized data",
                "months": entries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_dataset_identity_rejects_absent_requested_month(tmp_path: Path) -> None:
    january, _ = _partition_entry(tmp_path, "2020-01")
    march, _ = _partition_entry(tmp_path, "2020-03")
    _write_manifest(tmp_path, [january, march])

    with pytest.raises(ValueError, match="Requested fit months are missing: 2020-02"):
        dataset_identity(tmp_path, "2020-01-15", "2020-03-02")


def test_dataset_identity_rejects_same_size_tampered_parquet(tmp_path: Path) -> None:
    entry, partition = _partition_entry(tmp_path, "2020-01", content=b"alpha")
    _write_manifest(tmp_path, [entry])
    assert dataset_identity(tmp_path, "2020-01-01", "2020-01-31")[
        "complete_requested_month_coverage"
    ]

    partition.write_bytes(b"bravo")

    with pytest.raises(ValueError, match="Partition content does not match lineage"):
        dataset_identity(tmp_path, "2020-01-01", "2020-01-31")


def test_dataset_identity_rejects_unlisted_parquet(tmp_path: Path) -> None:
    entry, _ = _partition_entry(tmp_path, "2020-01")
    _write_manifest(tmp_path, [entry])
    unlisted = tmp_path / "scratch" / "unlisted.parquet"
    unlisted.parent.mkdir()
    unlisted.write_bytes(b"not in manifest")

    with pytest.raises(ValueError, match="Parquet inventory differs from the lineage manifest"):
        dataset_identity(tmp_path, "2020-01-01", "2020-01-31")


@pytest.mark.parametrize("version", [1, 999, None, "3", True])
def test_dataset_identity_rejects_unsupported_normalization_version(
    tmp_path: Path, version
) -> None:
    stale, _ = _partition_entry(tmp_path, "2020-01", normalization_version=version)
    _write_manifest(tmp_path, [stale])

    with pytest.raises(ValueError, match="Unsupported normalization version"):
        dataset_identity(tmp_path, "2020-01-01", "2020-01-31")


def test_source_hash_change_changes_declared_dataset_identity(tmp_path: Path) -> None:
    original_source_hash = "a" * 64
    changed_source_hash = "b" * 64
    entry, _ = _partition_entry(
        tmp_path,
        "2020-01",
        source_sha256=original_source_hash,
    )
    _write_manifest(tmp_path, [entry])
    original = dataset_identity(tmp_path, "2020-01-01", "2020-01-31")

    entry["source_sha256"] = changed_source_hash
    _write_manifest(tmp_path, [entry])
    changed = dataset_identity(tmp_path, "2020-01-01", "2020-01-31")

    assert original["fit_source_hashes"] == {"2020-01": original_source_hash}
    assert changed["fit_source_hashes"] == {"2020-01": changed_source_hash}
    assert changed["manifest_sha256"] != original["manifest_sha256"]
    assert changed["verified_partition_count"] == original["verified_partition_count"] == 1


def test_uncommitted_source_edit_changes_content_identity(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    source = source_dir / "example.py"
    source.write_text("value = 1\n")
    before = source_identity(tmp_path)
    source.write_text("value = 2\n")
    after = source_identity(tmp_path)
    assert before["source_content_sha256"] != after["source_content_sha256"]
    assert before["files_sha256"]["src/example.py"] != after["files_sha256"]["src/example.py"]


def test_source_identity_ignores_checkout_line_endings_but_not_other_bytes(tmp_path):
    source = tmp_path / "src" / "example.py"
    source.parent.mkdir()
    source.write_bytes(b"value = 1\nother = 2\n")
    lf = source_identity(tmp_path)
    exact_lf = file_sha256(source)
    source.write_bytes(b"value = 1\r\nother = 2\r\n")
    crlf = source_identity(tmp_path)
    assert crlf == lf
    assert file_sha256(source) != exact_lf
    source.write_bytes(b"value = 1 \r\nother = 2\r\n")
    assert source_identity(tmp_path)["source_content_sha256"] != lf["source_content_sha256"]


def test_source_identity_includes_lab_viewer_and_case_catalog(tmp_path):
    viewer = tmp_path / "src" / "flight_rl" / "lab.html"
    catalog = tmp_path / "examples" / "decision_lab_cases.json"
    viewer.parent.mkdir(parents=True)
    catalog.parent.mkdir()
    viewer.write_bytes(b"<title>Lab</title>\n")
    catalog.write_bytes(b'[{"scenario_seed": 42}]\n')
    before = source_identity(tmp_path)
    viewer.write_bytes(b"<title>Changed</title>\n")
    after_viewer = source_identity(tmp_path)
    catalog.write_bytes(b'[{"scenario_seed": 43}]\n')
    after_catalog = source_identity(tmp_path)
    assert (
        len(
            {
                before["source_content_sha256"],
                after_viewer["source_content_sha256"],
                after_catalog["source_content_sha256"],
            }
        )
        == 3
    )
