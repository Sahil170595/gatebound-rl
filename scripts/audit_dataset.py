#!/usr/bin/env python
"""Recheck source/Parquet identities, calendar units and retained outcome counts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flight_rl.evaluation import write_report
from flight_rl.provenance import dataset_identity, file_sha256, source_identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/bts_v1_default_airports"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--out", type=Path, default=Path("results/dataset_audit.json"))
    args = parser.parse_args()
    identity = dataset_identity(args.data, args.start, args.end)
    normalized = json.loads((args.data / "_preprocess_manifest.json").read_text())
    downloaded = json.loads((args.raw / "download_manifest.json").read_text())
    raw_entries = {entry["month"]: entry for entry in downloaded["files"]}
    totals = Counter()
    by_year = {}
    for entry in normalized["months"]:
        month = entry["month"]
        if month not in identity["requested_fit_months"]:
            continue
        raw = (args.raw / entry["source_archive"]).resolve()
        if not raw.is_relative_to(args.raw.resolve()):
            raise ValueError("Raw archive escapes the requested source directory")
        actual_hash = file_sha256(raw)
        if actual_hash != entry["source_sha256"] or actual_hash != raw_entries[month]["sha256"]:
            raise ValueError(f"Source SHA mismatch for {month}")
        if raw.stat().st_size != entry["source_bytes"]:
            raise ValueError(f"Source byte count mismatch for {month}")
        frame = (
            pq.ParquetFile(args.data / entry["parquet_path"])
            .read(
                columns=[
                    "flight_date",
                    "scheduled_departure_utc",
                    "scheduled_arrival_utc",
                    "cancelled",
                    "diverted",
                    "div_reached_dest",
                ]
            )
            .to_pandas()
        )
        if len(frame) != entry["quality"]["retained_rows"]:
            raise ValueError(f"Retained row count mismatch for {month}")
        if not frame["flight_date"].str.startswith(month).all():
            raise ValueError(f"Wrong source-calendar month in {month}")
        dep = pd.to_datetime(frame["scheduled_departure_utc"], unit="m", utc=True)
        local_date = pd.to_datetime(frame["flight_date"], utc=True)
        offset_minutes = (dep - local_date).dt.total_seconds() / 60
        if not offset_minutes.between(-1440, 2880).all():
            raise ValueError(f"Impossible UTC/calendar offset in {month}")
        if not (frame["scheduled_arrival_utc"] > frame["scheduled_departure_utc"]).all():
            raise ValueError(f"Nonpositive scheduled duration in {month}")
        cancelled, diverted = frame["cancelled"], frame["diverted"]
        counts = {
            "raw_rows": entry["quality"]["raw_rows"],
            "retained_rows": len(frame),
            "retained_cancelled_rows": int(cancelled.sum()),
            "retained_diverted_rows": int(diverted.sum()),
            "retained_cancelled_and_diverted_rows": int((cancelled & diverted).sum()),
            "retained_ordinary_rows": int((~cancelled & ~diverted).sum()),
            "retained_unresolved_diversion_rows": int(
                (~cancelled & diverted & ~frame["div_reached_dest"]).sum()
            ),
            "source_bytes": raw.stat().st_size,
        }
        for name in ("retained_cancelled_rows", "retained_diverted_rows", "retained_ordinary_rows"):
            if counts[name] != entry["quality"][name]:
                raise ValueError(f"Outcome retention count mismatch: {month} {name}")
        totals.update(counts)
        year = month[:4]
        by_year.setdefault(year, Counter()).update(counts)
    report = {
        "status": "passed",
        "dataset_identity": identity,
        "totals": dict(totals),
        "by_year": {year: dict(counts) for year, counts in by_year.items()},
        "download_manifest_sha256": file_sha256(args.raw / "download_manifest.json"),
        "source_identity": source_identity(Path(__file__).resolve().parents[1]),
        "scope": "Actual source SHA and normalized content/count/calendar checks; "
        "retained airport subset, not full-network policy evaluation",
    }
    write_report(args.out, report)
    print(json.dumps(report["totals"], indent=2))
    print(f"Report: {args.out.resolve()}")


if __name__ == "__main__":
    main()
