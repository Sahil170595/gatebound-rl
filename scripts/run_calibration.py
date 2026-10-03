#!/usr/bin/env python
"""Evaluate 2020-2024 hierarchical priors against untouched 2025 flight rows."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flight_rl.calibration import CALIBRATION_COLUMNS, build_calibration_report
from flight_rl.evaluation import write_report
from flight_rl.provenance import dataset_identity, source_identity


def _canonical_date(value: str, label: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a canonical YYYY-MM-DD date") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{label} must be a canonical YYYY-MM-DD date")
    return value


def _read_rows(path: Path, start: str, end: str) -> pd.DataFrame:
    try:
        rows = pd.read_parquet(
            path,
            columns=["flight_date", *CALIBRATION_COLUMNS],
            filters=[("flight_date", ">=", start), ("flight_date", "<=", end)],
        )
    except Exception as exc:
        raise ValueError(f"cannot read calibration rows from {path}: {exc}") from exc
    if rows.empty:
        raise ValueError(f"calibration slice is empty: {path} {start}..{end}")
    if not rows["flight_date"].between(start, end).all():
        raise ValueError(f"Parquet date filtering returned rows outside {start}..{end}")
    return rows


def main() -> None:
    source_root = Path(__file__).resolve().parents[1]
    initial_source = source_identity(source_root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-data", type=Path, default=Path("data/processed/bts_v1_default_airports")
    )
    parser.add_argument(
        "--observed-data",
        type=Path,
        default=Path("data/processed/bts2025_default_airports"),
    )
    parser.add_argument("--fit-start", default="2020-01-01")
    parser.add_argument("--fit-end", default="2024-12-31")
    parser.add_argument("--observed-start", default="2025-01-01")
    parser.add_argument("--observed-end", default="2025-12-31")
    parser.add_argument("--min-support", type=int, default=30)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("results/calibration_2025.json"))
    args = parser.parse_args()

    try:
        fit_start = _canonical_date(args.fit_start, "fit_start")
        fit_end = _canonical_date(args.fit_end, "fit_end")
        observed_start = _canonical_date(args.observed_start, "observed_start")
        observed_end = _canonical_date(args.observed_end, "observed_end")
        if fit_end < fit_start:
            raise ValueError("fit_end must not precede fit_start")
        if observed_end < observed_start:
            raise ValueError("observed_end must not precede observed_start")
        if fit_end >= observed_start:
            raise ValueError("fit_end must precede observed_start to prevent held-out leakage")
        fit_identity = dataset_identity(args.fit_data, fit_start, fit_end)
        observed_identity = dataset_identity(args.observed_data, observed_start, observed_end)
        if fit_identity["airport_scope"] != observed_identity["airport_scope"]:
            raise ValueError("fit and observed datasets have different airport scopes")
        if fit_identity["normalization_version"] != observed_identity["normalization_version"]:
            raise ValueError("fit and observed datasets have different normalization versions")
        fit_rows = _read_rows(args.fit_data, fit_start, fit_end)
        observed_rows = _read_rows(args.observed_data, observed_start, observed_end)
        report = build_calibration_report(
            fit_rows,
            observed_rows,
            min_support=args.min_support,
            bins=args.bins,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if source_identity(source_root) != initial_source:
        raise RuntimeError("Source changed during calibration; rerun from a stable local snapshot")
    payload = {
        "status": "completed",
        "source_identity": initial_source,
        "command_arguments": sys.argv[1:],
        "inputs": {
            "fit": {
                "path": str(args.fit_data.resolve()),
                "start": fit_start,
                "end": fit_end,
                "lineage": fit_identity,
            },
            "observed": {
                "path": str(args.observed_data.resolve()),
                "start": observed_start,
                "end": observed_end,
                "lineage": observed_identity,
            },
        },
        "versions": {
            name: importlib.metadata.version(name) for name in ["numpy", "pandas", "pyarrow"]
        },
        "calibration": report,
    }
    write_report(args.out, payload)
    for name, outcome in report["outcomes"].items():
        print(
            f"{name}: fit={outcome['fit_global_rate']} observed={outcome['observed_rate']} "
            f"brier={outcome['brier']['hierarchical_pool']} "
            f"n={outcome['denominators']['brier_rows']}",
            flush=True,
        )
    print(f"Report: {args.out.resolve()}")


if __name__ == "__main__":
    main()
