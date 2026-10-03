#!/usr/bin/env python
"""Normalize downloaded BTS monthly archives into partitioned Parquet."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from flight_rl.data import DEFAULT_CHUNK_ROWS, preprocess_archives
from flight_rl.models import DEFAULT_AIRPORTS

FILENAME_TEMPLATE = (
    "On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_{year}_{month}.zip"
)


def parse_year_month(value: str) -> date:
    try:
        parsed = date.fromisoformat(f"{value}-01")
    except ValueError as exc:
        raise ValueError(f"invalid year-month {value!r}; expected YYYY-MM") from exc
    if parsed.strftime("%Y-%m") != value:
        raise ValueError(f"invalid year-month {value!r}; expected YYYY-MM")
    return parsed


def month_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end month must not precede start month")
    months: list[date] = []
    current = start
    while current <= end:
        months.append(current)
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return months


def archive_paths(raw_dir: Path, months: list[date]) -> list[Path]:
    paths = [
        raw_dir / FILENAME_TEMPLATE.format(year=month.year, month=month.month) for month in months
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        preview = "\n".join(missing[:10])
        suffix = f"\n... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise ValueError(f"missing {len(missing)} required monthly archives:\n{preview}{suffix}")
    return paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="First source month, YYYY-MM.")
    parser.add_argument("--end", required=True, help="Last source month, YYYY-MM, inclusive.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/bts_v1"))
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--airports",
        nargs="+",
        metavar="IATA",
        help="Retain routes whose endpoints are both in this airport list.",
    )
    scope.add_argument(
        "--all-airports",
        action="store_true",
        help="Retain every route with a valid UTC schedule.",
    )
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        start = parse_year_month(args.start)
        end = parse_year_month(args.end)
        months = month_range(start, end)
        paths = archive_paths(args.raw_dir, months)
    except ValueError as exc:
        parser.error(str(exc))
    airports = None if args.all_airports else tuple(args.airports or DEFAULT_AIRPORTS)
    print(
        f"months={len(paths)} scope={'all-airports' if airports is None else ','.join(airports)} "
        f"chunk_rows={args.chunk_rows} out={args.out.resolve()}",
        flush=True,
    )
    records = preprocess_archives(
        paths,
        args.out,
        airports=airports,
        chunk_rows=args.chunk_rows,
        overwrite=args.overwrite,
        workers=args.workers,
    )
    raw_rows = sum(int(record["quality"]["raw_rows"]) for record in records)
    retained_rows = sum(int(record["quality"]["retained_rows"]) for record in records)
    excluded_rows = sum(int(record["quality"]["schedule_excluded"]) for record in records)
    print(
        f"complete months={len(records)} raw_rows={raw_rows} "
        f"retained_rows={retained_rows} schedule_excluded={excluded_rows}",
        flush=True,
    )


if __name__ == "__main__":
    main()
