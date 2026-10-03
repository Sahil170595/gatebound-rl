"""BTS ingestion, normalization, and empirical historical-flight data.

The normalized dataset keeps schedules separate from donor outcomes. Monthly
archives are processed independently into partitioned Parquet files, while
HistoricalFlightData samples a complete historical outcome row from one
explicitly selected comparable-flight pool.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import zipfile
from bisect import bisect_left
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .models import FlightCandidate, OutcomeSummary, SampledOutcome

RAW_COLUMNS = (
    "FlightDate",
    "Marketing_Airline_Network",
    "IATA_Code_Marketing_Airline",
    "Flight_Number_Marketing_Airline",
    "Operating_Airline",
    "IATA_Code_Operating_Airline",
    "Flight_Number_Operating_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "DepDelay",
    "ArrDelay",
    "ActualElapsedTime",
    "Cancelled",
    "CancellationCode",
    "Diverted",
    "DivReachedDest",
    "DivArrDelay",
    "DivActualElapsedTime",
    "Div1Airport",
    "Div2Airport",
    "Div3Airport",
    "Div4Airport",
    "Div5Airport",
    "Duplicate",
)

NORMALIZED_COLUMNS = (
    "flight_id",
    "donor_id",
    "flight_date",
    "origin",
    "dest",
    "carrier",
    "operating_carrier",
    "operating_flight_number",
    "scheduled_departure_utc",
    "scheduled_arrival_utc",
    "dep_hour_bucket",
    "season",
    "dep_delay_min",
    "arr_delay_min",
    "actual_elapsed_min",
    "cancelled",
    "diverted",
    "div_reached_dest",
    "div_arr_delay_min",
    "div_actual_elapsed_min",
    "div_airport",
    "source_month",
    "source_file",
    "source_row",
    "marketing_flight_number",
)

CONTRACT_COLUMNS = NORMALIZED_COLUMNS[:21]
PREPROCESS_MANIFEST = "_preprocess_manifest.json"
LINEAGE_FILENAME = "_lineage.json"
DEFAULT_CHUNK_ROWS = 100_000
NORMALIZATION_VERSION = 3
_MONTH_FROM_ARCHIVE = __import__("re").compile(r"_(\d{4})_(\d{1,2})\.zip$", __import__("re").I)
_NANOSECONDS_PER_MINUTE = 60_000_000_000
_INT64_EXCLUSIVE_MAX = 1 << 63

_PARQUET_SCHEMA = pa.schema(
    [
        pa.field("flight_id", pa.string(), nullable=False),
        pa.field("donor_id", pa.string(), nullable=False),
        pa.field("flight_date", pa.string(), nullable=False),
        pa.field("origin", pa.string(), nullable=False),
        pa.field("dest", pa.string(), nullable=False),
        pa.field("carrier", pa.string(), nullable=False),
        pa.field("operating_carrier", pa.string(), nullable=False),
        pa.field("operating_flight_number", pa.string(), nullable=False),
        pa.field("scheduled_departure_utc", pa.int64(), nullable=False),
        pa.field("scheduled_arrival_utc", pa.int64(), nullable=False),
        pa.field("dep_hour_bucket", pa.int8(), nullable=False),
        pa.field("season", pa.string(), nullable=False),
        pa.field("dep_delay_min", pa.float64()),
        pa.field("arr_delay_min", pa.float64()),
        pa.field("actual_elapsed_min", pa.float64()),
        pa.field("cancelled", pa.bool_(), nullable=False),
        pa.field("diverted", pa.bool_(), nullable=False),
        pa.field("div_reached_dest", pa.bool_(), nullable=False),
        pa.field("div_arr_delay_min", pa.float64()),
        pa.field("div_actual_elapsed_min", pa.float64()),
        pa.field("div_airport", pa.string()),
        pa.field("source_month", pa.string(), nullable=False),
        pa.field("source_file", pa.string(), nullable=False),
        pa.field("source_row", pa.int64(), nullable=False),
        pa.field("marketing_flight_number", pa.string(), nullable=False),
    ]
)


def _sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_month(path: Path) -> str:
    match = _MONTH_FROM_ARCHIVE.search(path.name)
    if match is None:
        raise ValueError(f"cannot determine source month from archive name: {path.name}")
    year, month = (int(part) for part in match.groups())
    if year < 2018 or not 1 <= month <= 12:
        raise ValueError(f"invalid source month in archive name: {path.name}")
    return f"{year:04d}-{month:02d}"


@contextmanager
def _open_csv(path: Path) -> Iterator[tuple[object, str]]:
    if not path.is_file():
        raise ValueError(f"raw BTS file does not exist: {path}")
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            csv_members = [
                info.filename
                for info in archive.infolist()
                if not info.is_dir() and info.filename.lower().endswith(".csv")
            ]
            if len(csv_members) != 1:
                raise ValueError(f"expected exactly one CSV member in {path}, found {csv_members}")
            with archive.open(csv_members[0]) as stream:
                yield stream, csv_members[0]
    elif path.suffix.lower() == ".csv":
        with path.open("rb") as stream:
            yield stream, path.name
    else:
        raise ValueError(f"expected a .zip or .csv BTS source, found {path}")


def _trim_and_validate_raw(frame: pd.DataFrame) -> pd.DataFrame:
    stripped = [str(column).strip() for column in frame.columns]
    if len(stripped) != len(set(stripped)):
        raise ValueError("raw BTS headers collide after whitespace trimming")
    frame = frame.copy()
    frame.columns = stripped
    missing = sorted(set(RAW_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"raw BTS schema is missing required columns: {missing}")
    return frame.loc[:, list(RAW_COLUMNS)]


def _read_csv_args() -> dict[str, object]:
    required = set(RAW_COLUMNS)
    return {
        "usecols": lambda name: str(name).strip() in required,
        "low_memory": False,
    }


def _iter_monthly_raw(path: Path, chunk_rows: int) -> Iterator[tuple[pd.DataFrame, str]]:
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    offset = 0
    with _open_csv(path) as (stream, member):
        reader = pd.read_csv(stream, chunksize=chunk_rows, **_read_csv_args())
        for chunk in reader:
            chunk = _trim_and_validate_raw(chunk)
            chunk.insert(
                0,
                "_source_row",
                np.arange(offset + 1, offset + len(chunk) + 1, dtype=np.int64),
            )
            offset += len(chunk)
            yield chunk, member


def _clean_string(series: pd.Series, *, uppercase: bool = False) -> pd.Series:
    result = series.astype("string").str.strip()
    result = result.mask(result.eq(""))
    return result.str.upper() if uppercase else result


def _flight_number(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    integer = numeric.where(numeric.notna() & numeric.eq(numeric.round()))
    result = integer.astype("Int64").astype("string")
    fallback = _clean_string(series)
    return result.fillna(fallback).fillna("")


def _flag(series: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = ~numeric.isin([0, 1])
    if invalid.any():
        examples = series.loc[invalid].head(5).tolist()
        raise ValueError(f"{name} contains values other than 0/1: {examples}")
    return numeric.eq(1)


def _hhmm(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce")
    whole = numeric.notna() & np.isfinite(numeric) & numeric.eq(numeric.round())
    representable = whole & numeric.between(0, 2400)
    integers = numeric.where(representable, -1).round().astype(np.int64)
    is_2400 = representable & integers.eq(2400)
    ordinary = representable & integers.between(0, 2359) & integers.mod(100).lt(60)
    valid = ordinary | is_2400
    minute_of_day = pd.Series(np.nan, index=series.index, dtype=np.float64)
    minute_of_day.loc[ordinary] = integers.loc[ordinary].floordiv(100).mul(60) + integers.loc[
        ordinary
    ].mod(100)
    minute_of_day.loc[is_2400] = 24 * 60
    return minute_of_day, valid, is_2400


@lru_cache(maxsize=1)
def _installed_airport_timezones() -> Mapping[str, str]:
    try:
        import airportsdata
    except ImportError as exc:
        raise ValueError("airportsdata is required for exact UTC normalization") from exc
    result = {
        str(code).upper(): str(record["tz"])
        for code, record in airportsdata.load("IATA").items()
        if record.get("tz")
    }
    return MappingProxyType(result)


@lru_cache(maxsize=1)
def _timezone_provenance() -> Mapping[str, str]:
    import airportsdata
    import tzdata

    mapping = _installed_airport_timezones()
    serialized = "\n".join(f"{code}\t{mapping[code]}" for code in sorted(mapping))
    return MappingProxyType(
        {
            "airportsdata_version": str(getattr(airportsdata, "__version__", "unknown")),
            "tzdata_version": str(getattr(tzdata, "__version__", "unknown")),
            "airport_timezone_map_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        }
    )


def _localize_departures(
    local_times: pd.Series,
    origins: pd.Series,
    eligible: pd.Series,
    airport_timezones: Mapping[str, str],
) -> tuple[pd.Series, dict[str, int]]:
    utc_minutes = pd.Series(np.nan, index=local_times.index, dtype=np.float64)
    zones = origins.map(airport_timezones)
    missing_zone = eligible & zones.isna()
    ambiguous_count = 0
    nonexistent_count = 0
    invalid_zone_count = 0

    for zone in sorted(str(value) for value in zones.loc[eligible & zones.notna()].unique()):
        mask = eligible & zones.eq(zone)
        values = local_times.loc[mask]
        try:
            ambiguous_probe = values.dt.tz_localize(
                zone, ambiguous="NaT", nonexistent="shift_forward"
            )
            nonexistent_probe = values.dt.tz_localize(zone, ambiguous=True, nonexistent="NaT")
            localized = values.dt.tz_localize(zone, ambiguous="NaT", nonexistent="NaT")
        except (TypeError, ValueError, KeyError):
            invalid_zone_count += int(mask.sum())
            continue
        ambiguous_count += int((values.notna() & ambiguous_probe.isna()).sum())
        nonexistent_count += int((values.notna() & nonexistent_probe.isna()).sum())
        valid = localized.notna()
        if valid.any():
            utc_values = (
                localized.loc[valid]
                .dt.as_unit("ns")
                .astype("int64")
                .floordiv(_NANOSECONDS_PER_MINUTE)
            )
            utc_minutes.loc[utc_values.index] = utc_values.astype(np.float64)

    return utc_minutes, {
        "missing_origin_timezone": int(missing_zone.sum()),
        "invalid_origin_timezone": invalid_zone_count,
        "dst_ambiguous_departure": ambiguous_count,
        "dst_nonexistent_departure": nonexistent_count,
    }


def _empty_normalized() -> pd.DataFrame:
    frame = pd.DataFrame(
        {field.name: pd.Series(dtype=field.type.to_pandas_dtype()) for field in _PARQUET_SCHEMA}
    )
    return frame.loc[:, list(NORMALIZED_COLUMNS)]


def normalize(
    raw: pd.DataFrame,
    *,
    source_month: str | None = None,
    source_file: str | None = None,
    source_row_offset: int = 0,
    airports: Iterable[str] | None = None,
    airport_timezones: Mapping[str, str] | None = None,
    return_quality: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, int]]:
    """Normalize one raw frame into stable UTC schedule and joint-outcome rows."""

    if source_row_offset < 0:
        raise ValueError("source_row_offset must be nonnegative")
    frame = raw.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = sorted(set(RAW_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"raw BTS schema is missing required columns: {missing}")
    frame = frame.reset_index(drop=True)
    if "_source_row" in raw.columns:
        source_rows = pd.to_numeric(raw["_source_row"], errors="raise").reset_index(drop=True)
    else:
        source_rows = pd.Series(
            np.arange(
                source_row_offset + 1,
                source_row_offset + len(frame) + 1,
                dtype=np.int64,
            )
        )
    if source_rows.duplicated().any() or source_rows.lt(1).any():
        raise ValueError("source rows must be unique positive integers within a source month")

    flight_dates = pd.to_datetime(frame["FlightDate"], format="%Y-%m-%d", errors="coerce")
    derived_months = flight_dates.dt.strftime("%Y-%m")
    if source_month is not None:
        try:
            parsed_source_month = date.fromisoformat(f"{source_month}-01")
        except ValueError as exc:
            raise ValueError("source_month must be YYYY-MM") from exc
        if parsed_source_month.strftime("%Y-%m") != source_month:
            raise ValueError("source_month must be YYYY-MM")
        wrong_month = flight_dates.notna() & derived_months.ne(source_month)
        if wrong_month.any():
            raise ValueError(
                f"{int(wrong_month.sum())} rows do not belong to source month {source_month}"
            )
        row_months = pd.Series(source_month, index=frame.index, dtype="string")
    else:
        row_months = derived_months.astype("string")

    origin = _clean_string(frame["Origin"], uppercase=True)
    dest = _clean_string(frame["Dest"], uppercase=True)
    carrier = _clean_string(frame["Marketing_Airline_Network"], uppercase=True)
    marketing_iata = _clean_string(frame["IATA_Code_Marketing_Airline"], uppercase=True)
    operating = _clean_string(frame["IATA_Code_Operating_Airline"], uppercase=True)
    operating = operating.fillna(_clean_string(frame["Operating_Airline"], uppercase=True))
    marketing_number = _flight_number(frame["Flight_Number_Marketing_Airline"])
    operating_number = _flight_number(frame["Flight_Number_Operating_Airline"])
    cancelled = _flag(frame["Cancelled"], "Cancelled")
    diverted = _flag(frame["Diverted"], "Diverted")

    dep_minute, valid_dep_hhmm, dep_2400 = _hhmm(frame["CRSDepTime"])
    _, valid_arr_hhmm, arr_2400 = _hhmm(frame["CRSArrTime"])
    elapsed = pd.to_numeric(frame["CRSElapsedTime"], errors="coerce")
    valid_elapsed = (
        elapsed.notna()
        & np.isfinite(elapsed)
        & elapsed.gt(0)
        & elapsed.lt(_INT64_EXCLUSIVE_MAX)
        & elapsed.eq(elapsed.round())
    )
    local_departure = flight_dates + pd.to_timedelta(dep_minute, unit="m")

    configured_airports = None
    if airports is not None:
        configured_airports = frozenset(
            str(airport).strip().upper() for airport in airports if str(airport).strip()
        )
        if not configured_airports:
            raise ValueError("airports must contain at least one nonempty code")
        airport_eligible = origin.isin(configured_airports) & dest.isin(configured_airports)
    else:
        airport_eligible = pd.Series(True, index=frame.index)

    same_airport = origin.notna() & origin.eq(dest)
    base_schedule_valid = (
        airport_eligible
        & flight_dates.notna()
        & row_months.notna()
        & origin.notna()
        & dest.notna()
        & carrier.notna()
        & valid_dep_hhmm
        & valid_arr_hhmm
        & valid_elapsed
        & ~same_airport
    )
    timezone_map = (
        {str(key).upper(): value for key, value in airport_timezones.items()}
        if airport_timezones is not None
        else _installed_airport_timezones()
    )
    departure_utc, timezone_quality = _localize_departures(
        local_departure, origin, base_schedule_valid, timezone_map
    )
    arrival_utc_candidate = departure_utc + elapsed.where(valid_elapsed)
    arrival_utc_representable = (
        arrival_utc_candidate.notna()
        & np.isfinite(arrival_utc_candidate)
        & arrival_utc_candidate.ge(np.iinfo(np.int64).min)
        & arrival_utc_candidate.lt(_INT64_EXCLUSIVE_MAX)
    )
    elapsed_overflows_arrival = (
        base_schedule_valid & departure_utc.notna() & ~arrival_utc_representable
    )
    valid_elapsed = valid_elapsed & ~elapsed_overflows_arrival
    schedule_valid = base_schedule_valid & departure_utc.notna() & arrival_utc_representable

    ordinary = ~cancelled & ~diverted
    dep_delay = pd.to_numeric(frame["DepDelay"], errors="coerce")
    arr_delay = pd.to_numeric(frame["ArrDelay"], errors="coerce")
    actual_elapsed = pd.to_numeric(frame["ActualElapsedTime"], errors="coerce")
    div_arr_delay = pd.to_numeric(frame["DivArrDelay"], errors="coerce")
    div_actual_elapsed = pd.to_numeric(frame["DivActualElapsedTime"], errors="coerce")
    div_reached_numeric = pd.to_numeric(frame["DivReachedDest"], errors="coerce")
    div_reached_present = (
        frame["DivReachedDest"].notna() & _clean_string(frame["DivReachedDest"]).notna()
    )
    invalid_div_reached = div_reached_present & ~div_reached_numeric.isin([0, 1])
    if invalid_div_reached.any():
        examples = frame.loc[invalid_div_reached, "DivReachedDest"].head(5).tolist()
        raise ValueError(
            "DivReachedDest contains values other than 0/1/null: "
            f"count={int(invalid_div_reached.sum())}, examples={examples}"
        )
    ordinary_elapsed_residual = (
        (elapsed + arr_delay - dep_delay - actual_elapsed).loc[ordinary].dropna()
    )
    reached_diversion = diverted & div_reached_numeric.eq(1)
    diversion_elapsed_residual = (
        (elapsed + div_arr_delay - dep_delay - div_actual_elapsed).loc[reached_diversion].dropna()
    )

    quality = {
        "raw_rows": len(frame),
        "airport_filter_excluded": int((~airport_eligible).sum()),
        "invalid_flight_date": int(flight_dates.isna().sum()),
        "invalid_departure_hhmm": int((~valid_dep_hhmm).sum()),
        "invalid_arrival_hhmm": int((~valid_arr_hhmm).sum()),
        "departure_2400": int(dep_2400.sum()),
        "arrival_2400": int(arr_2400.sum()),
        "invalid_crs_elapsed": int((~valid_elapsed).sum()),
        "same_airport_route": int(same_airport.sum()),
        "cancelled_rows": int(cancelled.sum()),
        "diverted_rows": int(diverted.sum()),
        "cancelled_and_diverted_rows": int((cancelled & diverted).sum()),
        "source_duplicate_flag_rows": int(
            _clean_string(frame["Duplicate"], uppercase=True).eq("Y").fillna(False).sum()
        ),
        "marketing_operating_different": int(carrier.ne(operating).fillna(True).sum()),
        "marketing_iata_mismatch": int(carrier.ne(marketing_iata).fillna(True).sum()),
        "ordinary_missing_dep_delay": int((ordinary & dep_delay.isna()).sum()),
        "ordinary_missing_arr_delay": int((ordinary & arr_delay.isna()).sum()),
        "ordinary_missing_actual_elapsed": int((ordinary & actual_elapsed.isna()).sum()),
        "ordinary_elapsed_inconsistent": int(ordinary_elapsed_residual.abs().gt(0.5).sum()),
        "diverted_missing_div_reached_dest": int((diverted & div_reached_numeric.isna()).sum()),
        "diverted_missing_div_arr_delay": int((diverted & div_arr_delay.isna()).sum()),
        "diverted_missing_div_actual_elapsed": int((diverted & div_actual_elapsed.isna()).sum()),
        "diverted_elapsed_inconsistent": int(diversion_elapsed_residual.abs().gt(0.5).sum()),
    }
    quality.update(timezone_quality)
    quality["schedule_excluded"] = int((airport_eligible & ~schedule_valid).sum())
    quality["schedule_excluded_cancelled"] = int(
        (airport_eligible & ~schedule_valid & cancelled).sum()
    )
    quality["schedule_excluded_diverted"] = int(
        (airport_eligible & ~schedule_valid & diverted).sum()
    )
    quality["retained_cancelled_rows"] = int((schedule_valid & cancelled).sum())
    quality["retained_diverted_rows"] = int((schedule_valid & diverted).sum())
    quality["retained_cancelled_and_diverted_rows"] = int(
        (schedule_valid & cancelled & diverted).sum()
    )
    quality["retained_ordinary_rows"] = int((schedule_valid & ~cancelled & ~diverted).sum())

    if not schedule_valid.any():
        normalized = _empty_normalized()
        quality["retained_rows"] = 0
        normalized.attrs["quality"] = quality
        return (normalized, quality) if return_quality else normalized

    selected = schedule_valid[schedule_valid].index
    departure_integer = departure_utc.loc[selected].round().astype(np.int64)
    elapsed_integer = elapsed.loc[selected].round().astype(np.int64)
    arrival_integer = departure_integer + elapsed_integer
    dep_bucket = dep_minute.mod(24 * 60).floordiv(180)
    season = flight_dates.dt.month.map(
        {
            1: "DJF",
            2: "DJF",
            3: "MAM",
            4: "MAM",
            5: "MAM",
            6: "JJA",
            7: "JJA",
            8: "JJA",
            9: "SON",
            10: "SON",
            11: "SON",
            12: "DJF",
        }
    )
    div_airport = pd.Series(pd.NA, index=selected, dtype="string")
    for number in range(1, 6):
        current_airport = _clean_string(frame.loc[selected, f"Div{number}Airport"], uppercase=True)
        div_airport = current_airport.fillna(div_airport)

    ids = pd.Series(
        [f"bts:{month}:{int(row):09d}" for month, row in zip(row_months, source_rows, strict=True)],
        index=frame.index,
        dtype="string",
    )
    normalized = pd.DataFrame(
        {
            "flight_id": ids.loc[selected],
            "donor_id": ids.loc[selected],
            "flight_date": flight_dates.loc[selected].dt.strftime("%Y-%m-%d"),
            "origin": origin.loc[selected],
            "dest": dest.loc[selected],
            "carrier": carrier.loc[selected],
            "operating_carrier": operating.loc[selected].fillna(""),
            "operating_flight_number": operating_number.loc[selected],
            "scheduled_departure_utc": departure_integer,
            "scheduled_arrival_utc": arrival_integer,
            "dep_hour_bucket": dep_bucket.loc[selected].astype(np.int8),
            "season": season.loc[selected],
            "dep_delay_min": dep_delay.loc[selected].astype(np.float64),
            "arr_delay_min": arr_delay.loc[selected].astype(np.float64),
            "actual_elapsed_min": actual_elapsed.loc[selected].astype(np.float64),
            "cancelled": cancelled.loc[selected].astype(bool),
            "diverted": diverted.loc[selected].astype(bool),
            "div_reached_dest": div_reached_numeric.loc[selected].eq(1).astype(bool),
            "div_arr_delay_min": div_arr_delay.loc[selected].astype(np.float64),
            "div_actual_elapsed_min": div_actual_elapsed.loc[selected].astype(np.float64),
            "div_airport": div_airport,
            "source_month": row_months.loc[selected],
            "source_file": source_file or str(raw.attrs.get("source_file", "")),
            "source_row": source_rows.loc[selected].astype(np.int64),
            "marketing_flight_number": marketing_number.loc[selected],
        }
    ).reset_index(drop=True)
    quality["retained_rows"] = len(normalized)
    normalized.attrs["quality"] = quality
    return (normalized, quality) if return_quality else normalized


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _lineage_matches(
    lineage: Mapping[str, object],
    *,
    source_sha256: str,
    airports: tuple[str, ...] | None,
    chunk_rows: int,
    timezone_provenance: Mapping[str, str],
) -> bool:
    recorded_airports = lineage.get("airports")
    expected_airports = list(airports) if airports is not None else None
    return (
        lineage.get("schema_version") == 1
        and lineage.get("normalization_version") == NORMALIZATION_VERSION
        and lineage.get("source_sha256") == source_sha256
        and recorded_airports == expected_airports
        and lineage.get("chunk_rows") == chunk_rows
        and lineage.get("timezone_provenance") == dict(timezone_provenance)
    )


def preprocess_archive(
    archive_path: Path,
    out_dir: Path,
    *,
    airports: Iterable[str] | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    overwrite: bool = False,
) -> dict[str, object]:
    """Normalize one monthly archive into one atomic partitioned Parquet file."""

    archive_path = Path(archive_path)
    out_dir = Path(out_dir)
    source_month = _archive_month(archive_path)
    year, month = source_month.split("-")
    selected_airports = (
        tuple(sorted({str(code).strip().upper() for code in airports if str(code).strip()}))
        if airports is not None
        else None
    )
    if selected_airports == ():
        raise ValueError("airports must contain at least one nonempty code")
    source_sha256 = _sha256_file(archive_path)
    source_bytes = archive_path.stat().st_size
    timezone_provenance = _timezone_provenance()
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"CRC failure in {archive_path}: {bad_member}")
        csv_members = [
            info.filename
            for info in archive.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".csv")
        ]
        if len(csv_members) != 1:
            raise ValueError(f"expected one CSV member in {archive_path}, found {csv_members}")
        csv_member = csv_members[0]

    partition_dir = out_dir / f"year={year}" / f"month={month}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = partition_dir / "flights.parquet"
    lineage_path = partition_dir / LINEAGE_FILENAME
    temporary_path = partition_dir / "_flights.parquet.part"
    legacy_temporary_path = parquet_path.with_suffix(parquet_path.suffix + ".part")

    if parquet_path.exists() or lineage_path.exists():
        if not parquet_path.exists() or not lineage_path.exists():
            if not overwrite:
                raise ValueError(f"incomplete existing output pair in {partition_dir}")
        else:
            lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
            if not overwrite and _lineage_matches(
                lineage,
                source_sha256=source_sha256,
                airports=selected_airports,
                chunk_rows=chunk_rows,
                timezone_provenance=timezone_provenance,
            ):
                expected_bytes = lineage.get("parquet_bytes")
                expected_sha256 = lineage.get("parquet_sha256")
                actual_bytes = parquet_path.stat().st_size
                actual_sha256 = _sha256_file(parquet_path)
                if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
                    raise ValueError(
                        f"existing Parquet integrity does not match lineage: {parquet_path}"
                    )
                reused = dict(lineage)
                reused["status"] = "reused"
                return reused
            if not overwrite:
                raise ValueError(
                    f"existing partition lineage does not match requested input: {partition_dir}"
                )

    for incomplete_path in (temporary_path, legacy_temporary_path):
        if incomplete_path.exists():
            incomplete_path.unlink()
    aggregate: Counter[str] = Counter()
    seen_identity_hashes: set[int] = set()
    identity_duplicates = 0
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(
            temporary_path,
            _PARQUET_SCHEMA,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        for raw_chunk, member in _iter_monthly_raw(archive_path, chunk_rows):
            normalized, quality = normalize(
                raw_chunk,
                source_month=source_month,
                source_file=member,
                airports=selected_airports,
                return_quality=True,
            )
            aggregate.update(quality)
            if not normalized.empty:
                identity_columns = [
                    "flight_date",
                    "carrier",
                    "marketing_flight_number",
                    "origin",
                    "dest",
                    "scheduled_departure_utc",
                    "scheduled_arrival_utc",
                ]
                identity_hashes = pd.util.hash_pandas_object(
                    normalized.loc[:, identity_columns], index=False
                ).to_numpy(dtype=np.uint64)
                for value in identity_hashes:
                    integer = int(value)
                    if integer in seen_identity_hashes:
                        identity_duplicates += 1
                    else:
                        seen_identity_hashes.add(integer)
                table = pa.Table.from_pandas(
                    normalized.loc[:, list(NORMALIZED_COLUMNS)],
                    schema=_PARQUET_SCHEMA,
                    preserve_index=False,
                    safe=True,
                )
                writer.write_table(table)
        writer.close()
        writer = None
        os.replace(temporary_path, parquet_path)
    except Exception:
        if writer is not None:
            writer.close()
        raise

    aggregate["identity_duplicate_rows_beyond_first"] = identity_duplicates
    record: dict[str, object] = {
        "schema_version": 1,
        "normalization_version": NORMALIZATION_VERSION,
        "status": "processed",
        "month": source_month,
        "source_archive": archive_path.name,
        "source_path": str(archive_path.resolve()),
        "source_bytes": source_bytes,
        "source_sha256": source_sha256,
        "csv_member": csv_member,
        "airports": list(selected_airports) if selected_airports is not None else None,
        "chunk_rows": chunk_rows,
        "timezone_provenance": dict(timezone_provenance),
        "parquet_path": str(parquet_path.relative_to(out_dir)).replace("\\", "/"),
        "parquet_bytes": parquet_path.stat().st_size,
        "parquet_sha256": _sha256_file(parquet_path),
        "quality": dict(sorted(aggregate.items())),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    _write_json_atomic(lineage_path, record)
    return record


def _load_preprocess_entries(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("months"), list):
        raise ValueError(f"unsupported preprocess manifest format: {path}")
    return {str(entry["month"]): entry for entry in payload["months"]}


def _manifest_entry_compatible(
    entry: Mapping[str, object],
    airports: tuple[str, ...] | None,
    timezone_provenance: Mapping[str, str],
) -> bool:
    expected_airports = list(airports) if airports is not None else None
    return (
        entry.get("normalization_version") == NORMALIZATION_VERSION
        and entry.get("airports") == expected_airports
        and entry.get("timezone_provenance") == dict(timezone_provenance)
    )


def _write_preprocess_manifest(
    manifest_path: Path,
    entries: Mapping[str, Mapping[str, object]],
    airports: tuple[str, ...] | None,
    timezone_provenance: Mapping[str, str],
) -> None:
    payload = {
        "schema_version": 1,
        "normalization_version": NORMALIZATION_VERSION,
        "dataset": "BTS Marketing Carrier On-Time Performance normalized schema v1",
        "airports": list(airports) if airports is not None else None,
        "timezone_provenance": dict(timezone_provenance),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "months": [entries[key] for key in sorted(entries)],
    }
    _write_json_atomic(manifest_path, payload)


def preprocess_archives(
    archive_paths: Iterable[Path],
    out_dir: Path,
    *,
    airports: Iterable[str] | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    overwrite: bool = False,
    workers: int = 1,
) -> tuple[dict[str, object], ...]:
    """Process monthly archives and atomically update lineage from this thread."""

    paths = sorted((Path(path) for path in archive_paths), key=_archive_month)
    if not paths:
        raise ValueError("no monthly archives were provided")
    months = [_archive_month(path) for path in paths]
    duplicate_months = sorted(month for month, count in Counter(months).items() if count > 1)
    if duplicate_months:
        raise ValueError(f"duplicate monthly archives were provided: {duplicate_months}")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    selected_airports = (
        tuple(sorted({str(code).strip().upper() for code in airports if str(code).strip()}))
        if airports is not None
        else None
    )
    if selected_airports == ():
        raise ValueError("airports must contain at least one nonempty code")
    timezone_provenance = _timezone_provenance()
    requested_months = set(months)
    manifest_path = Path(out_dir) / PREPROCESS_MANIFEST
    entries = _load_preprocess_entries(manifest_path)
    incompatible_untouched = [
        month
        for month, entry in entries.items()
        if month not in requested_months
        and not _manifest_entry_compatible(entry, selected_airports, timezone_provenance)
    ]
    if incompatible_untouched:
        raise ValueError(
            "output manifest contains untouched months with a different scope, "
            f"normalization version, or timezone provenance: {incompatible_untouched}"
        )
    entries = {
        month: entry
        for month, entry in entries.items()
        if _manifest_entry_compatible(entry, selected_airports, timezone_provenance)
    }
    records: list[dict[str, object]] = []
    failures: list[str] = []

    def record_completion(record: dict[str, object]) -> None:
        records.append(record)
        stable_record = dict(record)
        stable_record.pop("status", None)
        entries[str(record["month"])] = stable_record
        _write_preprocess_manifest(manifest_path, entries, selected_airports, timezone_provenance)
        quality = record.get("quality", {})
        print(
            f"{str(record['status']).upper():9s} {record['month']} "
            f"raw={quality.get('raw_rows', '?')} retained={quality.get('retained_rows', '?')} "
            f"parquet={record['parquet_bytes']}",
            flush=True,
        )

    if workers == 1:
        for path in paths:
            record_completion(
                preprocess_archive(
                    path,
                    out_dir,
                    airports=selected_airports,
                    chunk_rows=chunk_rows,
                    overwrite=overwrite,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(paths))) as executor:
            future_to_path = {
                executor.submit(
                    preprocess_archive,
                    path,
                    out_dir,
                    airports=selected_airports,
                    chunk_rows=chunk_rows,
                    overwrite=overwrite,
                ): path
                for path in paths
            }
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    record_completion(future.result())
                except Exception as exc:  # noqa: BLE001 - preserve every monthly result
                    month = _archive_month(path)
                    failures.append(f"{month}: {exc}")
                    print(f"FAILED {month}: {exc}", flush=True)
    records.sort(key=lambda record: str(record["month"]))
    if failures:
        raise RuntimeError("one or more monthly preprocess jobs failed:\n" + "\n".join(failures))
    print(f"manifest: {manifest_path.resolve()}", flush=True)
    return tuple(records)


_POOL_SPECS = (
    (
        "route_carrier_bucket_season",
        ("origin", "dest", "carrier", "dep_hour_bucket", "season"),
    ),
    ("route_carrier_season", ("origin", "dest", "carrier", "season")),
    ("route_season", ("origin", "dest", "season")),
    ("route", ("origin", "dest")),
)
_SCHEDULE_REQUIRED = (
    "flight_id",
    "flight_date",
    "origin",
    "dest",
    "carrier",
    "operating_carrier",
    "operating_flight_number",
    "scheduled_departure_utc",
    "scheduled_arrival_utc",
    "dep_hour_bucket",
    "season",
)
_DONOR_REQUIRED = (
    "donor_id",
    "origin",
    "dest",
    "carrier",
    "dep_hour_bucket",
    "season",
    "dep_delay_min",
    "arr_delay_min",
    "actual_elapsed_min",
    "cancelled",
    "diverted",
    "div_reached_dest",
    "div_arr_delay_min",
    "div_actual_elapsed_min",
    "div_airport",
)
_SUMMARY_COLUMNS = ("cancelled", "diverted", "arr_delay_min")


class _OutcomeValues(NamedTuple):
    donor_id: object
    cancelled: object
    diverted: object
    dep_delay_min: object
    arr_delay_min: object
    actual_elapsed_min: object
    div_reached_dest: object
    div_arr_delay_min: object
    div_actual_elapsed_min: object
    div_airport: object


_OUTCOME_COLUMNS = _OutcomeValues._fields


def _require_columns(frame: pd.DataFrame, required: Sequence[str], label: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{label} must be a pandas DataFrame")  # noqa: TRY004
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _validated_integer(series: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.notna() & np.isfinite(numeric) & numeric.eq(numeric.round())
    if not valid.all():
        raise ValueError(f"{name} must contain finite integers")
    return numeric.astype(np.int64)


def _validated_bool(series: pd.Series, name: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"{name} must not contain nulls")
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    numeric = pd.to_numeric(series, errors="coerce")
    if not numeric.isin([0, 1]).all():
        raise ValueError(f"{name} must contain only booleans or 0/1")
    return numeric.eq(1)


def _canonical_pool_key(values: object | tuple[object, ...]) -> tuple[object, ...]:
    raw = values if isinstance(values, tuple) else (values,)
    result: list[object] = []
    for value in raw:
        if isinstance(value, (np.integer, int)):
            result.append(int(value))
        else:
            result.append(str(value))
    return tuple(result)


def _group_index(
    frame: pd.DataFrame, columns: Sequence[str]
) -> dict[tuple[object, ...], np.ndarray]:
    grouped = frame.groupby(list(columns), observed=True, sort=False, dropna=False).indices
    return {
        _canonical_pool_key(key): _sorted_int32_indices(indices) for key, indices in grouped.items()
    }


def _sorted_int32_indices(indices: object) -> np.ndarray:
    result = np.asarray(indices, dtype=np.int32)
    result.sort()
    return result


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


class HistoricalFlightData:
    """Schedule candidates backed by hierarchical empirical donor-row pools."""

    def __init__(
        self,
        schedule_rows: pd.DataFrame,
        donor_rows: pd.DataFrame,
        min_support: int = 30,
    ) -> None:
        if isinstance(min_support, bool) or not isinstance(min_support, int) or min_support < 1:
            raise ValueError("min_support must be a positive integer")
        _require_columns(schedule_rows, _SCHEDULE_REQUIRED, "schedule_rows")
        _require_columns(donor_rows, _DONOR_REQUIRED, "donor_rows")
        self.min_support = min_support

        schedule = schedule_rows.loc[:, list(_SCHEDULE_REQUIRED)].copy()
        donor = donor_rows.loc[:, list(_DONOR_REQUIRED)].copy()
        for column in (
            "flight_id",
            "flight_date",
            "origin",
            "dest",
            "carrier",
            "operating_carrier",
            "operating_flight_number",
            "season",
        ):
            schedule[column] = schedule[column].astype("string").fillna("").str.strip()
        for column in ("donor_id", "origin", "dest", "carrier", "season"):
            donor[column] = donor[column].astype("string").fillna("").str.strip()

        if schedule["flight_id"].eq("").any() or schedule["flight_id"].duplicated().any():
            raise ValueError("schedule flight_id values must be nonempty and unique")
        if donor["donor_id"].eq("").any() or donor["donor_id"].duplicated().any():
            raise ValueError("donor donor_id values must be nonempty and unique")
        for label, frame in (("schedule", schedule), ("donor", donor)):
            for column in ("origin", "dest", "carrier", "season"):
                if frame[column].eq("").any():
                    raise ValueError(f"{label} {column} values must be nonempty")

        schedule["scheduled_departure_utc"] = _validated_integer(
            schedule["scheduled_departure_utc"], "scheduled_departure_utc"
        )
        schedule["scheduled_arrival_utc"] = _validated_integer(
            schedule["scheduled_arrival_utc"], "scheduled_arrival_utc"
        )
        if schedule["scheduled_arrival_utc"].le(schedule["scheduled_departure_utc"]).any():
            raise ValueError("scheduled arrival must be after scheduled departure")
        schedule["dep_hour_bucket"] = _validated_integer(
            schedule["dep_hour_bucket"], "schedule dep_hour_bucket"
        )
        donor["dep_hour_bucket"] = _validated_integer(
            donor["dep_hour_bucket"], "donor dep_hour_bucket"
        )
        if not schedule["dep_hour_bucket"].between(0, 7).all():
            raise ValueError("schedule dep_hour_bucket must be in 0..7")
        if not donor["dep_hour_bucket"].between(0, 7).all():
            raise ValueError("donor dep_hour_bucket must be in 0..7")
        valid_seasons = {"DJF", "MAM", "JJA", "SON"}
        if not set(schedule["season"]).issubset(valid_seasons):
            raise ValueError("schedule season contains an invalid value")
        if not set(donor["season"]).issubset(valid_seasons):
            raise ValueError("donor season contains an invalid value")

        donor["cancelled"] = _validated_bool(donor["cancelled"], "cancelled")
        donor["diverted"] = _validated_bool(donor["diverted"], "diverted")
        donor["div_reached_dest"] = _validated_bool(donor["div_reached_dest"], "div_reached_dest")
        for column in (
            "dep_delay_min",
            "arr_delay_min",
            "actual_elapsed_min",
            "div_arr_delay_min",
            "div_actual_elapsed_min",
        ):
            donor[column] = pd.to_numeric(donor[column], errors="coerce").astype(np.float64)
        donor["div_airport"] = donor["div_airport"].astype("string")

        for column in ("origin", "dest", "carrier", "season"):
            donor[column] = donor[column].astype("category")
        self._donor = donor.reset_index(drop=True)
        # Keep the donor strings in their existing column and let pandas build
        # its compact hash engine lazily on the first source-authentication lookup.
        self._donor_id_index = pd.Index(self._donor["donor_id"], copy=False)
        self._summary_rows = self._donor.loc[:, list(_SUMMARY_COLUMNS)]
        self._pool_indices = {
            level: _group_index(self._donor, columns) for level, columns in _POOL_SPECS
        }
        self._summary_cache: dict[tuple[object, ...], OutcomeSummary] = {}
        self._pool_cache: dict[tuple[str, str, str, int, str], tuple[str, np.ndarray]] = {}

        candidates_by_origin: dict[str, list[FlightCandidate]] = {}
        source_flights_by_id: dict[str, FlightCandidate] = {}
        schedules_without_donors = 0
        for row in schedule.itertuples(index=False):
            candidate = FlightCandidate(
                flight_id=str(row.flight_id),
                origin=str(row.origin),
                dest=str(row.dest),
                carrier=str(row.carrier),
                scheduled_departure_utc=int(row.scheduled_departure_utc),
                scheduled_arrival_utc=int(row.scheduled_arrival_utc),
                flight_date=str(row.flight_date),
                dep_hour_bucket=int(row.dep_hour_bucket),
                season=str(row.season),
                operating_carrier=str(row.operating_carrier),
                operating_flight_number=str(row.operating_flight_number),
            )
            level, indices = self._pool_for(candidate)
            if level == "no_route" or len(indices) == 0:
                schedules_without_donors += 1
                continue
            source_flights_by_id[candidate.flight_id] = candidate
            candidates_by_origin.setdefault(candidate.origin, []).append(candidate)

        self._source_flights_by_id = source_flights_by_id

        self._candidates_by_origin: dict[str, tuple[FlightCandidate, ...]] = {}
        self._departures_by_origin: dict[str, tuple[int, ...]] = {}
        for origin, candidates in candidates_by_origin.items():
            ordered = tuple(
                sorted(
                    candidates,
                    key=lambda item: (
                        item.scheduled_departure_utc,
                        item.scheduled_arrival_utc,
                        item.carrier,
                        item.flight_id,
                    ),
                )
            )
            self._candidates_by_origin[origin] = ordered
            self._departures_by_origin[origin] = tuple(
                item.scheduled_departure_utc for item in ordered
            )

        airport_values = set(schedule["origin"]) | set(schedule["dest"])
        airport_values |= set(self._donor["origin"].astype("string"))
        airport_values |= set(self._donor["dest"].astype("string"))
        carrier_values = set(schedule["carrier"]) | set(self._donor["carrier"].astype("string"))
        self.airports = tuple(sorted(str(value) for value in airport_values if str(value)))
        self.carriers = tuple(sorted(str(value) for value in carrier_values if str(value)))
        self._metadata: dict[str, object] = {
            "min_support": min_support,
            "schedule_rows": len(schedule),
            "donor_rows": len(donor),
            "schedule_candidates_with_donor_coverage": int(
                sum(len(values) for values in self._candidates_by_origin.values())
            ),
            "schedule_rows_without_route_donors": schedules_without_donors,
        }

    @property
    def metadata(self) -> Mapping[str, object]:
        return MappingProxyType(self._metadata)

    def _pool_for(self, flight: FlightCandidate) -> tuple[str, np.ndarray]:
        cache_key = (
            flight.origin,
            flight.dest,
            flight.carrier,
            int(flight.dep_hour_bucket),
            flight.season,
        )
        cached = self._pool_cache.get(cache_key)
        if cached is not None:
            return cached
        values = {
            "origin": flight.origin,
            "dest": flight.dest,
            "carrier": flight.carrier,
            "dep_hour_bucket": int(flight.dep_hour_bucket),
            "season": flight.season,
        }
        route_indices: np.ndarray | None = None
        for level, columns in _POOL_SPECS:
            key = tuple(values[column] for column in columns)
            indices = self._pool_indices[level].get(key)
            if level == "route":
                route_indices = indices
            if indices is not None and len(indices) >= self.min_support:
                result = (level, indices)
                self._pool_cache[cache_key] = result
                return result
        if route_indices is not None and len(route_indices):
            result = ("route_low_support", route_indices)
        else:
            result = ("no_route", np.empty(0, dtype=np.int32))
        self._pool_cache[cache_key] = result
        return result

    def candidates(
        self,
        origin: str,
        earliest_utc: int,
        horizon_utc: int,
        limit: int = 64,
    ) -> tuple[FlightCandidate, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if horizon_utc < earliest_utc:
            return ()
        normalized_origin = str(origin).strip().upper()
        candidates = self._candidates_by_origin.get(normalized_origin, ())
        departures = self._departures_by_origin.get(normalized_origin, ())
        start = bisect_left(departures, int(earliest_utc))
        result: list[FlightCandidate] = []
        for candidate in candidates[start:]:
            if candidate.scheduled_departure_utc > horizon_utc:
                break
            result.append(candidate)
            if len(result) == limit:
                break
        return tuple(result)

    def source_flight(self, flight_id: str) -> FlightCandidate | None:
        """Return one canonical schedule that is selectable in this source."""

        if not isinstance(flight_id, str) or not flight_id:
            return None
        return self._source_flights_by_id.get(flight_id)

    def source_transition_outcome(self, flight_id: str, donor_id: str) -> SampledOutcome | None:
        """Resolve one canonical donor from the canonical flight's active pool."""

        flight = self.source_flight(flight_id)
        if flight is None or not isinstance(donor_id, str) or not donor_id:
            return None
        try:
            donor_index = self._donor_id_index.get_loc(donor_id)
        except KeyError:
            return None
        if not isinstance(donor_index, (int, np.integer)):
            raise TypeError("unique donor_id index returned an ambiguous location")

        level, indices = self._pool_for(flight)
        index = int(donor_index)
        position = int(np.searchsorted(indices, index))
        if position >= len(indices) or int(indices[position]) != index:
            return None
        return self._outcome(index, support=len(indices), fallback_level=level)

    def _outcome(
        self,
        index: int,
        *,
        support: int,
        fallback_level: str,
    ) -> SampledOutcome:
        values = _OutcomeValues(*(self._donor.at[index, column] for column in _OUTCOME_COLUMNS))
        return self._outcome_from_values(values, support=support, fallback_level=fallback_level)

    @staticmethod
    def _outcome_from_values(
        row: _OutcomeValues,
        *,
        support: int,
        fallback_level: str,
    ) -> SampledOutcome:
        div_airport = None if pd.isna(row.div_airport) else str(row.div_airport)
        return SampledOutcome(
            donor_id=str(row.donor_id),
            cancelled=bool(row.cancelled),
            diverted=bool(row.diverted),
            dep_delay_min=_optional_float(row.dep_delay_min),
            arr_delay_min=_optional_float(row.arr_delay_min),
            actual_elapsed_min=_optional_float(row.actual_elapsed_min),
            div_reached_dest=bool(row.div_reached_dest),
            div_arr_delay_min=_optional_float(row.div_arr_delay_min),
            div_actual_elapsed_min=_optional_float(row.div_actual_elapsed_min),
            div_airport=div_airport,
            support=support,
            fallback_level=fallback_level,
        )

    def historical_outcomes(self, flight: FlightCandidate) -> tuple[SampledOutcome, ...]:
        level, indices = self._pool_for(flight)
        support = len(indices)
        rows = self._donor.loc[:, list(_OUTCOME_COLUMNS)].iloc[indices]
        return tuple(
            self._outcome_from_values(
                _OutcomeValues._make(values), support=support, fallback_level=level
            )
            for values in rows.itertuples(index=False, name=None)
        )

    def sample_outcome(self, flight: FlightCandidate, rng: np.random.Generator) -> SampledOutcome:
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy.random.Generator")  # noqa: TRY004
        level, indices = self._pool_for(flight)
        if len(indices) == 0:
            raise ValueError(f"no historical donor route for {flight.origin}-{flight.dest}")
        donor_index = int(indices[int(rng.integers(0, len(indices)))])
        return self._outcome(donor_index, support=len(indices), fallback_level=level)

    def outcome_summary(self, flight: FlightCandidate) -> OutcomeSummary:
        level, indices = self._pool_for(flight)
        cache_key = (
            level,
            flight.origin,
            flight.dest,
            flight.carrier,
            int(flight.dep_hour_bucket),
            flight.season,
        )
        cached = self._summary_cache.get(cache_key)
        if cached is not None:
            return cached
        support = len(indices)
        if support == 0:
            result = OutcomeSummary(
                p_cancelled=0.0,
                p_diverted=0.0,
                mean_arrival_delay_min=0.0,
                support=0,
                fallback_level="no_route",
            )
        else:
            rows = self._summary_rows.iloc[indices]
            ordinary = ~rows["cancelled"] & ~rows["diverted"]
            arrival_delays = rows.loc[ordinary, "arr_delay_min"].to_numpy(dtype=np.float64)
            finite_delays = arrival_delays[np.isfinite(arrival_delays)]
            result = OutcomeSummary(
                p_cancelled=float(rows["cancelled"].mean()),
                p_diverted=float((rows["diverted"] & ~rows["cancelled"]).mean()),
                mean_arrival_delay_min=(float(finite_delays.mean()) if len(finite_delays) else 0.0),
                support=support,
                fallback_level=level,
            )
        self._summary_cache[cache_key] = result
        return result


def _parse_iso_date(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date string")  # noqa: TRY004
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date string") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{label} must be canonical YYYY-MM-DD")
    return value


def _parquet_date_bounds(path: Path) -> tuple[str, str]:
    try:
        dates = pd.read_parquet(path, columns=["flight_date"])["flight_date"]
    except Exception as exc:
        raise ValueError(f"cannot read normalized Parquet dataset at {path}: {exc}") from exc
    if dates.empty:
        raise ValueError(f"normalized Parquet dataset is empty: {path}")
    return str(dates.min()), str(dates.max())


def _read_parquet_slice(
    path: Path,
    start: str,
    end: str,
    airports: tuple[str, ...] | None,
) -> pd.DataFrame:
    filters: list[tuple[str, str, object]] = [
        ("flight_date", ">=", start),
        ("flight_date", "<=", end),
    ]
    if airports is not None:
        filters.extend(
            [
                ("origin", "in", list(airports)),
                ("dest", "in", list(airports)),
            ]
        )
    try:
        return pd.read_parquet(
            path,
            columns=list(NORMALIZED_COLUMNS),
            filters=filters,
            engine="pyarrow",
        )
    except Exception as exc:
        raise ValueError(
            f"cannot read normalized Parquet slice {start}..{end} at {path}: {exc}"
        ) from exc


def _month_span(start: str, end: str) -> tuple[str, ...]:
    current = date.fromisoformat(start).replace(day=1)
    last = date.fromisoformat(end).replace(day=1)
    result: list[str] = []
    while current <= last:
        result.append(current.strftime("%Y-%m"))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return tuple(result)


def _validate_load_lineage(
    path: Path,
    required_months: Iterable[str],
    requested_airports: tuple[str, ...] | None,
) -> dict[str, object]:
    if path.is_file():
        lineage_path = path.parent / LINEAGE_FILENAME
        if not lineage_path.exists():
            return {"lineage_validation": "unmanaged_single_file"}
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        if lineage.get("normalization_version") != NORMALIZATION_VERSION:
            raise ValueError(f"stale normalization lineage for {path}")
        if path.stat().st_size != lineage.get("parquet_bytes") or _sha256_file(path) != lineage.get(
            "parquet_sha256"
        ):
            raise ValueError(f"Parquet integrity does not match lineage: {path}")
        return {
            "lineage_validation": "single_file_verified",
            "dataset_airports": lineage.get("airports"),
            "timezone_provenance": lineage.get("timezone_provenance"),
        }

    manifest_path = path / PREPROCESS_MANIFEST
    if not manifest_path.is_file():
        raise ValueError(f"partitioned Parquet dataset lacks {PREPROCESS_MANIFEST}: {path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported preprocess manifest schema: {manifest_path}")
    if payload.get("normalization_version") != NORMALIZATION_VERSION:
        raise ValueError(f"stale normalization manifest: {manifest_path}")
    manifest_airports = payload.get("airports")
    expected_timezone_provenance = dict(_timezone_provenance())
    if payload.get("timezone_provenance") != expected_timezone_provenance:
        raise ValueError(f"timezone dependency provenance changed since preprocessing: {path}")
    if requested_airports is not None and manifest_airports is not None:
        unavailable = sorted(set(requested_airports) - set(manifest_airports))
        if unavailable:
            raise ValueError(
                f"requested airports were excluded during preprocessing: {unavailable}"
            )
    raw_entries = payload.get("months")
    if not isinstance(raw_entries, list):
        raise ValueError(  # noqa: TRY004 - public data contract uses ValueError
            f"preprocess manifest months must be a list: {manifest_path}"
        )
    entries = {str(entry.get("month")): entry for entry in raw_entries}
    if len(entries) != len(raw_entries):
        raise ValueError(f"duplicate month entries in preprocess manifest: {manifest_path}")

    root = path.resolve()
    listed_paths: set[str] = set()
    for month, entry in entries.items():
        if (
            entry.get("normalization_version") != NORMALIZATION_VERSION
            or entry.get("airports") != manifest_airports
            or entry.get("timezone_provenance") != expected_timezone_provenance
        ):
            raise ValueError(f"mixed preprocessing lineage at month {month}")
        relative_text = entry.get("parquet_path")
        if not isinstance(relative_text, str):
            raise ValueError(  # noqa: TRY004 - malformed lineage is a value error
                f"missing Parquet path in lineage for {month}"
            )
        partition_path = (root / Path(relative_text)).resolve()
        try:
            partition_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Parquet lineage escapes dataset root for {month}") from exc
        listed_paths.add(partition_path.relative_to(root).as_posix())

    actual_paths = {
        item.resolve().relative_to(root).as_posix()
        for item in path.glob("year=*/month=*/flights.parquet")
    }
    if actual_paths != listed_paths:
        raise ValueError(
            "Parquet partitions and preprocess manifest differ: "
            f"unlisted={sorted(actual_paths - listed_paths)}, "
            f"missing={sorted(listed_paths - actual_paths)}"
        )

    for month in sorted(set(required_months)):
        entry = entries.get(month)
        if entry is None:
            raise ValueError(f"preprocess manifest lacks required month {month}")
        partition_path = (root / Path(str(entry["parquet_path"]))).resolve()
        if partition_path.stat().st_size != entry.get("parquet_bytes") or _sha256_file(
            partition_path
        ) != entry.get("parquet_sha256"):
            raise ValueError(f"Parquet integrity does not match manifest for {month}")
        lineage_path = partition_path.parent / LINEAGE_FILENAME
        if not lineage_path.is_file():
            raise ValueError(f"missing monthly lineage for {month}: {lineage_path}")
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        for field in (
            "normalization_version",
            "month",
            "source_sha256",
            "airports",
            "timezone_provenance",
            "parquet_bytes",
            "parquet_sha256",
        ):
            if lineage.get(field) != entry.get(field):
                raise ValueError(f"manifest/monthly lineage mismatch for {month}: {field}")

    return {
        "lineage_validation": "partitioned_dataset_verified",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_months": len(entries),
        "dataset_airports": manifest_airports,
        "timezone_provenance": expected_timezone_provenance,
    }


def load_flight_data(
    path: Path,
    *,
    schedule_start: str,
    schedule_end: str,
    fit_start: str | None = None,
    fit_end: str | None = None,
    airports: Iterable[str] | None = None,
    min_support: int = 30,
) -> HistoricalFlightData:
    """Load inclusive schedule and fit slices from partitioned Parquet."""

    path = Path(path)
    schedule_start = _parse_iso_date(schedule_start, "schedule_start")
    schedule_end = _parse_iso_date(schedule_end, "schedule_end")
    if schedule_end < schedule_start:
        raise ValueError("schedule_end must not precede schedule_start")
    no_bound_default = fit_start is None and fit_end is None
    if no_bound_default:
        resolved_fit_start, resolved_fit_end = schedule_start, schedule_end
    else:
        dataset_start: str | None = None
        dataset_end: str | None = None
        if fit_start is None or fit_end is None:
            dataset_start, dataset_end = _parquet_date_bounds(path)
        resolved_fit_start = (
            _parse_iso_date(fit_start, "fit_start") if fit_start is not None else dataset_start
        )
        resolved_fit_end = (
            _parse_iso_date(fit_end, "fit_end") if fit_end is not None else dataset_end
        )
        assert resolved_fit_start is not None and resolved_fit_end is not None
    if resolved_fit_end < resolved_fit_start:
        raise ValueError("fit_end must not precede fit_start")

    selected_airports = None
    if airports is not None:
        selected_airports = tuple(
            sorted({str(code).strip().upper() for code in airports if str(code).strip()})
        )
        if not selected_airports:
            raise ValueError("airports must contain at least one nonempty code")
    required_months = set(_month_span(schedule_start, schedule_end))
    required_months.update(_month_span(resolved_fit_start, resolved_fit_end))
    lineage_metadata = _validate_load_lineage(path, required_months, selected_airports)
    schedule_rows = _read_parquet_slice(path, schedule_start, schedule_end, selected_airports)
    if schedule_start == resolved_fit_start and schedule_end == resolved_fit_end:
        donor_rows = schedule_rows.copy()
    else:
        donor_rows = _read_parquet_slice(
            path, resolved_fit_start, resolved_fit_end, selected_airports
        )
    result = HistoricalFlightData(schedule_rows, donor_rows, min_support=min_support)
    result._metadata.update(
        {
            "path": str(path.resolve()),
            "schedule_start": schedule_start,
            "schedule_end": schedule_end,
            "fit_start": resolved_fit_start,
            "fit_end": resolved_fit_end,
            "fit_scope_defaulted_to_schedule": no_bound_default,
            "airports_filter": list(selected_airports) if selected_airports else None,
            **lineage_metadata,
        }
    )
    return result
