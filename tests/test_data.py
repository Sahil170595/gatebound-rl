from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from flight_rl.data import (
    NORMALIZATION_VERSION,
    NORMALIZED_COLUMNS,
    RAW_COLUMNS,
    HistoricalFlightData,
    load_flight_data,
    normalize,
    preprocess_archive,
    preprocess_archives,
)
from flight_rl.models import SampledOutcome


def raw_row(**overrides: object) -> dict[str, object]:
    row = {column: None for column in RAW_COLUMNS}
    row.update(
        {
            "FlightDate": "2020-01-01",
            "Marketing_Airline_Network": "AA",
            "IATA_Code_Marketing_Airline": "AA",
            "Flight_Number_Marketing_Airline": 101,
            "Operating_Airline": "AA",
            "IATA_Code_Operating_Airline": "AA",
            "Flight_Number_Operating_Airline": 101,
            "Origin": "LAX",
            "Dest": "JFK",
            "CRSDepTime": 30,
            "CRSArrTime": 900,
            "CRSElapsedTime": 330,
            "DepDelay": 0,
            "ArrDelay": 0,
            "ActualElapsedTime": 330,
            "Cancelled": 0,
            "Diverted": 0,
            "DivReachedDest": None,
            "DivArrDelay": None,
            "DivActualElapsedTime": None,
            "Duplicate": "N",
        }
    )
    row.update(overrides)
    return row


def normalized_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "flight_id": "f1",
        "donor_id": "d1",
        "flight_date": "2020-01-01",
        "origin": "JFK",
        "dest": "LAX",
        "carrier": "AA",
        "operating_carrier": "AA",
        "operating_flight_number": "101",
        "scheduled_departure_utc": 1000,
        "scheduled_arrival_utc": 1300,
        "dep_hour_bucket": 0,
        "season": "DJF",
        "dep_delay_min": 0.0,
        "arr_delay_min": 0.0,
        "actual_elapsed_min": 300.0,
        "cancelled": False,
        "diverted": False,
        "div_reached_dest": False,
        "div_arr_delay_min": np.nan,
        "div_actual_elapsed_min": np.nan,
        "div_airport": pd.NA,
        "source_month": "2020-01",
        "source_file": "fixture.csv",
        "source_row": 1,
        "marketing_flight_number": "101",
    }
    row.update(overrides)
    return row


def test_normalize_uses_exact_epoch_minutes_with_pandas_3_units() -> None:
    frame = pd.DataFrame([raw_row()])

    result = normalize(
        frame,
        source_month="2020-01",
        source_file="fixture.csv",
        airport_timezones={"LAX": "America/Los_Angeles"},
    )

    expected = int(
        datetime(2020, 1, 1, 0, 30, tzinfo=ZoneInfo("America/Los_Angeles")).timestamp() // 60
    )
    assert result.loc[0, "scheduled_departure_utc"] == expected
    assert result.loc[0, "scheduled_arrival_utc"] == expected + 330
    assert result.loc[0, "dep_hour_bucket"] == 0


def test_normalize_accepts_2400_and_preserves_disruptions() -> None:
    frame = pd.DataFrame(
        [
            raw_row(
                Flight_Number_Marketing_Airline=1,
                CRSDepTime=2400,
                CRSArrTime=2400,
                Cancelled=1,
                DepDelay=None,
                ArrDelay=None,
                ActualElapsedTime=None,
            ),
            raw_row(
                Flight_Number_Marketing_Airline=2,
                Diverted=1,
                ArrDelay=None,
                ActualElapsedTime=None,
                DivReachedDest=0,
                DivArrDelay=None,
                DivActualElapsedTime=None,
                Div1Airport="DEN",
                Div2Airport="SLC",
            ),
        ]
    )

    result, quality = normalize(
        frame,
        source_month="2020-01",
        airport_timezones={"LAX": "America/Los_Angeles"},
        return_quality=True,
    )

    midnight = int(datetime(2020, 1, 2, tzinfo=ZoneInfo("America/Los_Angeles")).timestamp() // 60)
    assert result.loc[0, "scheduled_departure_utc"] == midnight
    assert bool(result.loc[0, "cancelled"])
    assert pd.isna(result.loc[0, "arr_delay_min"])
    assert bool(result.loc[1, "diverted"])
    assert result.loc[1, "div_airport"] == "SLC"
    assert quality["departure_2400"] == 1
    assert quality["arrival_2400"] == 1


def test_normalize_excludes_unrepresentable_schedule_values_per_row() -> None:
    valid = pd.DataFrame(
        [
            raw_row(Flight_Number_Marketing_Airline=1),
            raw_row(
                Flight_Number_Marketing_Airline=2,
                Cancelled=1,
                DepDelay=None,
                ArrDelay=None,
                ActualElapsedTime=None,
            ),
            raw_row(
                Flight_Number_Marketing_Airline=3,
                Diverted=1,
                ArrDelay=None,
                ActualElapsedTime=None,
                DivReachedDest=1,
                DivArrDelay=30,
                DivActualElapsedTime=360,
            ),
        ]
    )
    invalid = pd.DataFrame(
        [
            raw_row(Flight_Number_Marketing_Airline=4, CRSDepTime=np.inf),
            raw_row(Flight_Number_Marketing_Airline=5, CRSArrTime=-np.inf),
            raw_row(Flight_Number_Marketing_Airline=6, CRSElapsedTime=np.inf),
            raw_row(Flight_Number_Marketing_Airline=7, CRSDepTime=1e100),
            raw_row(Flight_Number_Marketing_Airline=8, CRSElapsedTime=1e100),
            raw_row(
                Flight_Number_Marketing_Airline=9,
                CRSElapsedTime=np.nextafter(float(1 << 63), 0),
            ),
        ]
    )

    expected = normalize(
        valid,
        source_month="2020-01",
        airport_timezones={"LAX": "America/Los_Angeles"},
    )
    result, quality = normalize(
        pd.concat([valid, invalid], ignore_index=True),
        source_month="2020-01",
        airport_timezones={"LAX": "America/Los_Angeles"},
        return_quality=True,
    )

    pd.testing.assert_frame_equal(result, expected)
    assert quality["raw_rows"] == 9
    assert quality["invalid_departure_hhmm"] == 2
    assert quality["invalid_arrival_hhmm"] == 1
    assert quality["invalid_crs_elapsed"] == 3
    assert quality["schedule_excluded"] == 6
    assert quality["retained_rows"] == 3
    assert quality["retained_ordinary_rows"] == 1
    assert quality["retained_cancelled_rows"] == 1
    assert quality["retained_diverted_rows"] == 1


def test_normalize_ids_are_chunk_invariant_and_ignore_trailing_empty_column() -> None:
    raw = pd.DataFrame([raw_row(Flight_Number_Marketing_Airline=number) for number in (10, 11, 12)])
    raw[""] = None
    full = normalize(
        raw,
        source_month="2020-01",
        airport_timezones={"LAX": "America/Los_Angeles"},
    )
    first = normalize(
        raw.iloc[:1],
        source_month="2020-01",
        source_row_offset=0,
        airport_timezones={"LAX": "America/Los_Angeles"},
    )
    second = normalize(
        raw.iloc[1:],
        source_month="2020-01",
        source_row_offset=1,
        airport_timezones={"LAX": "America/Los_Angeles"},
    )
    chunked = pd.concat([first, second], ignore_index=True)

    assert chunked["flight_id"].tolist() == full["flight_id"].tolist()
    assert full["flight_id"].tolist() == [
        "bts:2020-01:000000001",
        "bts:2020-01:000000002",
        "bts:2020-01:000000003",
    ]


def test_normalize_counts_and_excludes_dst_gaps_and_folds() -> None:
    raw = pd.DataFrame(
        [
            raw_row(
                FlightDate="2020-03-08",
                Origin="JFK",
                Dest="LAX",
                CRSDepTime=230,
                CRSArrTime=600,
            ),
            raw_row(
                FlightDate="2020-11-01",
                Origin="JFK",
                Dest="LAX",
                CRSDepTime=130,
                CRSArrTime=500,
            ),
        ]
    )

    result, quality = normalize(
        raw,
        airport_timezones={"JFK": "America/New_York"},
        return_quality=True,
    )

    assert result.empty
    assert quality["dst_nonexistent_departure"] == 1
    assert quality["dst_ambiguous_departure"] == 1
    assert quality["schedule_excluded"] == 2


def test_normalize_rejects_invalid_div_reached_dest() -> None:
    frame = pd.DataFrame([raw_row(Diverted=1, DivReachedDest=2)])

    with pytest.raises(ValueError, match="DivReachedDest"):
        normalize(
            frame,
            source_month="2020-01",
            airport_timezones={"LAX": "America/Los_Angeles"},
        )


def test_historical_data_uses_identical_hierarchical_joint_pool() -> None:
    schedule = pd.DataFrame(
        [
            normalized_row(flight_id="candidate", donor_id="unused"),
            normalized_row(
                flight_id="low",
                donor_id="unused-2",
                dest="SFO",
                scheduled_departure_utc=900,
                scheduled_arrival_utc=1200,
            ),
        ]
    )
    donors = pd.DataFrame(
        [
            normalized_row(
                flight_id=f"source-{index}",
                donor_id=f"d{index}",
                dep_hour_bucket=bucket,
                cancelled=cancelled,
                diverted=diverted,
                dep_delay_min=delay,
                arr_delay_min=delay,
            )
            for index, bucket, cancelled, diverted, delay in (
                (1, 0, True, False, np.nan),
                (2, 0, False, True, np.nan),
                (3, 1, False, False, 20.0),
            )
        ]
        + [
            normalized_row(
                flight_id=f"sfo-{index}",
                donor_id=f"s{index}",
                dest="SFO",
                carrier="DL",
                dep_hour_bucket=index,
            )
            for index in (0, 1)
        ]
    )
    data = HistoricalFlightData(schedule, donors, min_support=3)
    candidate = data.candidates("JFK", 0, 2000)[1]

    history = data.historical_outcomes(candidate)
    summary = data.outcome_summary(candidate)

    assert candidate.flight_id == "candidate"
    assert [outcome.donor_id for outcome in history] == ["d1", "d2", "d3"]
    assert summary.fallback_level == "route_carrier_season"
    assert summary.support == len(history) == 3
    assert summary.p_cancelled == 1 / 3
    assert summary.p_diverted == 1 / 3
    sampled = data.sample_outcome(candidate, np.random.default_rng(7))
    matching = next(outcome for outcome in history if outcome.donor_id == sampled.donor_id)
    assert sampled == matching

    low = data.candidates("JFK", 0, 2000)[0]
    assert low.flight_id == "low"
    assert data.outcome_summary(low).fallback_level == "route_low_support"
    assert data.outcome_summary(low).support == 2


def test_summary_applies_cancellation_precedence_and_ordinary_delay_only() -> None:
    schedule = pd.DataFrame([normalized_row(flight_id="candidate")])
    donors = pd.DataFrame(
        [
            normalized_row(
                donor_id="both",
                cancelled=True,
                diverted=True,
                arr_delay_min=999.0,
            ),
            normalized_row(
                donor_id="ordinary",
                flight_id="f2",
                source_row=2,
                arr_delay_min=10.0,
            ),
        ]
    )
    data = HistoricalFlightData(schedule, donors, min_support=1)

    summary = data.outcome_summary(data.candidates("JFK", 0, 2000)[0])

    assert summary.p_cancelled == 0.5
    assert summary.p_diverted == 0.0
    assert summary.mean_arrival_delay_min == 10.0


def test_historical_outcomes_bulk_conversion_preserves_every_sampled_field() -> None:
    schedule = pd.DataFrame([normalized_row(flight_id="candidate")])
    donors = pd.DataFrame(
        [
            normalized_row(
                donor_id="cancelled",
                cancelled=True,
                dep_delay_min=np.nan,
                arr_delay_min=np.nan,
                actual_elapsed_min=np.nan,
            ),
            normalized_row(
                donor_id="diverted",
                flight_id="f2",
                source_row=2,
                diverted=True,
                dep_delay_min=7.0,
                arr_delay_min=np.nan,
                actual_elapsed_min=np.nan,
                div_reached_dest=True,
                div_arr_delay_min=42.0,
                div_actual_elapsed_min=335.0,
                div_airport="DEN",
            ),
            normalized_row(
                donor_id="missing",
                flight_id="f3",
                source_row=3,
                dep_delay_min=np.nan,
                arr_delay_min=np.nan,
                actual_elapsed_min=np.nan,
            ),
            normalized_row(
                donor_id="ordinary",
                flight_id="f4",
                source_row=4,
                dep_delay_min=3.0,
                arr_delay_min=8.0,
                actual_elapsed_min=305.0,
            ),
        ]
    )
    data = HistoricalFlightData(schedule, donors, min_support=1)
    candidate = data.candidates("JFK", 0, 2000)[0]
    data._donor = data._donor.loc[:, list(reversed(data._donor.columns))]

    outcomes = data.historical_outcomes(candidate)

    assert outcomes == (
        SampledOutcome(
            donor_id="cancelled",
            cancelled=True,
            dep_delay_min=None,
            arr_delay_min=None,
            actual_elapsed_min=None,
            support=4,
            fallback_level="route_carrier_bucket_season",
        ),
        SampledOutcome(
            donor_id="diverted",
            diverted=True,
            dep_delay_min=7.0,
            arr_delay_min=None,
            actual_elapsed_min=None,
            div_reached_dest=True,
            div_arr_delay_min=42.0,
            div_actual_elapsed_min=335.0,
            div_airport="DEN",
            support=4,
            fallback_level="route_carrier_bucket_season",
        ),
        SampledOutcome(
            donor_id="missing",
            dep_delay_min=None,
            arr_delay_min=None,
            actual_elapsed_min=None,
            support=4,
            fallback_level="route_carrier_bucket_season",
        ),
        SampledOutcome(
            donor_id="ordinary",
            dep_delay_min=3.0,
            arr_delay_min=8.0,
            actual_elapsed_min=305.0,
            support=4,
            fallback_level="route_carrier_bucket_season",
        ),
    )
    sampled = data.sample_outcome(candidate, np.random.default_rng(11))
    assert sampled == next(item for item in outcomes if item.donor_id == sampled.donor_id)


def test_preprocess_archive_roundtrip_and_loader_scope(tmp_path: Path) -> None:
    archive = tmp_path / (
        "On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_2020_1.zip"
    )
    rows = pd.DataFrame(
        [
            raw_row(Flight_Number_Marketing_Airline=1),
            raw_row(Flight_Number_Marketing_Airline=2, ArrDelay=20),
        ]
    )
    rows[""] = None
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        output.writestr("month.csv", rows.to_csv(index=False))

    record = preprocess_archives(
        [archive],
        tmp_path / "processed",
        airports=("LAX", "JFK"),
        chunk_rows=1,
    )[0]
    data = load_flight_data(
        tmp_path / "processed",
        schedule_start="2020-01-01",
        schedule_end="2020-01-01",
        fit_start="2020-01-01",
        fit_end="2020-01-01",
        airports=("LAX", "JFK"),
        min_support=1,
    )

    assert record["normalization_version"] == NORMALIZATION_VERSION
    assert record["quality"]["raw_rows"] == 2
    assert record["quality"]["retained_rows"] == 2
    assert record["quality"]["retained_ordinary_rows"] == 2
    assert set(record["timezone_provenance"]) == {
        "airportsdata_version",
        "tzdata_version",
        "airport_timezone_map_sha256",
    }
    assert data.metadata["fit_scope_defaulted_to_schedule"] is False
    assert len(data.candidates("LAX", 0, 10**10)) == 2
    with pytest.raises(ValueError, match="requested airports were excluded"):
        load_flight_data(
            tmp_path / "processed",
            schedule_start="2020-01-01",
            schedule_end="2020-01-01",
            airports=("LAX", "JFK", "SFO"),
        )
    lineage = json.loads(
        (tmp_path / "processed" / "year=2020" / "month=01" / "_lineage.json").read_text()
    )
    assert lineage["source_sha256"] == record["source_sha256"]
    parquet = pd.read_parquet(tmp_path / "processed")
    assert parquet.columns[: len(NORMALIZED_COLUMNS)].tolist() == list(NORMALIZED_COLUMNS)
    parquet_path = tmp_path / "processed" / "year=2020" / "month=01" / "flights.parquet"
    (parquet_path.parent / "_flights.parquet.part").write_bytes(b"in progress")
    concurrent_read = load_flight_data(
        tmp_path / "processed",
        schedule_start="2020-01-01",
        schedule_end="2020-01-01",
        min_support=1,
    )
    assert len(concurrent_read.candidates("LAX", 0, 10**10)) == 2
    parquet_path.write_bytes(parquet_path.read_bytes()[:100])
    with pytest.raises(ValueError, match="Parquet integrity"):
        load_flight_data(
            tmp_path / "processed",
            schedule_start="2020-01-01",
            schedule_end="2020-01-01",
        )
    with pytest.raises(ValueError, match="Parquet integrity"):
        preprocess_archive(
            archive,
            tmp_path / "processed",
            airports=("LAX", "JFK"),
            chunk_rows=1,
        )


def test_preprocess_archives_parallelizes_months_with_one_manifest_writer(
    tmp_path: Path,
) -> None:
    archives: list[Path] = []
    for month in (1, 2):
        archive = tmp_path / (
            f"On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_2020_{month}.zip"
        )
        row = pd.DataFrame(
            [
                raw_row(
                    FlightDate=f"2020-{month:02d}-01",
                    Flight_Number_Marketing_Airline=month,
                )
            ]
        )
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
            output.writestr("month.csv", row.to_csv(index=False))
        archives.append(archive)

    records = preprocess_archives(
        archives,
        tmp_path / "parallel",
        airports=("LAX", "JFK"),
        chunk_rows=1,
        workers=3,
    )

    assert [record["month"] for record in records] == ["2020-01", "2020-02"]
    manifest = json.loads((tmp_path / "parallel" / "_preprocess_manifest.json").read_text())
    assert [entry["month"] for entry in manifest["months"]] == [
        "2020-01",
        "2020-02",
    ]
    assert all("status" not in entry for entry in manifest["months"])
    assert all(
        (tmp_path / "parallel" / entry["parquet_path"]).is_file() for entry in manifest["months"]
    )
    with pytest.raises(ValueError, match="duplicate monthly archives"):
        preprocess_archives([archives[0], archives[0]], tmp_path / "duplicates", workers=2)
