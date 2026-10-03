from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from flight_rl.calibration import build_calibration_report, calibration_predictions
from flight_rl.data import HistoricalFlightData
from flight_rl.models import FlightCandidate


def row(
    *,
    origin: str = "JFK",
    dest: str = "LAX",
    carrier: str = "AA",
    bucket: int = 0,
    season: str = "DJF",
    cancelled: bool = False,
    diverted: bool = False,
    delay: float = 0.0,
) -> dict[str, object]:
    return {
        "origin": origin,
        "dest": dest,
        "carrier": carrier,
        "dep_hour_bucket": bucket,
        "season": season,
        "cancelled": cancelled,
        "diverted": diverted,
        "arr_delay_min": delay,
    }


def donor(row_values: dict[str, object], index: int) -> dict[str, object]:
    return {
        "donor_id": f"d{index}",
        **row_values,
        "dep_delay_min": row_values["arr_delay_min"],
        "actual_elapsed_min": 360.0,
        "div_reached_dest": False,
        "div_arr_delay_min": np.nan,
        "div_actual_elapsed_min": np.nan,
        "div_airport": pd.NA,
    }


def candidate(row_values: dict[str, object], index: int) -> FlightCandidate:
    return FlightCandidate(
        flight_id=f"f{index}",
        origin=str(row_values["origin"]),
        dest=str(row_values["dest"]),
        carrier=str(row_values["carrier"]),
        scheduled_departure_utc=index * 1000,
        scheduled_arrival_utc=index * 1000 + 360,
        flight_date="2025-01-01",
        dep_hour_bucket=int(row_values["dep_hour_bucket"]),
        season=str(row_values["season"]),
    )


def test_pool_selection_matches_historical_data_fallback_and_event_rates() -> None:
    fit_values = [
        row(bucket=0, delay=20.0),
        row(bucket=0, cancelled=True, delay=np.nan),
        row(bucket=1, diverted=True, delay=np.nan),
        row(carrier="DL", bucket=4, delay=0.0),
        row(season="MAM", carrier="DL", bucket=4, delay=30.0),
        row(origin="BOS", dest="SFO", carrier="B6", delay=0.0),
        row(origin="BOS", dest="SFO", carrier="B6", delay=15.0),
    ]
    observed_values = [
        row(bucket=0),
        row(carrier="UA", bucket=7),
        row(season="SON", carrier="UA", bucket=7),
        row(origin="BOS", dest="SFO", carrier="UA"),
        row(origin="MIA", dest="SEA", carrier="AA"),
    ]
    fit = pd.DataFrame(fit_values)
    observed = pd.DataFrame(observed_values)

    predictions = calibration_predictions(fit, observed, min_support=3)

    assert predictions["pool_level"].tolist() == [
        "route_carrier_season",
        "route_season",
        "route",
        "route_low_support",
        "no_route",
    ]
    assert predictions["pool_support"].tolist() == [3, 4, 5, 2, 0]
    assert predictions.loc[0, "p_cancelled"] == pytest.approx(1 / 3)
    assert predictions.loc[0, "p_diverted"] == pytest.approx(1 / 3)
    assert predictions.loc[0, "p_delayed_15"] == 1.0
    assert math.isnan(predictions.loc[4, "p_cancelled"])

    schedule = pd.DataFrame(
        [
            {
                **value,
                "flight_id": f"f{index}",
                "flight_date": "2025-01-01",
                "operating_carrier": str(value["carrier"]),
                "operating_flight_number": str(index),
                "scheduled_departure_utc": index * 1000,
                "scheduled_arrival_utc": index * 1000 + 360,
            }
            for index, value in enumerate(observed_values, start=1)
        ]
    )
    historical = HistoricalFlightData(
        schedule,
        pd.DataFrame([donor(value, index) for index, value in enumerate(fit_values)]),
        min_support=3,
    )
    for index, values in enumerate(observed_values[:-1]):
        flight = candidate(values, index + 1)
        summary = historical.outcome_summary(flight)
        assert predictions.loc[index, "pool_level"] == summary.fallback_level
        assert predictions.loc[index, "pool_support"] == summary.support
        assert predictions.loc[index, "p_cancelled"] == pytest.approx(summary.p_cancelled)
        assert predictions.loc[index, "p_diverted"] == pytest.approx(summary.p_diverted)
        history = historical.historical_outcomes(flight)
        finite_ordinary = [
            outcome.arr_delay_min
            for outcome in history
            if not outcome.cancelled
            and not outcome.diverted
            and outcome.arr_delay_min is not None
            and math.isfinite(outcome.arr_delay_min)
        ]
        expected_delay_rate = sum(delay >= 15 for delay in finite_ordinary) / len(finite_ordinary)
        assert predictions.loc[index, "p_delayed_15"] == expected_delay_rate


def test_report_preserves_disruptions_and_makes_delay_denominators_explicit() -> None:
    fit = pd.DataFrame(
        [
            row(cancelled=True, diverted=True, delay=999.0),
            row(diverted=True, delay=999.0),
            row(delay=20.0),
            row(delay=np.nan),
        ]
    )
    observed = pd.DataFrame(
        [
            row(cancelled=True, diverted=True, delay=999.0),
            row(diverted=True, delay=999.0),
            row(delay=20.0),
            row(delay=0.0),
            row(delay=np.inf),
        ]
    )

    report = build_calibration_report(fit, observed, min_support=1, bins=5)

    assert report["fit"]["rows"] == 4
    assert report["fit"]["effective_diverted_rows"] == 1
    assert report["fit"]["ordinary_finite_arrival_delay_rows"] == 1
    assert report["outcomes"]["cancelled"]["fit_global_rate"] == 0.25
    assert report["outcomes"]["diverted"]["fit_global_rate"] == 0.25
    assert report["outcomes"]["delayed_15"]["fit_global_rate"] == 1.0
    delay_denominators = report["outcomes"]["delayed_15"]["denominators"]
    assert delay_denominators == {
        "observed_total_rows": 5,
        "eligible_outcome_rows": 3,
        "known_outcomes": 2,
        "unknown_outcomes": 1,
        "excluded_by_conditioning": 2,
        "known_pool_predictions": 2,
        "unknown_pool_predictions": 0,
        "brier_rows": 2,
    }
    assert report["outcomes"]["delayed_15"]["observed_rate"] == 0.5
    assert report["outcomes"]["delayed_15"]["brier"]["hierarchical_pool"] == 0.5


def test_brier_baseline_uses_same_scored_rows_and_bins_include_probability_one() -> None:
    fit = pd.DataFrame(
        [
            row(origin="JFK", dest="LAX", cancelled=True, delay=np.nan),
            row(origin="BOS", dest="SFO", delay=0.0),
        ]
    )
    observed = pd.DataFrame(
        [
            row(origin="JFK", dest="LAX", cancelled=True),
            row(origin="BOS", dest="SFO", cancelled=True),
            row(origin="MIA", dest="SEA", cancelled=False),
        ]
    )

    report = build_calibration_report(fit, observed, min_support=1, bins=2)
    cancellation = report["outcomes"]["cancelled"]

    assert cancellation["denominators"]["known_outcomes"] == 3
    assert cancellation["denominators"]["unknown_pool_predictions"] == 1
    assert cancellation["denominators"]["brier_rows"] == 2
    assert cancellation["brier"]["hierarchical_pool"] == 0.5
    assert cancellation["brier"]["global_fit_rate_baseline_same_rows"] == 0.25
    assert [entry["count"] for entry in cancellation["reliability_bins"]] == [1, 1]
    final_bin = cancellation["reliability_bins"][-1]
    assert final_bin["upper_bound"] == 1.0
    assert final_bin["upper_bound_inclusive"] is True
    assert final_bin["mean_predicted_rate"] == 1.0


def test_delay_prediction_stays_unknown_when_selected_pool_has_no_known_delay() -> None:
    fit = pd.DataFrame(
        [
            row(cancelled=True, delay=np.nan),
            row(diverted=True, delay=np.nan),
        ]
    )
    observed = pd.DataFrame([row(delay=20.0)])

    report = build_calibration_report(fit, observed, min_support=1)

    delayed = report["outcomes"]["delayed_15"]
    assert delayed["fit_global_rate"] is None
    assert delayed["denominators"]["known_outcomes"] == 1
    assert delayed["denominators"]["unknown_pool_predictions"] == 1
    assert delayed["denominators"]["brier_rows"] == 0
    assert delayed["brier"]["hierarchical_pool"] is None
    assert delayed["brier"]["global_fit_rate_baseline_same_rows"] is None


@pytest.mark.parametrize(
    ("fit", "observed", "min_support", "bins", "match"),
    [
        (pd.DataFrame(), pd.DataFrame([row()]), 1, 10, "fit_rows is missing"),
        (
            pd.DataFrame([row()]),
            pd.DataFrame([row(cancelled=None)]),
            1,
            10,
            "must not contain nulls",
        ),
        (pd.DataFrame([row()]), pd.DataFrame([row(bucket=8)]), 1, 10, "integers in 0..7"),
        (pd.DataFrame([row()]), pd.DataFrame([row()]), 0, 10, "positive integer"),
        (pd.DataFrame([row()]), pd.DataFrame([row()]), 1, 0, "1..100"),
    ],
)
def test_invalid_inputs_fail_loudly(fit, observed, min_support, bins, match) -> None:
    with pytest.raises(ValueError, match=match):
        build_calibration_report(fit, observed, min_support=min_support, bins=bins)
