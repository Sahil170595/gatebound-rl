"""Held-out calibration diagnostics for hierarchical historical-flight priors.

The diagnostics operate on normalized flight rows. Pool selection deliberately
matches :class:`flight_rl.data.HistoricalFlightData`: total donor-row support,
including disruptions, determines the first eligible pool. Delay calibration
then conditions on ordinary rows with a finite arrival delay inside that same
selected pool.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

DELAY_THRESHOLD_MIN = 15.0
POOL_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "route_carrier_bucket_season",
        ("origin", "dest", "carrier", "dep_hour_bucket", "season"),
    ),
    ("route_carrier_season", ("origin", "dest", "carrier", "season")),
    ("route_season", ("origin", "dest", "season")),
    ("route", ("origin", "dest")),
)
POOL_LEVELS = tuple(level for level, _columns in POOL_SPECS) + (
    "route_low_support",
    "no_route",
)
CALIBRATION_COLUMNS = (
    "origin",
    "dest",
    "carrier",
    "dep_hour_bucket",
    "season",
    "cancelled",
    "diverted",
    "arr_delay_min",
)


def _require_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{label} must be a pandas DataFrame")  # noqa: TRY004
    missing = sorted(set(CALIBRATION_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{label} must contain at least one row")

    result = frame.loc[:, list(CALIBRATION_COLUMNS)].copy().reset_index(drop=True)
    for column in ("origin", "dest", "carrier", "season"):
        result[column] = result[column].astype("string").str.strip()
        if result[column].isna().any() or result[column].eq("").any():
            raise ValueError(f"{label} {column} values must be nonempty")

    bucket = pd.to_numeric(result["dep_hour_bucket"], errors="coerce")
    valid_bucket = bucket.notna() & np.isfinite(bucket) & bucket.eq(bucket.round())
    if not valid_bucket.all() or not bucket.between(0, 7).all():
        raise ValueError(f"{label} dep_hour_bucket must contain integers in 0..7")
    result["dep_hour_bucket"] = bucket.astype(np.int64)
    if not set(result["season"]).issubset({"DJF", "MAM", "JJA", "SON"}):
        raise ValueError(f"{label} season contains an invalid value")

    for column in ("cancelled", "diverted"):
        result[column] = _validated_bool(result[column], f"{label} {column}")
    result["arr_delay_min"] = pd.to_numeric(result["arr_delay_min"], errors="coerce").astype(
        np.float64
    )
    result["effective_diverted"] = result["diverted"] & ~result["cancelled"]
    result["ordinary"] = ~result["cancelled"] & ~result["diverted"]
    result["delay_known"] = result["ordinary"] & np.isfinite(result["arr_delay_min"])
    result["delayed_15"] = result["delay_known"] & result["arr_delay_min"].ge(DELAY_THRESHOLD_MIN)
    return result


def _validated_bool(series: pd.Series, name: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"{name} must not contain nulls")
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    numeric = pd.to_numeric(series, errors="coerce")
    if not numeric.isin([0, 1]).all():
        raise ValueError(f"{name} must contain only booleans or 0/1")
    return numeric.eq(1)


def _validate_options(min_support: int, bins: int | None = None) -> None:
    if isinstance(min_support, bool) or not isinstance(min_support, int) or min_support < 1:
        raise ValueError("min_support must be a positive integer")
    if bins is not None and (
        isinstance(bins, bool) or not isinstance(bins, int) or not 1 <= bins <= 100
    ):
        raise ValueError("bins must be an integer in 1..100")


def _pool_table(fit: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    grouped = fit.groupby(list(columns), observed=True, sort=False, dropna=False)
    table = grouped.agg(
        pool_support=("origin", "size"),
        cancellation_events=("cancelled", "sum"),
        diversion_events=("effective_diverted", "sum"),
        delay_known_support=("delay_known", "sum"),
        delay_events=("delayed_15", "sum"),
    )
    table["p_cancelled"] = table["cancellation_events"] / table["pool_support"]
    table["p_diverted"] = table["diversion_events"] / table["pool_support"]
    table["p_delayed_15"] = table["delay_events"] / table["delay_known_support"].replace(0, np.nan)
    return table.loc[
        :, ["pool_support", "delay_known_support", "p_cancelled", "p_diverted", "p_delayed_15"]
    ]


def _lookup_pool(
    table: pd.DataFrame, observed: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    keys = pd.MultiIndex.from_frame(observed.loc[:, list(columns)])
    matched = table.reindex(keys).reset_index(drop=True)
    matched["pool_support"] = matched["pool_support"].fillna(0).astype(np.int64)
    matched["delay_known_support"] = matched["delay_known_support"].fillna(0).astype(np.int64)
    return matched


def calibration_predictions(
    fit_rows: pd.DataFrame,
    observed_rows: pd.DataFrame,
    *,
    min_support: int = 30,
) -> pd.DataFrame:
    """Return row-aligned held-out events and fit-only hierarchical predictions.

    The selected pool is based on total fit-row support. A route with nonzero
    support below ``min_support`` is labeled ``route_low_support``; an unseen
    route is ``no_route`` and has no fitted prediction.
    """

    _validate_options(min_support)
    fit = _require_frame(fit_rows, "fit_rows")
    observed = _require_frame(observed_rows, "observed_rows")
    row_count = len(observed)
    selected_level = np.full(row_count, "no_route", dtype=object)
    selected_support = np.zeros(row_count, dtype=np.int64)
    selected_delay_support = np.zeros(row_count, dtype=np.int64)
    p_cancelled = np.full(row_count, np.nan, dtype=np.float64)
    p_diverted = np.full(row_count, np.nan, dtype=np.float64)
    p_delayed = np.full(row_count, np.nan, dtype=np.float64)

    for level, columns in POOL_SPECS:
        matched = _lookup_pool(_pool_table(fit, columns), observed, columns)
        support = matched["pool_support"].to_numpy(dtype=np.int64)
        unresolved = selected_level == "no_route"
        use = unresolved & (support >= min_support)
        if level == "route":
            low_support = unresolved & (support > 0) & (support < min_support)
            _assign_pool(
                low_support,
                "route_low_support",
                matched,
                selected_level,
                selected_support,
                selected_delay_support,
                p_cancelled,
                p_diverted,
                p_delayed,
            )
        _assign_pool(
            use,
            level,
            matched,
            selected_level,
            selected_support,
            selected_delay_support,
            p_cancelled,
            p_diverted,
            p_delayed,
        )

    return pd.DataFrame(
        {
            "pool_level": selected_level,
            "pool_support": selected_support,
            "pool_delay_known_support": selected_delay_support,
            "p_cancelled": p_cancelled,
            "p_diverted": p_diverted,
            "p_delayed_15": p_delayed,
            "observed_cancelled": observed["cancelled"].to_numpy(dtype=bool),
            "observed_diverted": observed["effective_diverted"].to_numpy(dtype=bool),
            "observed_ordinary": observed["ordinary"].to_numpy(dtype=bool),
            "observed_delay_known": observed["delay_known"].to_numpy(dtype=bool),
            "observed_delayed_15": observed["delayed_15"].to_numpy(dtype=bool),
        }
    )


def _assign_pool(
    use: np.ndarray,
    level: str,
    matched: pd.DataFrame,
    selected_level: np.ndarray,
    selected_support: np.ndarray,
    selected_delay_support: np.ndarray,
    p_cancelled: np.ndarray,
    p_diverted: np.ndarray,
    p_delayed: np.ndarray,
) -> None:
    if not use.any():
        return
    selected_level[use] = level
    selected_support[use] = matched.loc[use, "pool_support"].to_numpy(dtype=np.int64)
    selected_delay_support[use] = matched.loc[use, "delay_known_support"].to_numpy(dtype=np.int64)
    p_cancelled[use] = matched.loc[use, "p_cancelled"].to_numpy(dtype=np.float64)
    p_diverted[use] = matched.loc[use, "p_diverted"].to_numpy(dtype=np.float64)
    p_delayed[use] = matched.loc[use, "p_delayed_15"].to_numpy(dtype=np.float64)


def _rate(values: np.ndarray, mask: np.ndarray) -> float | None:
    return float(values[mask].mean()) if mask.any() else None


def _difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else float(left - right)


def _brier(
    events: np.ndarray, predictions: np.ndarray, mask: np.ndarray
) -> tuple[float | None, float | None]:
    if not mask.any():
        return None, None
    errors = np.square(predictions[mask] - events[mask].astype(np.float64))
    return float(errors.mean()), float(errors.sum())


def _reliability_bins(
    events: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    bins: int,
) -> list[dict[str, Any]]:
    selected_predictions = predictions[mask]
    selected_events = events[mask]
    indices = np.minimum((selected_predictions * bins).astype(np.int64), bins - 1)
    result: list[dict[str, Any]] = []
    for index in range(bins):
        in_bin = indices == index
        count = int(in_bin.sum())
        result.append(
            {
                "index": index,
                "lower_bound": index / bins,
                "upper_bound": (index + 1) / bins,
                "upper_bound_inclusive": index == bins - 1,
                "count": count,
                "mean_predicted_rate": (
                    float(selected_predictions[in_bin].mean()) if count else None
                ),
                "observed_rate": float(selected_events[in_bin].mean()) if count else None,
            }
        )
    return result


def _outcome_report(
    *,
    name: str,
    definition: str,
    conditioning: str,
    events: np.ndarray,
    eligible: np.ndarray,
    outcome_known: np.ndarray,
    predictions: np.ndarray,
    fit_rate: float | None,
    fit_denominator: int,
    bins: int,
) -> dict[str, Any]:
    known = eligible & outcome_known
    prediction_known = np.isfinite(predictions)
    scored = known & prediction_known
    observed_rate = _rate(events, known)
    scored_observed_rate = _rate(events, scored)
    hierarchical_brier, hierarchical_brier_sum = _brier(events, predictions, scored)
    if fit_rate is None:
        global_brier, global_brier_sum = None, None
    else:
        global_predictions = np.full(len(events), fit_rate, dtype=np.float64)
        global_brier, global_brier_sum = _brier(events, global_predictions, scored)
    return {
        "name": name,
        "definition": definition,
        "conditioning": conditioning,
        "fit_global_rate": fit_rate,
        "fit_rate_denominator": int(fit_denominator),
        "observed_rate": observed_rate,
        "observed_minus_fit_rate": _difference(observed_rate, fit_rate),
        "mean_hierarchical_prediction_on_scored_rows": _rate(predictions, scored),
        "observed_rate_on_scored_rows": scored_observed_rate,
        "denominators": {
            "observed_total_rows": len(events),
            "eligible_outcome_rows": int(eligible.sum()),
            "known_outcomes": int(known.sum()),
            "unknown_outcomes": int((eligible & ~outcome_known).sum()),
            "excluded_by_conditioning": int((~eligible).sum()),
            "known_pool_predictions": int(scored.sum()),
            "unknown_pool_predictions": int((known & ~prediction_known).sum()),
            "brier_rows": int(scored.sum()),
        },
        "brier": {
            "hierarchical_pool": hierarchical_brier,
            "hierarchical_pool_squared_error_sum": hierarchical_brier_sum,
            "global_fit_rate_baseline_same_rows": global_brier,
            "global_fit_rate_baseline_squared_error_sum": global_brier_sum,
        },
        "reliability_bins": _reliability_bins(events, predictions, scored, bins),
    }


def _support_summary(values: np.ndarray) -> dict[str, Any]:
    if not len(values):
        return {"rows": 0, "minimum": None, "maximum": None, "mean": None}
    return {
        "rows": len(values),
        "minimum": int(values.min()),
        "maximum": int(values.max()),
        "mean": float(values.mean()),
    }


def _row_counts(frame: pd.DataFrame) -> dict[str, int]:
    cancelled = frame["cancelled"].to_numpy(dtype=bool)
    diverted = frame["diverted"].to_numpy(dtype=bool)
    ordinary = frame["ordinary"].to_numpy(dtype=bool)
    delay_known = frame["delay_known"].to_numpy(dtype=bool)
    return {
        "rows": len(frame),
        "cancelled_rows": int(cancelled.sum()),
        "raw_diverted_rows": int(diverted.sum()),
        "cancelled_and_diverted_rows": int((cancelled & diverted).sum()),
        "effective_diverted_rows": int((diverted & ~cancelled).sum()),
        "ordinary_rows": int(ordinary.sum()),
        "ordinary_finite_arrival_delay_rows": int(delay_known.sum()),
        "ordinary_unknown_arrival_delay_rows": int((ordinary & ~delay_known).sum()),
    }


def build_calibration_report(
    fit_rows: pd.DataFrame,
    observed_rows: pd.DataFrame,
    *,
    min_support: int = 30,
    bins: int = 10,
) -> dict[str, Any]:
    """Compare fit-only empirical priors with untouched held-out flight rows."""

    _validate_options(min_support, bins)
    fit = _require_frame(fit_rows, "fit_rows")
    observed = _require_frame(observed_rows, "observed_rows")
    predictions = calibration_predictions(fit, observed, min_support=min_support)

    fit_all = np.ones(len(fit), dtype=bool)
    fit_delay_known = fit["delay_known"].to_numpy(dtype=bool)
    fit_cancelled = fit["cancelled"].to_numpy(dtype=bool)
    fit_diverted = fit["effective_diverted"].to_numpy(dtype=bool)
    fit_delayed = fit["delayed_15"].to_numpy(dtype=bool)
    fit_rates = {
        "cancelled": _rate(fit_cancelled, fit_all),
        "diverted": _rate(fit_diverted, fit_all),
        "delayed_15": _rate(fit_delayed, fit_delay_known),
    }

    observed_count = len(predictions)
    all_observed = np.ones(observed_count, dtype=bool)
    ordinary = predictions["observed_ordinary"].to_numpy(dtype=bool)
    delay_known = predictions["observed_delay_known"].to_numpy(dtype=bool)
    outcomes = {
        "cancelled": _outcome_report(
            name="cancelled",
            definition="BTS Cancelled flag equals one",
            conditioning="all held-out rows; cancellation retains precedence over diversion",
            events=predictions["observed_cancelled"].to_numpy(dtype=bool),
            eligible=all_observed,
            outcome_known=all_observed,
            predictions=predictions["p_cancelled"].to_numpy(dtype=np.float64),
            fit_rate=fit_rates["cancelled"],
            fit_denominator=len(fit),
            bins=bins,
        ),
        "diverted": _outcome_report(
            name="diverted",
            definition="BTS Diverted flag equals one and Cancelled flag equals zero",
            conditioning="all held-out rows; cancellation retains precedence over diversion",
            events=predictions["observed_diverted"].to_numpy(dtype=bool),
            eligible=all_observed,
            outcome_known=all_observed,
            predictions=predictions["p_diverted"].to_numpy(dtype=np.float64),
            fit_rate=fit_rates["diverted"],
            fit_denominator=len(fit),
            bins=bins,
        ),
        "delayed_15": _outcome_report(
            name="delayed_15",
            definition=f"finite BTS ArrDelay greater than or equal to {DELAY_THRESHOLD_MIN:g} minutes",
            conditioning="rows with Cancelled=0, Diverted=0, and finite ArrDelay",
            events=predictions["observed_delayed_15"].to_numpy(dtype=bool),
            eligible=ordinary,
            outcome_known=delay_known,
            predictions=predictions["p_delayed_15"].to_numpy(dtype=np.float64),
            fit_rate=fit_rates["delayed_15"],
            fit_denominator=int(fit_delay_known.sum()),
            bins=bins,
        ),
    }

    level_counts = {level: int(predictions["pool_level"].eq(level).sum()) for level in POOL_LEVELS}
    by_level: dict[str, Any] = {}
    for level in POOL_LEVELS:
        selected = predictions["pool_level"].eq(level).to_numpy()
        by_level[level] = {
            "selected_rows": int(selected.sum()),
            "total_pool_support": _support_summary(
                predictions.loc[selected, "pool_support"].to_numpy(dtype=np.int64)
            ),
            "ordinary_finite_delay_pool_support": _support_summary(
                predictions.loc[selected, "pool_delay_known_support"].to_numpy(dtype=np.int64)
            ),
        }

    return {
        "schema_version": 1,
        "method": {
            "min_support": min_support,
            "pool_order": [level for level, _columns in POOL_SPECS],
            "pool_support_basis": "all fit rows, including cancellations and diversions",
            "low_support_rule": "use a nonempty route pool when every pool has support below min_support",
            "unseen_route_rule": "no fitted prediction",
            "delay_threshold_min": DELAY_THRESHOLD_MIN,
            "reliability_bin_count": bins,
            "reliability_bin_strategy": (
                "equal-width probability intervals; left-closed/right-open except final bin includes 1"
            ),
        },
        "fit": {**_row_counts(fit), "global_rates": fit_rates},
        "observed": _row_counts(observed),
        "pool_selection": {
            "row_counts": level_counts,
            "rows_with_route_support": int(observed_count - level_counts["no_route"]),
            "rows_without_route_support": level_counts["no_route"],
            "by_level": by_level,
        },
        "outcomes": outcomes,
        "claim_scope": (
            "Retrospective flight-row calibration of fit-only empirical pool rates against held-out "
            "observations. It does not estimate passenger-level itinerary reliability or causal effects."
        ),
    }
