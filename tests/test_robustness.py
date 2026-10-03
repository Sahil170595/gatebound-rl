from __future__ import annotations

import json

import numpy as np
import pytest

from flight_rl.baselines import (
    NonstopFirstPolicy,
    RandomPolicy,
    ShortestScheduledArrivalPolicy,
)
from flight_rl.fixtures import DemoFlightData, make_demo_scenario
from flight_rl.models import FlightCandidate, SampledOutcome, TripRequest, utc_minutes
from flight_rl.robustness import (
    CRITERION_NAMES,
    _approx_normal_mean_interval95,
    paired_condition_differences,
    reweighted_score,
    run_paired_scenarios,
    weight_profiles,
)
from flight_rl.scenarios import ScenarioPoolCache
from flight_rl.verifier import LEGACY_SIX_V1_WEIGHTS


def _direct_data() -> tuple[TripRequest, DemoFlightData]:
    ready = utc_minutes("2024-01-01T12:00:00Z")
    request = TripRequest("AAA", "BBB", ready, ready + 180, ready + 300, max_attempts=1)
    flight = FlightCandidate(
        "direct",
        "AAA",
        "BBB",
        "ZZ",
        ready + 60,
        ready + 120,
        "2024-01-01",
        2,
        "DJF",
    )
    outcomes = (
        SampledOutcome("on-time", actual_elapsed_min=60.0),
        SampledOutcome(
            "late",
            dep_delay_min=120.0,
            arr_delay_min=120.0,
            actual_elapsed_min=60.0,
        ),
        SampledOutcome("cancelled", cancelled=True, dep_delay_min=None, arr_delay_min=None),
    )
    return request, DemoFlightData((flight,), {flight.flight_id: outcomes})


def test_reweighted_score_uses_explicit_relative_weights() -> None:
    raw = {
        "arrived": 1.0,
        "on_time_arrival": 0.0,
        "total_delay": 0.5,
        "cancellation_exposure": 1.0,
        "connections_count": 1.0,
        "connection_buffer": 1.0,
    }

    assert reweighted_score(raw, LEGACY_SIX_V1_WEIGHTS) == pytest.approx(0.70)
    assert reweighted_score(
        raw, {name: 2 * weight for name, weight in LEGACY_SIX_V1_WEIGHTS.items()}
    ) == pytest.approx(0.70)


@pytest.mark.parametrize(
    "weights",
    [
        {name: 0.0 for name in CRITERION_NAMES},
        {**LEGACY_SIX_V1_WEIGHTS, "arrived": -1.0},
        {**LEGACY_SIX_V1_WEIGHTS, "arrived": np.nan},
        {**LEGACY_SIX_V1_WEIGHTS, "arrived": np.inf},
        {**LEGACY_SIX_V1_WEIGHTS, "arrived": True},
        {name: 1e308 for name in CRITERION_NAMES},
        {"arrived": 1.0},
    ],
)
def test_reweighted_score_rejects_invalid_weights(weights: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="weight"):
        reweighted_score({name: 1.0 for name in CRITERION_NAMES}, weights)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.1, 1.1, True, "1"])
def test_reweighted_score_rejects_invalid_raw_scores(bad: object) -> None:
    raw: dict[str, object] = {name: 1.0 for name in CRITERION_NAMES}
    raw["total_delay"] = bad
    with pytest.raises(ValueError, match="raw score"):
        reweighted_score(raw, LEGACY_SIX_V1_WEIGHTS)


def test_weight_profiles_preserve_the_frozen_legacy_profile() -> None:
    custom = {name: 0.25 for name in CRITERION_NAMES}
    profiles = weight_profiles({"equal": custom})

    assert profiles["legacy_six_v1"] == dict(LEGACY_SIX_V1_WEIGHTS)
    assert profiles["equal"] == custom
    with pytest.raises(ValueError, match="legacy_six_v1"):
        weight_profiles({"legacy_six_v1": custom})


def test_normal_mean_interval_does_not_claim_certainty_without_variation() -> None:
    assert _approx_normal_mean_interval95([0.5], lower=0.0, upper=1.0) is None
    assert _approx_normal_mean_interval95([0.5, 0.5], lower=0.0, upper=1.0) is None
    interval = _approx_normal_mean_interval95([0.0, 1.0], lower=0.0, upper=1.0)
    assert interval is not None
    assert interval[0] < 0.5 < interval[1]


def test_paired_run_is_repeatable_json_safe_and_policy_order_independent() -> None:
    request, data = make_demo_scenario()
    keys = tuple(range(30, 42))
    forward_factories = {
        "random": lambda seed: RandomPolicy(seed),
        "nonstop_first": lambda _seed: NonstopFirstPolicy(),
        "shortest_scheduled": lambda _seed: ShortestScheduledArrivalPolicy(),
    }
    reverse_factories = dict(reversed(tuple(forward_factories.items())))

    first = run_paired_scenarios(
        request=request,
        data=data,
        policy_factories=forward_factories,
        scenario_keys=keys,
        dependence=0.5,
        trace_count=2,
    )
    repeated = run_paired_scenarios(
        request=request,
        data=data,
        policy_factories=forward_factories,
        scenario_keys=keys,
        dependence=0.5,
        trace_count=2,
    )
    reordered = run_paired_scenarios(
        request=request,
        data=data,
        policy_factories=reverse_factories,
        scenario_keys=keys,
        dependence=0.5,
        trace_count=2,
    )

    assert first == repeated
    assert first["policies"] == reordered["policies"]
    assert first["scenario"] == reordered["scenario"]
    assert json.loads(json.dumps(first, allow_nan=False)) == first
    assert sum(first["scenario"]["coupling_regime_counts"].values()) == len(keys)
    assert "different chosen flights" in first["scenario"]["pairing"]
    assert "not reoptimized" in first["weight_analysis"]
    assert "historical model error" in first["uncertainty_scope"]
    assert "zero empirical variance" in first["uncertainty_scope"]
    assert first["score_profile"] == "legacy_six_v1"
    for policy in first["policies"].values():
        assert policy["score_profile"] == "legacy_six_v1"
        assert policy["mean_score"] == pytest.approx(policy["mean_return"])
        assert policy["mean_legacy_six_v1"] == pytest.approx(policy["mean_return"])
        assert len(policy["traces"]) == 2
        assert [row["scenario_key"] for row in policy["scenario_outcomes"]] == list(keys)

    condition_difference = paired_condition_differences(first, repeated)
    assert condition_difference["pairs"] == len(keys)
    for policy in condition_difference["policies"].values():
        assert policy["on_time_arrival"]["mean_second_minus_first"] == 0.0
        assert policy["on_time_arrival"]["paired_approx_normal_interval95"] is None


def test_profiles_reuse_one_outcome_record_and_do_not_create_policy_runs() -> None:
    request, data = _direct_data()
    factory_calls = 0

    def policy_factory(_seed: int) -> NonstopFirstPolicy:
        nonlocal factory_calls
        factory_calls += 1
        return NonstopFirstPolicy()

    report = run_paired_scenarios(
        request=request,
        data=data,
        policy_factories={"nonstop": policy_factory},
        scenario_keys=(1, 2, 3, 4),
        weight_profile_overrides={"equal": {name: 0.25 for name in CRITERION_NAMES}},
        trace_count=4,
    )

    assert factory_calls == 4
    for trace in report["policies"]["nonstop"]["traces"]:
        for profile, weights in report["weight_profiles"].items():
            assert trace["profile_scores"][profile] == pytest.approx(
                reweighted_score(trace["raw_scores"], weights)
            )


def test_shared_pool_cache_materializes_a_flight_pool_once_across_scenarios() -> None:
    request, data = _direct_data()
    calls = 0
    original = data.historical_outcomes

    def counted_transition(flight: FlightCandidate) -> tuple[SampledOutcome, ...]:
        nonlocal calls
        calls += 1
        return original(flight)

    data.transition_outcomes = counted_transition  # type: ignore[attr-defined]
    cached = ScenarioPoolCache(data, maxsize=4)
    report = run_paired_scenarios(
        request=request,
        data=cached,
        policy_factories={
            "first": lambda _seed: NonstopFirstPolicy(),
            "second": lambda _seed: NonstopFirstPolicy(),
        },
        scenario_keys=(10, 11, 12),
        trace_count=0,
    )

    assert calls == 1
    assert cached.cache_info() == {"hits": 2, "misses": 1, "maxsize": 4, "currsize": 1}
    assert report["policies"]["first"] == report["policies"]["second"]
    assert (
        report["paired_policy_differences"]["second_minus_first"]["metrics"]["on_time_arrival"][
            "mean_second_minus_first"
        ]
        == 0.0
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("scenario_keys", (), "nonempty"),
        ("scenario_keys", (1, 1), "unique"),
        ("scenario_keys", (True,), "non-boolean"),
        ("max_candidates", True, "max_candidates"),
        ("max_candidates", 0, "max_candidates"),
        ("trace_count", True, "trace_count"),
        ("trace_count", -1, "trace_count"),
    ],
)
def test_paired_run_rejects_invalid_configuration(field: str, value: object, match: str) -> None:
    request, data = _direct_data()
    arguments: dict[str, object] = {
        "request": request,
        "data": data,
        "policy_factories": {"nonstop": lambda _seed: NonstopFirstPolicy()},
        "scenario_keys": (1,),
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=match):
        run_paired_scenarios(**arguments)  # type: ignore[arg-type]
