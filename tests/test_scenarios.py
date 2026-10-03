from __future__ import annotations

import json

import numpy as np
import pytest

from flight_rl.models import FlightCandidate, OutcomeSummary, SampledOutcome
from flight_rl.scenarios import (
    SEVERITY_ORDER,
    ScenarioData,
    ScenarioPoolCache,
    _ordered_outcomes,
    _quantile_index,
    _select_quantile,
    _uint64_to_uniform,
)


def _flight(flight_id: str, origin: str = "AAA", dest: str = "BBB") -> FlightCandidate:
    return FlightCandidate(
        flight_id=flight_id,
        origin=origin,
        dest=dest,
        carrier="ZZ",
        scheduled_departure_utc=1_000,
        scheduled_arrival_utc=1_120,
        flight_date="2024-01-01",
        dep_hour_bucket=12,
        season="winter",
    )


def _ordinary(donor_id: str, delay: float) -> SampledOutcome:
    return SampledOutcome(
        donor_id=donor_id,
        dep_delay_min=delay / 2,
        arr_delay_min=delay,
        actual_elapsed_min=120 + delay / 2,
        fallback_level="test",
    )


def _ranked_pool(prefix: str) -> tuple[SampledOutcome, ...]:
    return (
        _ordinary(f"{prefix}-ordinary", 5.0),
        SampledOutcome(
            donor_id=f"{prefix}-missing",
            dep_delay_min=None,
            arr_delay_min=None,
            fallback_level="test",
        ),
        SampledOutcome(
            donor_id=f"{prefix}-diverted",
            diverted=True,
            dep_delay_min=4.0,
            div_reached_dest=False,
            fallback_level="test",
        ),
        SampledOutcome(
            donor_id=f"{prefix}-cancelled",
            cancelled=True,
            dep_delay_min=None,
            arr_delay_min=None,
            fallback_level="test",
        ),
    )


class _FitOnlyData:
    airports = ("AAA", "BBB", "CCC")
    carriers = ("ZZ",)

    def __init__(self, pools: dict[str, tuple[SampledOutcome, ...]]) -> None:
        self.pools = pools
        self.summary = OutcomeSummary(0.1, 0.2, 3.0, 10, "fit-only")

    def candidates(
        self, origin: str, earliest_utc: int, horizon_utc: int, limit: int = 64
    ) -> tuple[FlightCandidate, ...]:
        del earliest_utc, horizon_utc
        return tuple(
            flight for flight in (_flight("A", origin, "BBB"), _flight("B", origin, "CCC"))
        )[:limit]

    def sample_outcome(self, flight: FlightCandidate, rng: np.random.Generator) -> SampledOutcome:
        del flight, rng
        raise AssertionError("ScenarioData must select its own deterministic donor")

    def outcome_summary(self, flight: FlightCandidate) -> OutcomeSummary:
        del flight
        return self.summary

    def historical_outcomes(self, flight: FlightCandidate) -> tuple[SampledOutcome, ...]:
        return self.pools[flight.flight_id]


class _HeldoutData(_FitOnlyData):
    def __init__(
        self,
        fit_pools: dict[str, tuple[SampledOutcome, ...]],
        transition_pools: dict[str, tuple[SampledOutcome, ...]],
    ) -> None:
        super().__init__(fit_pools)
        self.transition_pools = transition_pools

    def transition_outcomes(self, flight: FlightCandidate) -> tuple[SampledOutcome, ...]:
        return self.transition_pools[flight.flight_id]


def test_draws_are_paired_and_independent_of_call_order_or_rng_state() -> None:
    flights = (_flight("A"), _flight("B"))
    pools = {flight.flight_id: _ranked_pool(flight.flight_id) for flight in flights}
    forward = ScenarioData(_FitOnlyData(pools), scenario_seed=314, dependence=0.0)
    reverse = ScenarioData(_FitOnlyData(pools), scenario_seed=314, dependence=0.0)
    first_rng = np.random.default_rng(1)
    second_rng = np.random.default_rng(999)

    forward_draws = {
        flight.flight_id: forward.sample_outcome(flight, first_rng).donor_id for flight in flights
    }
    second_rng.random(100)
    reverse_draws = {
        flight.flight_id: reverse.sample_outcome(flight, second_rng).donor_id
        for flight in reversed(flights)
    }

    assert forward_draws == reverse_draws
    assert (
        forward.sample_outcome(flights[0], np.random.default_rng(2)).donor_id == forward_draws["A"]
    )


def test_transition_pool_is_used_only_for_simulator_sampling() -> None:
    flight = _flight("A")
    fit = (_ordinary("fit-donor", 0.0),)
    heldout = (_ordinary("heldout-donor", 30.0),)
    data = _HeldoutData({"A": fit}, {"A": heldout})
    scenario = ScenarioData(data, scenario_seed=10)

    assert scenario.historical_outcomes(flight) == fit
    assert scenario.outcome_summary(flight) is data.summary
    assert scenario.transition_outcomes(flight) == heldout
    assert scenario.sample_outcome(flight, np.random.default_rng()).donor_id == "heldout-donor"
    assert scenario.diagnostics()["transition_pool_source"] == "transition_outcomes"


def test_fit_history_is_the_transition_fallback_when_hook_is_absent() -> None:
    flight = _flight("A")
    fit = (_ordinary("fit-donor", 0.0),)
    scenario = ScenarioData(_FitOnlyData({"A": fit}), scenario_seed=10)

    assert scenario.transition_outcomes(flight) == fit
    assert scenario.sample_outcome(flight, np.random.default_rng()).donor_id == "fit-donor"
    assert scenario.diagnostics()["transition_pool_source"] == "historical_outcomes"


def test_severity_order_is_stable_and_ignores_redundant_elapsed_disagreement() -> None:
    ordinary = SampledOutcome(
        donor_id="ordinary",
        dep_delay_min=10.0,
        arr_delay_min=20.0,
        actual_elapsed_min=-999.0,
    )
    missing = SampledOutcome(donor_id="missing_timing", dep_delay_min=None, arr_delay_min=0.0)
    diverted = SampledOutcome(
        donor_id="diverted",
        diverted=True,
        dep_delay_min=None,
        div_reached_dest=False,
    )
    cancelled = SampledOutcome(
        donor_id="cancelled",
        cancelled=True,
        diverted=True,
        dep_delay_min=0.0,
        arr_delay_min=0.0,
    )

    ordered = _ordered_outcomes((ordinary, missing, cancelled, diverted))

    assert tuple(outcome.donor_id for outcome in ordered) == SEVERITY_ORDER


def test_empirical_quantile_bins_preserve_each_donor_marginal_exactly() -> None:
    for pool_size in range(1, 18):
        pool = tuple(_ordinary(f"donor-{index:02d}", float(index)) for index in range(pool_size))
        selected = [
            _select_quantile(pool, (index + 0.5) / pool_size).donor_id for index in range(pool_size)
        ]
        assert sorted(selected) == sorted(outcome.donor_id for outcome in pool)
        assert [
            _quantile_index((index + 0.5) / pool_size, pool_size) for index in range(pool_size)
        ] == list(range(pool_size))


def test_uint64_uniform_conversion_is_exactly_bounded_below_one() -> None:
    assert _uint64_to_uniform(0) == 0.0
    assert _uint64_to_uniform((1 << 64) - 1) == ((1 << 53) - 1) / (1 << 53)
    assert 0.0 <= _uint64_to_uniform((1 << 64) - 1) < 1.0
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        _uint64_to_uniform(1 << 64)


def test_common_dependence_uses_the_same_severity_rank_across_flights() -> None:
    flights = (_flight("A"), _flight("B"))
    pools = {"A": _ranked_pool("A"), "B": _ranked_pool("B")}
    scenario = ScenarioData(_FitOnlyData(pools), scenario_seed=72, dependence=1.0)

    selected_indices = []
    for flight in flights:
        ordered = _ordered_outcomes(pools[flight.flight_id])
        selected = scenario.sample_outcome(flight, np.random.default_rng())
        selected_indices.append(ordered.index(selected))

    assert selected_indices[0] == selected_indices[1]
    assert scenario.diagnostics()["coupling_mode"] == "common_severity_rank"


def test_dependence_gate_is_nested_and_endpoints_are_exact() -> None:
    data = _FitOnlyData({"A": _ranked_pool("A")})
    independent = ScenarioData(data, scenario_seed=91, dependence=0.0).diagnostics()
    midpoint = ScenarioData(data, scenario_seed=91, dependence=0.5).diagnostics()
    common = ScenarioData(data, scenario_seed=91, dependence=1.0).diagnostics()

    assert independent["coupling_mode"] == "flight_keyed_independent"
    assert common["coupling_mode"] == "common_severity_rank"
    expected_midpoint = (
        "common_severity_rank" if midpoint["coupling_gate"] < 0.5 else "flight_keyed_independent"
    )
    assert midpoint["coupling_mode"] == expected_midpoint
    assert independent["coupling_gate"] == midpoint["coupling_gate"] == common["coupling_gate"]
    assert (
        independent["common_quantile"] == midpoint["common_quantile"] == common["common_quantile"]
    )


def test_source_pool_order_cannot_change_selected_donor() -> None:
    flight = _flight("A")
    pool = _ranked_pool("A")
    first = ScenarioData(_FitOnlyData({"A": pool}), scenario_seed=18, dependence=1.0)
    second = ScenarioData(
        _FitOnlyData({"A": tuple(reversed(pool))}), scenario_seed=18, dependence=1.0
    )
    assert first.sample_outcome(flight, np.random.default_rng()) == second.sample_outcome(
        flight, np.random.default_rng()
    )


def test_explicit_pool_cache_is_bounded_source_owned_and_clearable() -> None:
    flights = tuple(_flight(name) for name in ("A", "B", "C"))
    pools = {flight.flight_id: (_ordinary(f"{flight.flight_id}-old", 0.0),) for flight in flights}
    data = _FitOnlyData(pools)
    cache = ScenarioPoolCache(data, maxsize=2)

    for index, flight in enumerate(flights):
        ScenarioData(cache, scenario_seed=index).sample_outcome(flight, np.random.default_rng())
    assert cache.cache_info() == {"hits": 0, "misses": 3, "maxsize": 2, "currsize": 2}

    # B is resident, so a new scenario reuses the exact immutable tuple.
    ScenarioData(cache, scenario_seed=9).sample_outcome(flights[1], np.random.default_rng())
    assert cache.cache_info()["hits"] == 1

    data.pools["B"] = (_ordinary("B-new", 0.0),)
    still_snapshot = ScenarioData(cache, scenario_seed=10).sample_outcome(
        flights[1], np.random.default_rng()
    )
    assert still_snapshot.donor_id == "B-old"
    cache.clear()
    refreshed = ScenarioData(cache, scenario_seed=10).sample_outcome(
        flights[1], np.random.default_rng()
    )
    assert refreshed.donor_id == "B-new"
    assert cache.cache_info() == {"hits": 0, "misses": 1, "maxsize": 2, "currsize": 1}


@pytest.mark.parametrize("maxsize", [True, 0, -1, 1.5])
def test_pool_cache_rejects_invalid_capacity(maxsize: object) -> None:
    with pytest.raises(ValueError, match="maxsize"):
        ScenarioPoolCache(_FitOnlyData({}), maxsize=maxsize)  # type: ignore[arg-type]


def test_delegation_and_diagnostics_are_json_safe() -> None:
    data = _FitOnlyData({"A": _ranked_pool("A"), "B": _ranked_pool("B")})
    scenario = ScenarioData(data, scenario_seed=np.int64(5), dependence=np.float64(0.25))

    assert scenario.candidates("AAA", 0, 2_000, limit=1) == (_flight("A", "AAA", "BBB"),)
    assert scenario.scenario_seed == 5
    assert scenario.dependence == 0.25
    assert json.loads(json.dumps(scenario.diagnostics()))["severity_order"] == list(SEVERITY_ORDER)
    assert "not estimated correlation" in str(scenario.diagnostics()["dependence_interpretation"])


@pytest.mark.parametrize("scenario_seed", [True, np.bool_(False), 1.5, "1"])
def test_scenario_seed_requires_a_non_boolean_integer(scenario_seed: object) -> None:
    with pytest.raises(TypeError, match="scenario_seed"):
        ScenarioData(_FitOnlyData({}), scenario_seed=scenario_seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("dependence", [True, np.nan, np.inf, -0.01, 1.01, "0.5"])
def test_dependence_rejects_invalid_values(dependence: object) -> None:
    with pytest.raises(ValueError, match="dependence"):
        ScenarioData(_FitOnlyData({}), scenario_seed=1, dependence=dependence)  # type: ignore[arg-type]


def test_sampling_rejects_invalid_rng_and_transition_pools() -> None:
    flight = _flight("A")
    valid = ScenarioData(_FitOnlyData({"A": (_ordinary("one", 0.0),)}), scenario_seed=1)
    with pytest.raises(TypeError, match="rng"):
        valid.sample_outcome(flight, object())  # type: ignore[arg-type]

    empty = ScenarioData(_FitOnlyData({"A": ()}), scenario_seed=1)
    with pytest.raises(ValueError, match="nonempty"):
        empty.sample_outcome(flight, np.random.default_rng())

    duplicate = _ordinary("duplicate", 0.0)
    duplicate_pool = ScenarioData(_FitOnlyData({"A": (duplicate, duplicate)}), scenario_seed=1)
    with pytest.raises(ValueError, match="unique"):
        duplicate_pool.sample_outcome(flight, np.random.default_rng())

    wrong_type = ScenarioData(_FitOnlyData({"A": (object(),)}), scenario_seed=1)  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="SampledOutcome"):
        wrong_type.sample_outcome(flight, np.random.default_rng())
