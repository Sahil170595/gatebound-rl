from __future__ import annotations

import numpy as np
import pytest

from flight_rl.baselines import NonstopFirstPolicy
from flight_rl.env import FlightRouteEnv
from flight_rl.fixtures import DemoFlightData
from flight_rl.models import FlightCandidate, SampledOutcome, TripRequest
from flight_rl.planning import DeadlinePlannerPolicy
from flight_rl.scenarios import ScenarioData
from flight_rl.split_data import SplitFlightData


def _sources():
    request = TripRequest("AAA", "CCC", 0, 100, 200, min_connection_min=5)
    flights = (
        FlightCandidate("AB", "AAA", "BBB", "X", 10, 30, "1970-01-01", 0, "DJF"),
        FlightCandidate("AC", "AAA", "CCC", "X", 15, 60, "1970-01-01", 0, "DJF"),
        FlightCandidate("BC", "BBB", "CCC", "X", 40, 70, "1970-01-01", 0, "DJF"),
    )
    fixed_outcomes = DemoFlightData(
        flights,
        {flight.flight_id: (SampledOutcome(f"fixed-{flight.flight_id}"),) for flight in flights},
    )
    connection_favored = DemoFlightData(
        flights,
        {
            "AB": (
                SampledOutcome("fit-a-ab-ok"),
                SampledOutcome("fit-a-ab-late", dep_delay_min=20, arr_delay_min=20),
            ),
            "AC": (
                SampledOutcome("fit-a-ac-ok"),
                SampledOutcome("fit-a-ac-cancel", cancelled=True),
                SampledOutcome("fit-a-ac-late", dep_delay_min=100, arr_delay_min=100),
            ),
            "BC": (SampledOutcome("fit-a-bc-ok"),),
        },
    )
    direct_favored = DemoFlightData(
        flights,
        {
            "AB": (SampledOutcome("fit-b-ab-cancel", cancelled=True),),
            "AC": (SampledOutcome("fit-b-ac-ok"),),
            "BC": (SampledOutcome("fit-b-bc-ok"),),
        },
    )
    routes = frozenset((flight.origin, flight.dest) for flight in flights)
    return request, fixed_outcomes, connection_favored, direct_favored, routes


def _run_nonstop(request: TripRequest, data: SplitFlightData):
    env = FlightRouteEnv(request, ScenarioData(data, scenario_seed=73))
    policy = NonstopFirstPolicy()
    observation, _ = env.reset(seed=11)
    while True:
        observation, _, terminated, truncated, _ = env.step(policy.act(observation))
        if terminated or truncated:
            return env.record


def test_fit_inputs_change_planner_values_without_changing_evaluation_universe() -> None:
    request, fixed, fit_a, fit_b, routes = _sources()
    first = SplitFlightData(fixed, fit_a, fixed, candidate_routes=routes)
    second = SplitFlightData(fixed, fit_b, fixed, candidate_routes=routes)
    first_env = FlightRouteEnv(request, first)
    second_env = FlightRouteEnv(request, second)
    first_observation, _ = first_env.reset(seed=1)
    second_observation, _ = second_env.reset(seed=1)

    assert tuple(flight.flight_id for flight in first_env.available_flights) == ("AB", "AC")
    assert first_env.available_flights == second_env.available_flights
    np.testing.assert_array_equal(
        first_observation["candidates"][:, :5], second_observation["candidates"][:, :5]
    )
    np.testing.assert_array_equal(
        first_observation["action_mask"], second_observation["action_mask"]
    )
    assert not np.array_equal(
        first_observation["candidates"][:, 5:9], second_observation["candidates"][:, 5:9]
    )

    first_planner = DeadlinePlannerPolicy(
        request, first, max_branches=None, time_bin_min=1, outcome_bins=0
    )
    second_planner = DeadlinePlannerPolicy(
        request, second, max_branches=None, time_bin_min=1, outcome_bins=0
    )
    assert first_planner.action_values(first_observation) == pytest.approx((0.5, 1 / 3))
    assert second_planner.action_values(second_observation) == pytest.approx((0.0, 1.0))
    assert first_planner.act(first_observation) == 0
    assert second_planner.act(second_observation) == 1

    for flight in fixed.flights:
        first_draw = ScenarioData(first, scenario_seed=73).sample_outcome(
            flight, np.random.default_rng(1)
        )
        second_draw = ScenarioData(second, scenario_seed=73).sample_outcome(
            flight, np.random.default_rng(999)
        )
        assert first_draw == second_draw
        assert first_draw.donor_id == f"fixed-{flight.flight_id}"

    assert _run_nonstop(request, first) == _run_nonstop(request, second)


def test_explicit_candidate_routes_reject_an_unsupported_fit_route() -> None:
    request, fixed, fit_a, _, routes = _sources()
    original_summary = fit_a.outcome_summary
    fit_a.outcome_summary = lambda flight: (
        type(original_summary(flight))(0.0, 0.0, 0.0, 0, "no_route")
        if flight.flight_id == "AC"
        else original_summary(flight)
    )
    data = SplitFlightData(fixed, fit_a, fixed, candidate_routes=routes)

    with pytest.raises(ValueError, match="AAA-CCC"):
        data.candidates("AAA", request.ready_utc, request.horizon_utc)


def test_explicit_candidate_routes_filter_and_refill_the_fixed_schedule() -> None:
    request, fixed, fit_a, _, _ = _sources()
    data = SplitFlightData(
        fixed,
        fit_a,
        fixed,
        candidate_routes={("AAA", "CCC"), ("BBB", "CCC")},
    )

    assert tuple(
        flight.flight_id
        for flight in data.candidates("AAA", request.ready_utc, request.horizon_utc, limit=2)
    ) == ("AC",)
    assert data.metadata["unique_queried_schedules_outside_candidate_routes"] == 1


@pytest.mark.parametrize(
    "routes",
    ["AAA-BBB", (("AAA",),), (("AAA", "AAA"),), (("AAA", ""),)],
)
def test_candidate_routes_reject_invalid_contracts(routes: object) -> None:
    _, fixed, fit_a, _, _ = _sources()
    with pytest.raises((TypeError, ValueError), match="candidate_routes"):
        SplitFlightData(fixed, fit_a, fixed, candidate_routes=routes)  # type: ignore[arg-type]
