"""Adversarial contract checks for the deadline planner."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from flight_rl.env import FlightRouteEnv
from flight_rl.fixtures import DemoFlightData
from flight_rl.models import CANDIDATE_FEATURES, FlightCandidate, SampledOutcome, TripRequest
from flight_rl.planning import DeadlinePlannerPolicy


def direct_flight(
    flight_id: str = "AC",
    *,
    departure: int = 10,
    arrival: int = 50,
) -> FlightCandidate:
    return FlightCandidate(
        flight_id,
        "AAA",
        "CCC",
        "X",
        departure,
        arrival,
        "1970-01-01",
        0,
        "DJF",
    )


def exact_planner(
    outcomes: tuple[SampledOutcome, ...],
    *,
    selected: FlightCandidate | None = None,
    request: TripRequest | None = None,
) -> tuple[DeadlinePlannerPolicy, DemoFlightData, dict]:
    selected = selected or direct_flight()
    request = request or TripRequest(
        "AAA",
        "CCC",
        0,
        100,
        200,
        max_attempts=3,
        min_connection_min=0,
        delay_budget_min=200,
    )
    data = DemoFlightData((selected,), {selected.flight_id: outcomes})
    candidates = np.zeros((1, len(CANDIDATE_FEATURES)), dtype=np.float32)
    candidates[0, CANDIDATE_FEATURES.index("destination_index")] = data.airports.index("CCC")
    candidates[0, CANDIDATE_FEATURES.index("departure_in_min")] = (
        selected.scheduled_departure_utc - request.ready_utc
    )
    candidates[0, CANDIDATE_FEATURES.index("arrival_in_min")] = (
        selected.scheduled_arrival_utc - request.ready_utc
    )
    candidates[0, CANDIDATE_FEATURES.index("scheduled_elapsed_min")] = (
        selected.scheduled_elapsed_min
    )
    observation = {
        "current_airport": data.airports.index("AAA"),
        "time": np.asarray(
            [0, request.deadline_utc, request.horizon_utc, request.max_attempts],
            dtype=np.float32,
        ),
        "action_mask": np.asarray([1], dtype=np.int8),
        "candidates": candidates,
    }
    planner = DeadlinePlannerPolicy(
        request,
        data,
        max_branches=None,
        time_bin_min=1,
        outcome_bins=0,
    )
    return planner, data, observation


def test_zero_realized_duration_is_not_a_successful_arrival() -> None:
    # Scheduled 10 -> 50; the donor moves departure to 50 and leaves arrival
    # at 50. The environment rejects this nonpositive realized duration too.
    planner, _data, observation = exact_planner(
        (SampledOutcome("zero-duration", dep_delay_min=40, arr_delay_min=0),)
    )
    assert planner.action_values(observation) == (0.0,)


@pytest.mark.parametrize("bad", [True, "0", float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["dep_delay_min", "arr_delay_min"])
def test_bool_string_and_nonfinite_ordinary_timing_fail_closed(field: str, bad: object) -> None:
    outcome = replace(SampledOutcome("bad-ordinary"), **{field: bad})
    planner, _data, observation = exact_planner((outcome,))
    assert planner.action_values(observation) == (0.0,)


@pytest.mark.parametrize("bad", [True, "30", float("nan"), float("inf")])
def test_bool_string_and_nonfinite_diversion_fallback_fail_closed(bad: object) -> None:
    outcome = SampledOutcome(
        "bad-diversion",
        diverted=True,
        dep_delay_min=0,
        div_reached_dest=True,
        div_arr_delay_min=None,
        div_actual_elapsed_min=bad,  # type: ignore[arg-type]
    )
    planner, _data, observation = exact_planner((outcome,))
    assert planner.action_values(observation) == (0.0,)


def test_realized_departure_before_current_clock_is_a_missed_flight() -> None:
    selected = direct_flight(departure=20, arrival=60)
    planner, _data, observation = exact_planner(
        (SampledOutcome("departed-already", dep_delay_min=-10, arr_delay_min=0),),
        selected=selected,
    )
    observation["time"][0] = 15
    observation["candidates"][0, CANDIDATE_FEATURES.index("departure_in_min")] -= 15
    observation["candidates"][0, CANDIDATE_FEATURES.index("arrival_in_min")] -= 15
    assert planner.action_values(observation) == (0.0,)


def test_reached_destination_diversion_uses_elapsed_fallback() -> None:
    selected = direct_flight(departure=10, arrival=50)
    outcome = SampledOutcome(
        "diversion-fallback",
        diverted=True,
        dep_delay_min=5,
        div_reached_dest=True,
        div_arr_delay_min=None,
        div_actual_elapsed_min=35,
    )
    planner, _data, observation = exact_planner((outcome,), selected=selected)
    # Actual arrival is 10 + 5 + 35 = 50.
    assert planner.action_values(observation) == (1.0,)


def test_arrival_at_deadline_counts_and_one_minute_late_does_not() -> None:
    selected = direct_flight(departure=10, arrival=100)
    planner, _data, observation = exact_planner(
        (
            SampledOutcome("at-deadline", dep_delay_min=0, arr_delay_min=0),
            SampledOutcome("after-deadline", dep_delay_min=0, arr_delay_min=1),
        ),
        selected=selected,
    )
    assert planner.action_values(observation) == pytest.approx((0.5,))


def test_candidate_cap_mismatch_fails_before_returning_wrong_index() -> None:
    first = direct_flight("first", departure=10, arrival=60)
    second = direct_flight("second", departure=20, arrival=50)
    pools = {
        first.flight_id: (SampledOutcome("first-donor"),),
        second.flight_id: (SampledOutcome("second-donor"),),
    }
    data = DemoFlightData((first, second), pools)
    request = TripRequest("AAA", "CCC", 0, 100, 200, min_connection_min=0)
    planner = DeadlinePlannerPolicy(request, data, max_candidates=1)
    observation = {
        "current_airport": data.airports.index("AAA"),
        "time": np.asarray([0, 100, 200, 3], dtype=np.float32),
        "action_mask": np.asarray([1, 1], dtype=np.int8),
        "candidates": np.zeros((2, len(CANDIDATE_FEATURES)), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="candidates do not match"):
        planner.act(observation)


@pytest.mark.parametrize(
    ("parameter", "bad"),
    [
        ("max_candidates", True),
        ("max_candidates", 1.5),
        ("max_candidates", float("nan")),
        ("time_bin_min", True),
        ("time_bin_min", 1.5),
        ("time_bin_min", float("inf")),
        ("outcome_bins", True),
        ("outcome_bins", 1.5),
        ("outcome_bins", float("nan")),
        ("max_branches", True),
        ("max_branches", 1.5),
        ("max_branches", float("inf")),
    ],
)
def test_planner_capacities_require_non_bool_integers(parameter: str, bad: object) -> None:
    selected = direct_flight()
    data = DemoFlightData((selected,), {selected.flight_id: (SampledOutcome("donor"),)})
    request = TripRequest("AAA", "CCC", 0, 100, 200, min_connection_min=0)
    with pytest.raises(ValueError, match=parameter):
        DeadlinePlannerPolicy(request, data, **{parameter: bad})


class UnorderedFlightData(DemoFlightData):
    """Return matching-origin rows; the environment enforces its own ordering/window/cap."""

    def candidates(self, origin, earliest_utc, horizon_utc, limit=64):
        return tuple(f for f in reversed(self.flights) if f.origin == origin)


class ReversedFlightData(DemoFlightData):
    """Honor the requested window and cap, returning only the order differently."""

    def candidates(self, origin, earliest_utc, horizon_utc, limit=64):
        return tuple(reversed(super().candidates(origin, earliest_utc, horizon_utc, limit)))


@pytest.mark.parametrize("data_type", [UnorderedFlightData, ReversedFlightData])
@pytest.mark.parametrize("cap", [1, 2])
def test_planner_matches_environment_for_unordered_candidates(cap: int, data_type) -> None:
    early = direct_flight("early", departure=10, arrival=50)
    late = direct_flight("late", departure=20, arrival=90)
    past = direct_flight("past", departure=-5, arrival=5)
    data = data_type(
        (late, past, early),
        {
            "early": (SampledOutcome("ok"),),
            "late": (SampledOutcome("cancelled", cancelled=True),),
            "past": (SampledOutcome("past-ok"),),
        },
    )
    request = TripRequest("AAA", "CCC", 0, 100, 200, min_connection_min=5)
    env = FlightRouteEnv(request, data, max_candidates=cap)
    observation, _ = env.reset(seed=1)
    planner = DeadlinePlannerPolicy(request, data, max_candidates=cap)
    assert planner.action_values(observation) == (1.0, 0.0)[:cap]
    assert planner.act(observation) == 0
    _, _, terminated, _, info = env.step(planner.act(observation))
    assert terminated and info["metrics"]["on_time_arrival"]
    env.close()


def test_planner_rejects_connection_before_boarding_buffer() -> None:
    first = replace(direct_flight("AB", departure=10, arrival=40), dest="BBB")
    too_tight = replace(direct_flight("tight", departure=42, arrival=70), origin="BBB")
    later = replace(direct_flight("later", departure=60, arrival=80), origin="BBB")
    data = UnorderedFlightData(
        (first, too_tight, later),
        {
            "AB": (SampledOutcome("first-ok"),),
            "tight": (SampledOutcome("tight-ok"),),
            "later": (SampledOutcome("later-cancel", cancelled=True),),
        },
    )
    request = TripRequest("AAA", "CCC", 0, 100, 200, min_connection_min=5)
    env = FlightRouteEnv(request, data)
    observation, _ = env.reset(seed=1)
    planner = DeadlinePlannerPolicy(
        request, data, max_branches=None, time_bin_min=1, outcome_bins=0
    )
    assert planner.action_values(observation) == (0.0,)
    observation, _, terminated, _, _ = env.step(0)
    assert not terminated
    assert [f.flight_id for f in env.available_flights] == ["later"]
    assert planner.action_values(observation) == (0.0,)
    env.close()
