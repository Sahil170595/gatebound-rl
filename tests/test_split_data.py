from dataclasses import replace

import numpy as np
import pytest

from flight_rl.env import FlightRouteEnv
from flight_rl.fixtures import DemoFlightData, make_demo_scenario
from flight_rl.models import OutcomeSummary
from flight_rl.scenarios import ScenarioData
from flight_rl.split_data import SplitFlightData


def split_fixture():
    request, schedule = make_demo_scenario()
    fit = DemoFlightData(
        schedule.flights,
        {
            f.flight_id: (replace(schedule.historical_outcomes(f)[0], donor_id="fit-2024"),)
            for f in schedule.flights
        },
    )
    heldout = DemoFlightData(
        schedule.flights,
        {
            f.flight_id: (
                replace(
                    schedule.historical_outcomes(f)[0],
                    donor_id="heldout-2025",
                    cancelled=True,
                    dep_delay_min=None,
                    arr_delay_min=None,
                ),
            )
            for f in schedule.flights
        },
    )
    return request, schedule, fit, heldout


def test_priors_and_planner_history_use_fit_but_transitions_use_heldout():
    request, schedule, fit, heldout = split_fixture()
    split = SplitFlightData(schedule, fit, heldout)
    env = FlightRouteEnv(request, ScenarioData(split, 19))
    obs, _ = env.reset(seed=19)
    legal = np.flatnonzero(obs["action_mask"])
    assert len(legal) > 0
    assert np.all(obs["candidates"][legal, 5] == 0)
    flight = env.available_flights[0]
    assert {x.donor_id for x in split.historical_outcomes(flight)} == {"fit-2024"}
    _, reward, terminated, _, _ = env.step(0)
    assert terminated and reward == 0
    assert env.record.termination_reason == "cancelled"
    assert env.record.legs[0].outcome.donor_id == "heldout-2025"


def test_outcome_changes_do_not_change_observation_or_vocabulary():
    request, schedule, fit, heldout = split_fixture()
    heldout.airports = (*heldout.airports, "ZZZ")
    heldout.carriers = (*heldout.carriers, "SECRET")
    a = FlightRouteEnv(request, SplitFlightData(schedule, fit, heldout))
    b = FlightRouteEnv(request, SplitFlightData(schedule, fit, fit))
    left, _ = a.reset(seed=4)
    right, _ = b.reset(seed=4)
    assert a.data.airports == b.data.airports
    assert a.data.carriers == b.data.carriers
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def test_fit_coverage_filter_refills_candidate_cap():
    request, schedule, fit, heldout = split_fixture()
    unsupported = schedule.flights[0].flight_id
    original_summary = fit.outcome_summary
    fit.outcome_summary = lambda flight: (
        OutcomeSummary(0, 0, 0, 0, "no_route")
        if flight.flight_id == unsupported
        else original_summary(flight)
    )
    split = SplitFlightData(schedule, fit, heldout)
    candidates = split.candidates(request.origin, request.ready_utc, request.horizon_utc, 2)
    assert len(candidates) == 2
    assert unsupported not in {f.flight_id for f in candidates}
    assert split.metadata["unique_queried_schedules_without_fit_coverage"] == 1
    assert len(split.candidates(request.origin, request.ready_utc, request.horizon_utc, 64)) == 2


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5])
def test_rejects_invalid_candidate_limit(limit):
    request, schedule, fit, heldout = split_fixture()
    with pytest.raises(ValueError):
        SplitFlightData(schedule, fit, heldout).candidates(
            request.origin, request.ready_utc, request.horizon_utc, limit
        )
