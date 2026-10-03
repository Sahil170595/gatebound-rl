"""Hand-enumerated policy values, including missed connections and failed donors."""

import numpy as np
import pytest

from flight_rl.fixtures import DemoFlightData
from flight_rl.models import FlightCandidate, SampledOutcome, TripRequest
from flight_rl.planning import DeadlinePlannerPolicy


def scenario():
    request = TripRequest("AAA", "CCC", 0, 100, 200, min_connection_min=5, delay_budget_min=200)
    flights = (
        FlightCandidate("AB", "AAA", "BBB", "X", 10, 30, "1970-01-01", 0, "DJF"),
        FlightCandidate("AC", "AAA", "CCC", "X", 15, 60, "1970-01-01", 0, "DJF"),
        FlightCandidate("BC", "BBB", "CCC", "X", 40, 70, "1970-01-01", 0, "DJF"),
    )
    pools = {
        "AB": (
            SampledOutcome("AB-ok"),
            SampledOutcome("AB-late", dep_delay_min=20, arr_delay_min=20),
        ),
        "AC": (
            SampledOutcome("AC-ok"),
            SampledOutcome("AC-cancel", cancelled=True),
            SampledOutcome("AC-too-late", dep_delay_min=100, arr_delay_min=100),
        ),
        "BC": (SampledOutcome("BC-ok"),),
    }
    data = DemoFlightData(flights, pools)
    obs = {
        "current_airport": data.airports.index("AAA"),
        "time": np.array([0, 100, 200, 3]),
        "action_mask": np.array([1, 1]),
        "candidates": np.array([[1, 0, 10, 30, 20], [2, 0, 15, 60, 45]]),
    }
    return request, data, obs


def test_exact_small_model_prefers_reliable_connection():
    request, data, obs = scenario()
    planner = DeadlinePlannerPolicy(
        request, data, max_branches=None, time_bin_min=1, outcome_bins=0
    )
    # Half of AB's donors arrive in time for BC; only one third of AC succeeds by deadline.
    assert planner.action_values(obs) == pytest.approx((0.5, 1 / 3))
    assert planner.act(obs) == 0
    assert planner.action_values(obs) == pytest.approx((0.5, 1 / 3))


def test_attempt_budget_blocks_connection_and_keeps_failure_mass():
    request, data, obs = scenario()
    obs["time"][3] = 1
    planner = DeadlinePlannerPolicy(
        request, data, max_branches=None, time_bin_min=1, outcome_bins=0
    )
    assert planner.action_values(obs) == pytest.approx((0, 1 / 3))
    assert planner.act(obs) == 1


def test_empty_mask_has_contract_sentinel():
    request, data, obs = scenario()
    obs["action_mask"][:] = 0
    assert DeadlinePlannerPolicy(request, data).act(obs) == 0
