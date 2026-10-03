"""Adaptive connections lose onward options when the inbound flight is late."""

import pytest

from flight_rl.env import FlightRouteEnv
from flight_rl.fixtures import DemoFlightData
from flight_rl.models import FlightCandidate, SampledOutcome, TripRequest
from flight_rl.planning import DeadlinePlannerPolicy


def connection_data(inbound_delay):
    flights = (
        FlightCandidate("inbound", "AAA", "BBB", "X", 60, 120, "1970-01-01", 0, "DJF"),
        FlightCandidate("tight", "BBB", "CCC", "X", 165, 220, "1970-01-01", 0, "DJF"),
        FlightCandidate("later", "BBB", "CCC", "X", 210, 280, "1970-01-01", 0, "DJF"),
    )
    return DemoFlightData(
        flights,
        {
            "inbound": (SampledOutcome("inbound-donor", arr_delay_min=inbound_delay),),
            "tight": (SampledOutcome("tight-donor"),),
            "later": (SampledOutcome("later-donor"),),
        },
    )


@pytest.mark.parametrize(
    ("inbound_delay", "next_flight", "on_time"), [(0, "tight", True), (30, "later", False)]
)
def test_inbound_delay_removes_connection_and_changes_deadline_success(
    inbound_delay, next_flight, on_time
):
    request = TripRequest("AAA", "CCC", 0, 250, 300)
    data = connection_data(inbound_delay)
    with FlightRouteEnv(request, data) as env:
        obs, _ = env.reset(seed=0)
        planner = DeadlinePlannerPolicy(
            request, data, max_branches=None, time_bin_min=1, outcome_bins=0
        )
        assert planner.action_values(obs)[0] == float(on_time)
        obs, _, terminated, _, _ = env.step(0)
        assert not terminated
        # Availability uses realized inbound arrival plus the 45-minute boarding buffer.
        assert env.available_flights[0].flight_id == next_flight
        _, _, terminated, _, info = env.step(0)
        assert terminated
        assert info["metrics"]["on_time_arrival"] is on_time
        assert info["metrics"]["validity"]
