"""Committed itinerary invariants and missed-connection reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from unittest.mock import ANY

import numpy as np
import pytest

from flight_rl.env import FlightRouteEnv
from flight_rl.models import (
    FlightCandidate,
    OutcomeSummary,
    SampledOutcome,
    TripRequest,
)
from flight_rl.prebooked import (
    PrebookedRouteEnv,
    plan_scheduled_itinerary,
    verify_prebooked,
)


@dataclass
class RecordingData:
    flights: tuple[FlightCandidate, ...]
    outcomes: dict[str, SampledOutcome]
    sampled: list[str] = field(default_factory=list)
    summary_reads: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.flights = tuple(
            sorted(
                self.flights,
                key=lambda flight: (
                    flight.scheduled_departure_utc,
                    flight.scheduled_arrival_utc,
                    flight.carrier,
                    flight.flight_id,
                ),
            )
        )
        self.airports = tuple(
            sorted({airport for flight in self.flights for airport in (flight.origin, flight.dest)})
        )
        self.carriers = tuple(sorted({flight.carrier for flight in self.flights}))

    def candidates(
        self, origin: str, earliest_utc: int, horizon_utc: int, limit: int = 64
    ) -> tuple[FlightCandidate, ...]:
        return tuple(
            flight
            for flight in self.flights
            if flight.origin == origin
            and earliest_utc <= flight.scheduled_departure_utc <= horizon_utc
        )[:limit]

    def sample_outcome(self, flight: FlightCandidate, rng: np.random.Generator) -> SampledOutcome:
        del rng
        self.sampled.append(flight.flight_id)
        return self.outcomes[flight.flight_id]

    def outcome_summary(self, flight: FlightCandidate) -> OutcomeSummary:
        self.summary_reads.append(flight.flight_id)
        return OutcomeSummary(0.0, 0.0, 0.0, 1, "fixture")

    def historical_outcomes(self, flight: FlightCandidate) -> tuple[SampledOutcome, ...]:
        return (self.outcomes[flight.flight_id],)

    def source_flight(self, flight_id: str) -> FlightCandidate | None:
        return next((flight for flight in self.flights if flight.flight_id == flight_id), None)

    def source_transition_outcome(self, flight_id: str, donor_id: str) -> SampledOutcome | None:
        outcome = self.outcomes.get(flight_id)
        return outcome if outcome is not None and outcome.donor_id == donor_id else None


def _flight(
    flight_id: str, origin: str, destination: str, departure: int, arrival: int
) -> FlightCandidate:
    return FlightCandidate(
        flight_id,
        origin,
        destination,
        "X",
        departure,
        arrival,
        "1970-01-01",
        0,
        "DJF",
    )


def _connection_case(
    inbound_delay: float,
) -> tuple[
    TripRequest,
    RecordingData,
    tuple[FlightCandidate, FlightCandidate],
]:
    inbound = _flight("inbound", "AAA", "BBB", 60, 120)
    early = _flight("early", "BBB", "CCC", 165, 220)
    later = _flight("later", "BBB", "CCC", 210, 280)
    request = TripRequest(
        "AAA",
        "CCC",
        ready_utc=0,
        deadline_utc=250,
        horizon_utc=320,
        max_attempts=2,
        min_connection_min=45,
    )
    data = RecordingData(
        (inbound, early, later),
        {
            "inbound": SampledOutcome("inbound-donor", arr_delay_min=inbound_delay),
            "early": SampledOutcome("early-donor"),
            "later": SampledOutcome("later-donor"),
        },
    )
    return request, data, (inbound, early)


def _run(env: PrebookedRouteEnv, seed: int = 7) -> None:
    env.reset(seed=seed)
    for _ in range(env.request.max_attempts + 1):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            return
    raise AssertionError("environment did not terminate")


def test_schedule_only_plan_is_committed_before_any_outcome_and_executes() -> None:
    request, data, expected = _connection_case(0.0)
    itinerary = plan_scheduled_itinerary(request, data, route=("AAA", "BBB", "CCC"))
    assert itinerary == expected
    assert data.sampled == []
    assert data.summary_reads == []

    with PrebookedRouteEnv(request, data, itinerary) as env:
        observation, _ = env.reset(seed=2)
        assert data.sampled == []
        assert env.available_flights == (expected[0],)
        assert observation["action_mask"].tolist() == [1]
        _, _, terminated, _, _ = env.step(0)
        assert not terminated
        assert env.available_flights == (expected[1],)
        _, _, terminated, _, _ = env.step(0)
        assert terminated
        record = env.record

    verification = verify_prebooked(record, data)
    assert record.termination_reason == "arrived"
    assert tuple(leg.flight for leg in record.episode.legs) == itinerary
    assert verification.validity
    assert verification.source_authenticated
    assert verification.reason == "verified"
    assert verification.score.aggregate_score > 0.9


def test_plan_waits_for_earliest_destination_instead_of_first_discovery() -> None:
    request, data, expected = _connection_case(0.0)
    slow_direct = _flight("slow-direct", "AAA", "CCC", 50, 300)
    data = RecordingData((slow_direct, *data.flights), data.outcomes)
    assert plan_scheduled_itinerary(request, data) == expected


def test_later_arrival_state_can_expose_a_flight_beyond_candidate_cap() -> None:
    request, _, _ = _connection_case(0.0)
    early_inbound = _flight("early-inbound", "AAA", "BBB", 50, 100)
    later_inbound = _flight("later-inbound", "AAA", "BBB", 60, 150)
    flights = (
        early_inbound,
        later_inbound,
        _flight("dead-end-one", "BBB", "DDD", 145, 180),
        _flight("dead-end-two", "BBB", "DDD", 150, 190),
        _flight("connection", "BBB", "CCC", 200, 250),
    )
    data = RecordingData(flights, {})
    plan = plan_scheduled_itinerary(request, data, max_candidates=2)
    assert [flight.flight_id for flight in plan] == ["later-inbound", "connection"]


def test_late_inbound_misses_fixed_cutoff_without_sampling_onward() -> None:
    request, data, itinerary = _connection_case(30.0)
    with PrebookedRouteEnv(request, data, itinerary) as env:
        env.reset(seed=3)
        _, reward, terminated, _, info = env.step(0)
        record = env.record

    assert terminated
    assert reward == 0.0
    assert record.termination_reason == "missed_connection"
    assert record.missed_flight_id == "early"
    assert record.episode.termination_reason == "no_candidates"
    assert info["missed_connection"] is True
    assert data.sampled == ["inbound"]
    verification = verify_prebooked(record, data)
    assert verification.validity
    assert verification.source_authenticated
    assert verification.score.aggregate_score == 0.0


def test_exact_fixed_cutoff_is_catchable() -> None:
    request, data, itinerary = _connection_case(0.0)
    with PrebookedRouteEnv(request, data, itinerary) as env:
        env.reset(seed=5)
        _, _, terminated, _, _ = env.step(0)
        assert not terminated
        assert env.clock_utc + request.min_connection_min == itinerary[1].scheduled_departure_utc
        assert env.available_flights == (itinerary[1],)
        _, _, terminated, _, _ = env.step(0)
        assert terminated
        record = env.record
    assert record.termination_reason == "arrived"
    assert verify_prebooked(record, data).score.aggregate_score > 0.0


def test_verifier_reconstructs_reason_and_booking_instead_of_trusting_flags() -> None:
    request, data, itinerary = _connection_case(30.0)
    with PrebookedRouteEnv(request, data, itinerary) as env:
        _run(env)
        authentic = env.record

    forged_reason = replace(authentic, termination_reason="arrived", missed_flight_id=None)
    forged_result = verify_prebooked(forged_reason, data)
    assert not forged_result.validity
    assert forged_result.source_authenticated
    assert forged_result.score.aggregate_score == 0.0

    later = next(flight for flight in data.flights if flight.flight_id == "later")
    changed_booking = replace(authentic, itinerary=(itinerary[0], later))
    changed_result = verify_prebooked(changed_booking, data)
    assert not changed_result.validity
    assert changed_result.source_authenticated
    assert changed_result.score.aggregate_score == 0.0


def test_all_unflown_booked_schedules_must_authenticate() -> None:
    request, data, itinerary = _connection_case(30.0)
    with PrebookedRouteEnv(request, data, itinerary) as env:
        _run(env)
        record = env.record
    forged_early = replace(itinerary[1], carrier="FORGED")
    forged = replace(record, itinerary=(itinerary[0], forged_early))

    verification = verify_prebooked(forged, data)
    assert verification.validity
    assert not verification.source_authenticated
    assert "source authentication failed at leg 1" in verification.reason
    assert verification.score.aggregate_score == 0.0


def test_adaptive_environment_can_take_later_service_after_same_inbound_draw() -> None:
    request, prebooked_data, itinerary = _connection_case(30.0)
    with PrebookedRouteEnv(request, prebooked_data, itinerary) as env:
        _run(env, seed=11)
        prebooked = env.record

    request, adaptive_data, _ = _connection_case(30.0)
    with FlightRouteEnv(request, adaptive_data, max_candidates=3) as env:
        env.reset(seed=11)
        _, _, terminated, _, _ = env.step(0)
        assert not terminated
        assert tuple(flight.flight_id for flight in env.available_flights) == ("later",)
        _, _, terminated, _, _ = env.step(0)
        assert terminated
        adaptive = env.record

    assert prebooked.episode.legs[0].outcome == adaptive.legs[0].outcome
    assert prebooked.termination_reason == "missed_connection"
    assert [leg.flight.flight_id for leg in adaptive.legs] == ["inbound", "later"]
    assert adaptive.termination_reason == "arrived"


@pytest.mark.parametrize(
    "itinerary",
    [
        (_flight("too-soon", "AAA", "CCC", 44, 100),),
        (
            _flight("premature", "AAA", "CCC", 60, 100),
            _flight("after-destination", "CCC", "DDD", 145, 200),
        ),
        (_flight("past-horizon", "AAA", "CCC", 60, 321),),
    ],
)
def test_booking_rejects_invalid_schedule_shape(
    itinerary: tuple[FlightCandidate, ...],
) -> None:
    request, data, _ = _connection_case(0.0)
    with pytest.raises(ValueError):
        PrebookedRouteEnv(request, data, itinerary)


@pytest.mark.parametrize("value", [0.0, ANY, np.array([0, 1])])
def test_flown_booking_selector_requires_canonical_primitive(value) -> None:
    request, data, itinerary = _connection_case(0.0)
    with PrebookedRouteEnv(request, data, itinerary) as env:
        _run(env)
        record = env.record
    forged = replace(record, itinerary=(replace(itinerary[0], dep_hour_bucket=value), itinerary[1]))
    result = verify_prebooked(forged, data)
    assert not result.source_authenticated
    assert result.score.aggregate_score == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"termination_reason": ANY},
        {"termination_reason": np.array(["arrived", "cancelled"])},
        {"missed_flight_id": ANY},
        {"itinerary": None},
        {"itinerary": ()},
    ],
)
def test_malformed_outer_booking_fails_closed(changes) -> None:
    request, data, itinerary = _connection_case(0.0)
    with PrebookedRouteEnv(request, data, itinerary) as env:
        _run(env)
        record = env.record
    result = verify_prebooked(replace(record, **changes), data)
    assert not result.validity
    assert result.score.aggregate_score == 0
