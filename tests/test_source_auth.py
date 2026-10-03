from __future__ import annotations

from dataclasses import replace
from unittest.mock import ANY

import numpy as np
import pandas as pd
import pytest

from flight_rl.data import HistoricalFlightData
from flight_rl.fixtures import DemoFlightData
from flight_rl.models import EpisodeRecord, FlightCandidate, LegRecord, SampledOutcome, TripRequest
from flight_rl.scenarios import ScenarioData, ScenarioPoolCache
from flight_rl.source_auth import SourceBackedVerifier
from flight_rl.split_data import SplitFlightData
from flight_rl.verifier import verify_deadline_first


def _flight(
    flight_id: str = "flight-a",
    *,
    carrier: str = "XX",
    departure: int = 1_100,
    arrival: int = 1_250,
) -> FlightCandidate:
    return FlightCandidate(
        flight_id=flight_id,
        origin="AAA",
        dest="CCC",
        carrier=carrier,
        scheduled_departure_utc=departure,
        scheduled_arrival_utc=arrival,
        flight_date="2024-01-01",
        dep_hour_bucket=1,
        season="DJF",
        operating_carrier=carrier,
        operating_flight_number="10",
    )


def _request() -> TripRequest:
    return TripRequest("AAA", "CCC", 1_000, 1_350, 1_500, min_connection_min=30)


def _record(flight: FlightCandidate, outcome: SampledOutcome) -> EpisodeRecord:
    request = _request()
    if outcome.cancelled:
        leg = LegRecord(flight, outcome, None, None, flight.origin)
        return EpisodeRecord(
            request, (leg,), flight.origin, flight.scheduled_departure_utc, "cancelled"
        )

    assert outcome.dep_delay_min is not None
    departure = flight.scheduled_departure_utc + round(outcome.dep_delay_min)
    if outcome.diverted:
        if outcome.div_arr_delay_min is not None:
            arrival = flight.scheduled_arrival_utc + round(outcome.div_arr_delay_min)
        elif outcome.div_actual_elapsed_min is not None:
            arrival = departure + round(outcome.div_actual_elapsed_min)
        else:
            arrival = None
        resolved = flight.dest if outcome.div_reached_dest else outcome.div_airport
    else:
        assert outcome.arr_delay_min is not None
        arrival = flight.scheduled_arrival_utc + round(outcome.arr_delay_min)
        resolved = flight.dest
    leg = LegRecord(flight, outcome, departure, arrival, resolved)
    if outcome.diverted and not outcome.div_reached_dest:
        clock = departure if arrival is None else arrival
        return EpisodeRecord(request, (leg,), flight.origin, clock, "unresolved_diversion")
    assert arrival is not None
    return EpisodeRecord(request, (leg,), flight.dest, arrival, "arrived")


def _demo_source() -> tuple[FlightCandidate, DemoFlightData]:
    flight = _flight()
    outcomes = (
        SampledOutcome("normal", dep_delay_min=5.0, arr_delay_min=10.0),
        SampledOutcome(
            "cancelled", cancelled=True, diverted=True, dep_delay_min=None, arr_delay_min=None
        ),
        SampledOutcome(
            "diverted",
            diverted=True,
            dep_delay_min=4.0,
            arr_delay_min=None,
            div_reached_dest=True,
            div_arr_delay_min=20.0,
            div_actual_elapsed_min=166.0,
            div_airport="BBB",
        ),
    )
    return flight, DemoFlightData((flight,), {flight.flight_id: outcomes})


def _source_outcome(data: object, flight: FlightCandidate, donor_id: str) -> SampledOutcome:
    outcome = data.source_transition_outcome(flight.flight_id, donor_id)  # type: ignore[attr-defined]
    assert isinstance(outcome, SampledOutcome)
    return outcome


@pytest.mark.parametrize("value", [ANY, np.array([1, 2])])
@pytest.mark.parametrize("target", ["flight", "outcome"])
def test_nonprimitive_fields_cannot_override_source_equality(value, target):
    flight, data = _demo_source()
    record = _record(flight, _source_outcome(data, flight, "normal"))
    leg = record.legs[0]
    if target == "flight":
        leg = replace(leg, flight=replace(leg.flight, carrier=value))
    else:
        leg = replace(leg, outcome=replace(leg.outcome, actual_elapsed_min=value))
    result = SourceBackedVerifier(data).verify(replace(record, legs=(leg,)))
    assert not result.authentication.authenticated
    assert result.score.aggregate_score == 0


@pytest.mark.parametrize("donor_id", ["normal", "cancelled", "diverted"])
def test_authentic_normal_cancellation_and_diversion_records_pass(donor_id: str) -> None:
    flight, data = _demo_source()
    record = _record(flight, _source_outcome(data, flight, donor_id))

    result = SourceBackedVerifier(data).verify(record)

    assert result.authentication.authenticated
    assert result.authentication.reason == "authenticated"
    assert len(result.score.breakdown) == 3


def test_self_consistent_altered_donor_payload_is_hard_gated_to_zero() -> None:
    flight, data = _demo_source()
    authentic = _source_outcome(data, flight, "normal")
    forged = replace(authentic, arr_delay_min=30.0)
    record = _record(flight, forged)
    assert verify_deadline_first(record).aggregate_score > 0.0

    result = SourceBackedVerifier(data).verify(record)

    assert result.authentication.reason == "outcome_payload_mismatch"
    assert result.authentication.leg_index == 0
    assert result.score.aggregate_score == 0.0
    assert all(item.raw_score == 0.0 for item in result.score.breakdown)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("carrier", "FORGED"),
        ("scheduled_departure_utc", 1_110),
        ("scheduled_arrival_utc", 1_260),
        ("dep_hour_bucket", 2),
        ("season", "JJA"),
    ],
)
def test_changed_schedule_and_pool_selectors_are_rejected(field: str, value: object) -> None:
    flight, data = _demo_source()
    forged_flight = replace(flight, **{field: value})
    outcome = _source_outcome(data, flight, "normal")
    record = _record(forged_flight, outcome)
    assert verify_deadline_first(record).aggregate_score > 0.0

    result = SourceBackedVerifier(data).verify(record)

    assert result.authentication.reason == "flight_payload_mismatch"
    assert result.score.aggregate_score == 0.0


@pytest.mark.parametrize(
    "mutation",
    [
        {"donor_id": "unknown"},
        {"support": 999},
        {"fallback_level": "forged"},
        {"actual_elapsed_min": 999.0},
        {"diverted": True},
    ],
)
def test_unknown_donor_and_unused_or_derived_field_mutations_are_rejected(
    mutation: dict[str, object],
) -> None:
    flight, data = _demo_source()
    authentic = _source_outcome(data, flight, "normal")
    forged = replace(authentic, **mutation)

    result = SourceBackedVerifier(data).verify(_record(flight, forged))

    expected = (
        "unknown_or_ineligible_donor"
        if mutation.get("donor_id") == "unknown"
        else "outcome_payload_mismatch"
    )
    assert result.authentication.reason == expected
    assert result.score.aggregate_score == 0.0


@pytest.mark.parametrize(
    ("donor_id", "mutation"),
    [
        ("cancelled", {"diverted": False}),
        ("diverted", {"div_airport": "ZZZ"}),
        ("diverted", {"div_actual_elapsed_min": 999.0}),
    ],
)
def test_cancellation_and_diversion_payload_mutations_are_rejected(
    donor_id: str, mutation: dict[str, object]
) -> None:
    flight, data = _demo_source()
    authentic = _source_outcome(data, flight, donor_id)
    forged = replace(authentic, **mutation)

    result = SourceBackedVerifier(data).verify(_record(flight, forged))

    assert result.authentication.reason == "outcome_payload_mismatch"
    assert result.score.aggregate_score == 0.0


def _schedule_row(flight: FlightCandidate) -> dict[str, object]:
    return {
        "flight_id": flight.flight_id,
        "flight_date": flight.flight_date,
        "origin": flight.origin,
        "dest": flight.dest,
        "carrier": flight.carrier,
        "operating_carrier": flight.operating_carrier,
        "operating_flight_number": flight.operating_flight_number,
        "scheduled_departure_utc": flight.scheduled_departure_utc,
        "scheduled_arrival_utc": flight.scheduled_arrival_utc,
        "dep_hour_bucket": flight.dep_hour_bucket,
        "season": flight.season,
    }


def _donor_row(donor_id: str, flight: FlightCandidate) -> dict[str, object]:
    return {
        "donor_id": donor_id,
        "origin": flight.origin,
        "dest": flight.dest,
        "carrier": flight.carrier,
        "dep_hour_bucket": flight.dep_hour_bucket,
        "season": flight.season,
        "dep_delay_min": 5.0,
        "arr_delay_min": 10.0,
        "actual_elapsed_min": 155.0,
        "cancelled": False,
        "diverted": False,
        "div_reached_dest": False,
        "div_arr_delay_min": None,
        "div_actual_elapsed_min": None,
        "div_airport": None,
    }


def test_historical_lookup_rejects_wrong_pool_without_materializing_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _flight("flight-a", carrier="XX")
    second = _flight("flight-b", carrier="YY", departure=1_120, arrival=1_270)
    data = HistoricalFlightData(
        pd.DataFrame([_schedule_row(first), _schedule_row(second)]),
        pd.DataFrame([_donor_row("donor-a", first), _donor_row("donor-b", second)]),
        min_support=1,
    )
    authentic = _source_outcome(data, first, "donor-a")
    wrong_pool = _source_outcome(data, second, "donor-b")

    def fail_history(_flight: FlightCandidate):
        raise AssertionError("source authentication must not materialize historical_outcomes")

    monkeypatch.setattr(data, "historical_outcomes", fail_history)
    wrapped = ScenarioData(ScenarioPoolCache(data), scenario_seed=7)
    verifier = SourceBackedVerifier(wrapped)

    assert verifier.verify(_record(first, authentic)).authentication.authenticated
    rejected = verifier.verify(_record(first, wrong_pool))
    assert rejected.authentication.reason == "unknown_or_ineligible_donor"
    assert rejected.score.aggregate_score == 0.0


def test_split_authenticates_heldout_transition_and_rejects_fit_substitution() -> None:
    flight = _flight()
    schedule = DemoFlightData(
        (flight,), {flight.flight_id: (SampledOutcome("schedule", arr_delay_min=0.0),)}
    )
    fit = DemoFlightData(
        (flight,), {flight.flight_id: (SampledOutcome("fit-only", arr_delay_min=0.0),)}
    )
    heldout = DemoFlightData(
        (flight,), {flight.flight_id: (SampledOutcome("heldout", arr_delay_min=20.0),)}
    )
    split = SplitFlightData(schedule, fit, heldout)
    verifier = SourceBackedVerifier(split)

    heldout_record = _record(flight, _source_outcome(heldout, flight, "heldout"))
    fit_record = _record(flight, _source_outcome(fit, flight, "fit-only"))

    assert verifier.verify(heldout_record).authentication.authenticated
    rejected = verifier.verify(fit_record)
    assert rejected.authentication.reason == "unknown_or_ineligible_donor"
    assert rejected.score.aggregate_score == 0.0


def test_split_source_flight_honors_route_filter_and_transition_schedule_identity() -> None:
    flight = _flight()
    source = DemoFlightData(
        (flight,), {flight.flight_id: (SampledOutcome("source", arr_delay_min=0.0),)}
    )
    excluded = SplitFlightData(source, source, source, candidate_routes=(("AAA", "BBB"),))
    assert excluded.source_flight(flight.flight_id) is None

    drifted_flight = replace(flight, carrier="YY", operating_carrier="YY")
    drifted = DemoFlightData(
        (drifted_flight,),
        {drifted_flight.flight_id: (SampledOutcome("drifted", arr_delay_min=0.0),)},
    )
    split = SplitFlightData(source, source, drifted)
    assert split.source_transition_outcome(flight.flight_id, "drifted") is None


def test_profiles_and_unsupported_source_failures_are_explicit() -> None:
    flight, data = _demo_source()
    record = _record(flight, _source_outcome(data, flight, "normal"))
    verifier = SourceBackedVerifier(data)

    assert len(verifier.verify(record).score.breakdown) == 3
    assert len(verifier.verify(record, "legacy_six_v1").score.breakdown) == 6
    with pytest.raises(ValueError, match="Unknown score profile"):
        verifier.verify(record, "unknown")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="source_flight"):
        SourceBackedVerifier(object())  # type: ignore[arg-type]
