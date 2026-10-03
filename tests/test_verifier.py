from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from flight_rl.models import EpisodeRecord, FlightCandidate, LegRecord, SampledOutcome, TripRequest
from flight_rl.verifier import (
    DEADLINE_FIRST_V1_WEIGHTS,
    LEGACY_SIX_V1_WEIGHTS,
    PRIMARY_SCORE_PROFILE,
    Criterion,
    Verifier,
    episode_metrics,
    make_default_verifier,
    make_legacy_six_v1_verifier,
    validated_deadline_first_scores,
    verify_deadline_first,
    verify_episode,
    verify_legacy_six_v1,
)


@pytest.fixture
def trip_request() -> TripRequest:
    return TripRequest(
        origin="AAA",
        destination="CCC",
        ready_utc=1_000,
        deadline_utc=1_300,
        horizon_utc=1_500,
        max_attempts=3,
        min_connection_min=30,
        delay_budget_min=180,
    )


def flight(
    *,
    flight_id: str = "F1",
    origin: str = "AAA",
    dest: str = "CCC",
    departure: int = 1_100,
    arrival: int = 1_250,
) -> FlightCandidate:
    return FlightCandidate(
        flight_id=flight_id,
        origin=origin,
        dest=dest,
        carrier="XX",
        scheduled_departure_utc=departure,
        scheduled_arrival_utc=arrival,
        flight_date="2024-01-01",
        dep_hour_bucket=1,
        season="DJF",
    )


def ordinary_leg(*, selected: FlightCandidate | None = None, donor_id: str = "D1") -> LegRecord:
    selected = selected or flight()
    outcome = SampledOutcome(
        donor_id=donor_id,
        dep_delay_min=10.4,
        arr_delay_min=20.4,
        actual_elapsed_min=160.0,
        support=40,
    )
    return LegRecord(
        flight=selected,
        outcome=outcome,
        departure_utc=selected.scheduled_departure_utc + 10,
        arrival_utc=selected.scheduled_arrival_utc + 20,
        resolved_airport=selected.dest,
    )


def arrived_record(request: TripRequest, *, leg: LegRecord | None = None) -> EpisodeRecord:
    leg = leg or ordinary_leg()
    assert leg.arrival_utc is not None
    return EpisodeRecord(
        request=request,
        legs=(leg,),
        final_airport=request.destination,
        clock_utc=leg.arrival_utc,
        termination_reason="arrived",
    )


def direct_arrival_record(
    request: TripRequest,
    arrival_utc: int,
    *,
    scheduled_arrival_utc: int | None = None,
) -> EpisodeRecord:
    if scheduled_arrival_utc is None:
        scheduled_arrival_utc = arrival_utc
    selected = flight(departure=1_050, arrival=scheduled_arrival_utc)
    outcome = SampledOutcome(
        donor_id=f"arrival-{arrival_utc}",
        dep_delay_min=0,
        arr_delay_min=arrival_utc - scheduled_arrival_utc,
    )
    leg = LegRecord(selected, outcome, 1_050, arrival_utc, request.destination)
    return EpisodeRecord(request, (leg,), request.destination, arrival_utc, "arrived")


def scores(record: EpisodeRecord) -> dict[str, float]:
    return {item.name: item.raw_score for item in verify_legacy_six_v1(record).breakdown}


def test_default_verifier_is_deadline_first_v1(trip_request: TripRequest) -> None:
    record = arrived_record(trip_request)
    result = verify_episode(record)
    by_name = {item.name: item for item in result.breakdown}

    assert PRIMARY_SCORE_PROFILE == "deadline_first_v1"
    assert tuple(by_name) == ("on_time_arrival", "arrived", "earliness")
    assert dict(DEADLINE_FIRST_V1_WEIGHTS) == {
        "on_time_arrival": 0.8,
        "arrived": 0.1,
        "earliness": 0.1,
    }
    assert by_name["on_time_arrival"].raw_score == 1.0
    assert by_name["arrived"].raw_score == 1.0
    assert by_name["earliness"].raw_score == pytest.approx((1_500 - 1_270) / 500)
    assert result == verify_deadline_first(record)


def test_legacy_six_v1_preserves_the_original_vector(trip_request: TripRequest) -> None:
    record = arrived_record(trip_request)
    result = verify_legacy_six_v1(record)
    by_name = {item.name: item for item in result.breakdown}

    assert tuple(by_name) == tuple(LEGACY_SIX_V1_WEIGHTS)
    assert [item.weight for item in result.breakdown] == pytest.approx(
        [0.4, 0.25, 0.1, 0.1, 0.05, 0.1]
    )
    assert by_name["arrived"].raw_score == 1.0
    assert by_name["on_time_arrival"].raw_score == 1.0
    assert by_name["total_delay"].raw_score == pytest.approx(1.0 - 20 / 180)
    assert by_name["connections_count"].raw_score == 1.0
    assert by_name["cancellation_exposure"].raw_score == 1.0
    assert by_name["connection_buffer"].raw_score == 1.0
    assert result.aggregate_score == pytest.approx(1 - 0.1 * 20 / 180)


def test_arrival_after_deadline_is_still_valid_arrival(trip_request: TripRequest) -> None:
    late_leg = ordinary_leg(
        selected=flight(departure=1_250, arrival=1_400),
        donor_id="late-donor",
    )
    record = arrived_record(trip_request, leg=late_leg)

    assert episode_metrics(record) == {
        "validity": True,
        "arrived": True,
        "on_time_arrival": False,
        "elapsed_min": 420,
        "disrupted": False,
        "attempts": 1,
        "termination_reason": "arrived",
    }
    assert scores(record) == pytest.approx(
        {
            "arrived": 1.0,
            "on_time_arrival": 0.0,
            "total_delay": 1 - 20 / 180,
            "connections_count": 1.0,
            "cancellation_exposure": 1.0,
            "connection_buffer": 1.0,
        }
    )


def test_deadline_first_breakdown_and_boundary_dominance(trip_request: TripRequest) -> None:
    at_deadline = verify_deadline_first(direct_arrival_record(trip_request, 1_300))
    just_late = verify_deadline_first(direct_arrival_record(trip_request, 1_301))
    at_horizon = verify_deadline_first(direct_arrival_record(trip_request, 1_500))

    assert [item.name for item in at_deadline.breakdown] == [
        "on_time_arrival",
        "arrived",
        "earliness",
    ]
    assert [item.weight for item in at_deadline.breakdown] == pytest.approx([0.8, 0.1, 0.1])
    assert [item.raw_score for item in at_deadline.breakdown] == pytest.approx([1.0, 1.0, 0.4])
    assert at_deadline.aggregate_score == pytest.approx(0.94)
    assert at_deadline.aggregate_score > just_late.aggregate_score > at_horizon.aggregate_score
    assert at_horizon.aggregate_score == pytest.approx(0.1)


def test_deadline_first_handles_expired_deadline_and_rejects_inconsistent_record(
    trip_request: TripRequest,
) -> None:
    expired = replace(trip_request, deadline_utc=900)
    earlier = direct_arrival_record(expired, 1_200)
    later = direct_arrival_record(expired, 1_300)

    assert (
        verify_deadline_first(earlier).aggregate_score
        > verify_deadline_first(later).aggregate_score
    )

    nonarrival = EpisodeRecord(expired, (), expired.origin, expired.ready_utc, "no_candidates")
    assert verify_deadline_first(nonarrival).aggregate_score == 0.0

    inconsistent = replace(earlier, clock_utc=earlier.clock_utc + 1)
    result = verify_deadline_first(inconsistent)
    assert result.aggregate_score == 0.0
    assert all(item.raw_score == 0.0 for item in result.breakdown)


def test_multileg_record_requires_spatial_and_connection_continuity(
    trip_request: TripRequest,
) -> None:
    first_flight = flight(flight_id="AB", dest="BBB", departure=1_060, arrival=1_150)
    first = LegRecord(
        first_flight,
        SampledOutcome("AB-donor", dep_delay_min=0, arr_delay_min=0, actual_elapsed_min=90),
        1_060,
        1_150,
        "BBB",
    )
    second_flight = flight(flight_id="BC", origin="BBB", dest="CCC", departure=1_180, arrival=1_280)
    second = LegRecord(
        second_flight,
        SampledOutcome("BC-donor", dep_delay_min=0, arr_delay_min=0, actual_elapsed_min=100),
        1_180,
        1_280,
        "CCC",
    )
    record = EpisodeRecord(trip_request, (first, second), "CCC", 1_280, "arrived")
    assert episode_metrics(record)["validity"] is True

    too_tight = replace(
        second,
        flight=replace(second.flight, scheduled_departure_utc=1_179),
        departure_utc=1_179,
    )
    forged = replace(record, legs=(first, too_tight))
    assert episode_metrics(forged)["validity"] is False
    assert all(value == 0.0 for value in scores(forged).values())

    wrong_origin = replace(
        second,
        flight=replace(second.flight, origin="DDD"),
    )
    forged = replace(record, legs=(first, wrong_origin))
    assert episode_metrics(forged)["validity"] is False
    assert all(value == 0.0 for value in scores(forged).values())


def test_diversion_reaching_destination_is_arrival_but_disrupted(
    trip_request: TripRequest,
) -> None:
    selected = flight()
    outcome = SampledOutcome(
        donor_id="diverted-donor",
        diverted=True,
        dep_delay_min=5.0,
        arr_delay_min=None,
        div_reached_dest=True,
        div_arr_delay_min=25.0,
        support=31,
    )
    leg = LegRecord(selected, outcome, 1_105, 1_275, "CCC")
    record = arrived_record(trip_request, leg=leg)

    assert episode_metrics(record)["disrupted"] is True
    assert scores(record)["cancellation_exposure"] == 1.0
    assert scores(record)["total_delay"] == pytest.approx(1 - 25 / 180)


def test_cancellation_is_valid_nonarrival_and_every_score_is_zero(
    trip_request: TripRequest,
) -> None:
    selected = flight()
    outcome = SampledOutcome(donor_id="cancelled-donor", cancelled=True, diverted=True, support=40)
    leg = LegRecord(selected, outcome, None, None, "AAA")
    record = EpisodeRecord(trip_request, (leg,), "AAA", 1_100, "cancelled")

    metrics = episode_metrics(record)
    assert metrics["validity"] is True
    assert metrics["disrupted"] is True
    assert all(value == 0.0 for value in scores(record).values())


@pytest.mark.parametrize(
    ("outcome", "arrival", "resolved"),
    [
        (
            SampledOutcome(
                donor_id="known-diversion",
                diverted=True,
                dep_delay_min=5.0,
                div_reached_dest=False,
                div_arr_delay_min=30.0,
                div_airport="DDD",
                support=50,
            ),
            1_280,
            "DDD",
        ),
        (
            SampledOutcome(
                donor_id="unknown-diversion",
                diverted=True,
                dep_delay_min=5.0,
                div_reached_dest=False,
                div_arr_delay_min=None,
                div_actual_elapsed_min=None,
                div_airport="DDD",
                support=50,
            ),
            None,
            "DDD",
        ),
    ],
)
def test_unresolved_diversion_preserves_detail_but_not_claimed_position(
    trip_request: TripRequest,
    outcome: SampledOutcome,
    arrival: int | None,
    resolved: str,
) -> None:
    leg = LegRecord(flight(), outcome, 1_105, arrival, resolved)
    expected_clock = arrival if arrival is not None else 1_105
    record = EpisodeRecord(trip_request, (leg,), "AAA", expected_clock, "unresolved_diversion")

    assert episode_metrics(record)["validity"] is True
    assert all(value == 0.0 for value in scores(record).values())


def test_destination_at_horizon_succeeds_but_beyond_horizon_does_not(
    trip_request: TripRequest,
) -> None:
    at_horizon_flight = flight(departure=1_300, arrival=1_480)
    at_horizon_outcome = SampledOutcome(donor_id="at-horizon", dep_delay_min=0, arr_delay_min=20)
    at_horizon_leg = LegRecord(at_horizon_flight, at_horizon_outcome, 1_300, 1_500, "CCC")
    at_horizon = EpisodeRecord(trip_request, (at_horizon_leg,), "CCC", 1_500, "arrived")
    assert episode_metrics(at_horizon)["validity"] is True
    assert scores(at_horizon)["arrived"] == 1.0

    beyond_flight = flight(departure=1_300, arrival=1_490)
    beyond_outcome = SampledOutcome(donor_id="beyond", dep_delay_min=0, arr_delay_min=20)
    beyond_leg = LegRecord(beyond_flight, beyond_outcome, 1_300, 1_510, "CCC")
    beyond = EpisodeRecord(trip_request, (beyond_leg,), "AAA", 1_500, "horizon")
    assert episode_metrics(beyond)["validity"] is True
    assert all(value == 0.0 for value in scores(beyond).values())


def test_intermediate_arrival_at_horizon_is_failure_at_physical_airport(
    trip_request: TripRequest,
) -> None:
    selected = flight(dest="BBB", departure=1_300, arrival=1_480)
    outcome = SampledOutcome(donor_id="connection-at-horizon", dep_delay_min=0, arr_delay_min=20)
    leg = LegRecord(selected, outcome, 1_300, 1_500, "BBB")
    record = EpisodeRecord(trip_request, (leg,), "BBB", 1_500, "horizon")

    assert episode_metrics(record)["validity"] is True
    assert all(value == 0.0 for value in scores(record).values())


def test_source_row_swap_cannot_forge_an_arrival(trip_request: TripRequest) -> None:
    truthful_leg = ordinary_leg()
    forged_outcome = replace(truthful_leg.outcome, donor_id="different-source-row", arr_delay_min=5)
    forged_leg = replace(truthful_leg, outcome=forged_outcome)
    forged_record = arrived_record(trip_request, leg=forged_leg)

    assert episode_metrics(forged_record)["validity"] is False
    assert all(value == 0.0 for value in scores(forged_record).values())


def test_authoritative_delay_fields_win_over_inconsistent_redundant_elapsed(
    trip_request: TripRequest,
) -> None:
    leg = ordinary_leg()
    leg = replace(leg, outcome=replace(leg.outcome, actual_elapsed_min=999.0))
    record = arrived_record(trip_request, leg=leg)

    # Real BTS rows occasionally disagree in redundant elapsed fields. The
    # environment resolves ordinary legs from DepDelay and ArrDelay, so the
    # verifier checks those same authoritative source fields.
    assert episode_metrics(record)["validity"] is True
    assert scores(record)["arrived"] == 1.0


@pytest.mark.parametrize(
    "record_factory",
    [
        lambda request: replace(arrived_record(request), final_airport="BBB"),
        lambda request: replace(arrived_record(request), termination_reason="unknown"),
        lambda request: arrived_record(
            request,
            leg=ordinary_leg(selected=flight(departure=1_020, arrival=1_250)),
        ),
        lambda request: arrived_record(
            request,
            leg=LegRecord(
                flight(),
                SampledOutcome(donor_id="early", dep_delay_min=-200, arr_delay_min=20),
                900,
                1_270,
                "CCC",
            ),
        ),
    ],
)
def test_spatial_reason_boarding_and_realized_time_forgeries_score_zero(
    trip_request: TripRequest,
    record_factory,
) -> None:
    record = record_factory(trip_request)
    assert episode_metrics(record)["validity"] is False
    assert all(value == 0.0 for value in scores(record).values())


def test_metrics_are_json_safe_for_invalid_record(trip_request: TripRequest) -> None:
    record = replace(arrived_record(trip_request), clock_utc=1_301)
    metrics = episode_metrics(record)
    assert metrics["validity"] is False
    assert json.loads(json.dumps(metrics)) == metrics


@pytest.mark.parametrize("bad_weight", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_criterion_rejects_negative_or_nonfinite_weight(bad_weight: float) -> None:
    with pytest.raises(ValueError):
        Criterion("bad", bad_weight, "invalid weight", lambda _: 0.0)


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), float("-inf"), -0.1, 1.1])
def test_verifier_rejects_nonfinite_or_out_of_range_score(bad_score: float) -> None:
    verifier = Verifier((Criterion("bad", 1.0, "invalid score", lambda _: bad_score),))
    with pytest.raises(ValueError):
        verifier.score(None)


def test_verifier_requires_positive_sum_unique_names_and_immutable_results() -> None:
    zero = Criterion("zero", 0.0, "zero weight", lambda _: 1.0)
    with pytest.raises(ValueError):
        Verifier((zero,))
    with pytest.raises(ValueError):
        Verifier(
            (
                Criterion("same", 1.0, "first", lambda _: 1.0),
                Criterion("same", 1.0, "second", lambda _: 0.0),
            )
        )
    with pytest.raises(ValueError):
        Verifier(
            (
                Criterion("huge-a", 1e308, "large", lambda _: 1.0),
                Criterion("huge-b", 1e308, "large", lambda _: 1.0),
            )
        )

    source = [Criterion("ok", 1.0, "valid", lambda _: 1.0)]
    verifier = Verifier(source)
    source.clear()
    assert len(verifier.criteria) == 1
    with pytest.raises(AttributeError):
        verifier.criteria = ()  # type: ignore[misc]
    result = verifier.score(None)
    assert isinstance(result.breakdown, tuple)
    with pytest.raises(FrozenInstanceError):
        result.aggregate_score = 0.0  # type: ignore[misc]


def test_make_default_verifier_returns_independent_fixed_instances() -> None:
    first = make_default_verifier()
    second = make_default_verifier()
    assert first is not second
    assert first.criteria is not second.criteria
    assert tuple(item.name for item in first.criteria) == tuple(
        item.name for item in second.criteria
    )
    assert tuple(item.name for item in make_legacy_six_v1_verifier().criteria) == tuple(
        LEGACY_SIX_V1_WEIGHTS
    )


def test_default_verifier_validates_once_and_still_rejects_forgery(
    monkeypatch: pytest.MonkeyPatch, trip_request: TripRequest
) -> None:
    import flight_rl.verifier as verifier_module

    record = arrived_record(trip_request)
    forged = replace(record, clock_utc=record.clock_utc + 1)
    original = verifier_module._record_is_valid
    validation_calls: list[object] = []

    def count_validation(candidate: object) -> bool:
        validation_calls.append(candidate)
        return original(candidate)

    monkeypatch.setattr(verifier_module, "_record_is_valid", count_validation)

    result = make_default_verifier().score(record)
    assert len(validation_calls) == 1
    assert result == verify_deadline_first(record)

    validation_calls.clear()
    forged_result = verify_episode(forged)
    assert len(validation_calls) == 1
    assert forged_result.aggregate_score == 0.0
    assert all(item.raw_score == 0.0 for item in forged_result.breakdown)


def test_public_validated_scores_match_direct_default_criteria(
    trip_request: TripRequest,
) -> None:
    record = arrived_record(trip_request)
    expected = validated_deadline_first_scores(
        record.request,
        arrived=True,
        clock_utc=record.clock_utc,
    )
    direct = {
        criterion.name: criterion.score_fn(record) for criterion in make_default_verifier().criteria
    }

    assert direct == pytest.approx(expected)


def _quality_trip(request, *, gap=90, first_delay=0, second_delay=0):
    first_flight = flight(flight_id="AB", dest="BBB", departure=1045, arrival=1100)
    second_flight = flight(flight_id="BC", origin="BBB", departure=1100 + gap, arrival=1250)
    legs = tuple(
        LegRecord(
            selected,
            SampledOutcome(selected.flight_id, arr_delay_min=delay),
            selected.scheduled_departure_utc,
            selected.scheduled_arrival_utc + delay,
            selected.dest,
        )
        for selected, delay in ((first_flight, first_delay), (second_flight, second_delay))
    )
    return EpisodeRecord(request, legs, "CCC", 1250 + second_delay, "arrived")


def test_deadline_first_prefers_absolute_earlier_arrival_over_rubric_quirks(trip_request):
    earlier_connected_and_delayed = _quality_trip(trip_request, gap=30, second_delay=20)
    later_direct = direct_arrival_record(trip_request, 1_280)

    assert (
        verify_legacy_six_v1(earlier_connected_and_delayed).aggregate_score
        < verify_legacy_six_v1(later_direct).aggregate_score
    )
    assert (
        verify_deadline_first(earlier_connected_and_delayed).aggregate_score
        > verify_deadline_first(later_direct).aggregate_score
    )


def test_buffer_changes_score_for_equally_fast_completed_connections(trip_request):
    tight = _quality_trip(trip_request, gap=30)
    roomy = _quality_trip(trip_request, gap=90)
    assert episode_metrics(tight)["validity"]
    assert episode_metrics(roomy)["validity"]
    tight_scores, roomy_scores = scores(tight), scores(roomy)
    assert tight_scores.pop("connection_buffer") == pytest.approx(1 / 3)
    assert roomy_scores.pop("connection_buffer") == 1
    assert tight_scores == roomy_scores
    assert verify_legacy_six_v1(roomy).aggregate_score - verify_legacy_six_v1(
        tight
    ).aggregate_score == pytest.approx(0.1 * 2 / 3)


def test_connections_penalize_equal_arrival_without_penalizing_nonstop_buffer(trip_request):
    connected = _quality_trip(trip_request)
    selected = flight(departure=1045, arrival=1250)
    direct = EpisodeRecord(
        trip_request,
        (LegRecord(selected, SampledOutcome("direct"), 1045, 1250, "CCC"),),
        "CCC",
        1250,
        "arrived",
    )
    direct_scores, connected_scores = scores(direct), scores(connected)
    assert direct_scores.pop("connections_count") == 1
    assert connected_scores.pop("connections_count") == 0.5
    assert direct_scores == connected_scores
    assert verify_legacy_six_v1(direct).aggregate_score - verify_legacy_six_v1(
        connected
    ).aggregate_score == pytest.approx(0.05)


def test_total_delay_sums_late_legs_without_offset_from_early_arrival(trip_request):
    request = replace(trip_request, deadline_utc=1450)
    two_late = _quality_trip(request, first_delay=60, second_delay=30)
    early_then_late = _quality_trip(request, first_delay=-10, second_delay=60)
    assert episode_metrics(two_late)["validity"]
    assert episode_metrics(early_then_late)["validity"]
    assert scores(two_late)["total_delay"] == pytest.approx(0.5)
    assert scores(early_then_late)["total_delay"] == pytest.approx(1 - 60 / 180)
    capped = replace(two_late, request=replace(request, delay_budget_min=60))
    assert scores(capped)["total_delay"] == 0
