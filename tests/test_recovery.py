from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from flight_rl.baselines import NonstopFirstPolicy
from flight_rl.env import FlightRouteEnv
from flight_rl.models import FlightCandidate, OutcomeSummary, SampledOutcome, TripRequest
from flight_rl.recovery import (
    RecoveryConfig,
    RecoveryEnv,
    evaluate_recovery_policy,
    recovery_metrics,
    verify_recovery,
    verify_recovery_legacy_six_v1,
)


def flight(
    flight_id: str,
    origin: str,
    dest: str,
    departure: int,
    arrival: int,
) -> FlightCandidate:
    return FlightCandidate(
        flight_id=flight_id,
        origin=origin,
        dest=dest,
        carrier="ZZ",
        scheduled_departure_utc=departure,
        scheduled_arrival_utc=arrival,
        flight_date="2025-01-15",
        dep_hour_bucket=departure // 60 % 24,
        season="winter",
    )


def ordinary(donor_id: str = "ordinary", *, arrival_delay: float = 0.0) -> SampledOutcome:
    return SampledOutcome(
        donor_id,
        dep_delay_min=0.0,
        arr_delay_min=arrival_delay,
        actual_elapsed_min=120.0,
    )


def cancelled(donor_id: str = "cancelled") -> SampledOutcome:
    return SampledOutcome(
        donor_id,
        cancelled=True,
        dep_delay_min=None,
        arr_delay_min=None,
        actual_elapsed_min=None,
    )


def request(
    *,
    deadline: int = 1_600,
    horizon: int = 1_800,
    attempts: int = 3,
    connection: int = 30,
    budget: int = 600,
) -> TripRequest:
    return TripRequest(
        "AAA",
        "BBB",
        1_000,
        deadline,
        horizon,
        max_attempts=attempts,
        min_connection_min=connection,
        delay_budget_min=budget,
    )


class StubData:
    airports = ("AAA", "BBB", "CCC")
    carriers = ("ZZ",)

    def __init__(
        self,
        flights: tuple[FlightCandidate, ...],
        outcomes: dict[str, SampledOutcome],
    ) -> None:
        self.flights = flights
        self.outcomes = outcomes
        self.candidate_calls: list[tuple[str, int, int, int]] = []

    def candidates(self, origin: str, earliest_utc: int, horizon_utc: int, limit: int = 64):
        self.candidate_calls.append((origin, earliest_utc, horizon_utc, limit))
        eligible = [
            item
            for item in self.flights
            if item.origin == origin and earliest_utc <= item.scheduled_departure_utc <= horizon_utc
        ]
        return tuple(eligible[:limit])

    def sample_outcome(self, selected: FlightCandidate, rng):
        del rng
        return self.outcomes[selected.flight_id]

    def outcome_summary(self, selected: FlightCandidate) -> OutcomeSummary:
        outcome = self.outcomes[selected.flight_id]
        return OutcomeSummary(
            p_cancelled=float(outcome.cancelled),
            p_diverted=float(outcome.diverted),
            mean_arrival_delay_min=0.0,
            support=10,
            fallback_level="fixture",
        )

    def historical_outcomes(self, selected: FlightCandidate):
        return (self.outcomes[selected.flight_id],)


def assert_observations_equal(left, right) -> None:
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def test_recovery_defaults_to_primary_and_canonicalizes_the_legacy_alias() -> None:
    primary = RecoveryEnv(request(), StubData((), {}), RecoveryConfig())
    legacy = RecoveryEnv(request(), StubData((), {}), RecoveryConfig(), reward_mode="rubric")

    assert primary.reward_mode == "deadline_first"
    assert legacy.reward_mode == "legacy_six_v1"


def successful_recovery_record():
    first = flight("cancel", "AAA", "BBB", 1_060, 1_160)
    second = flight("replacement", "AAA", "BBB", 1_150, 1_300)
    data = StubData(
        (first, second),
        {"cancel": cancelled(), "replacement": ordinary()},
    )
    env = RecoveryEnv(request(), data, RecoveryConfig(20, 30, 2), max_candidates=4)
    observation, _ = env.reset(seed=4)
    assert observation["action_mask"].tolist() == [1, 1, 0, 0]
    observation, reward, terminated, truncated, info = env.step(0)
    assert (reward, terminated, truncated) == (0.0, False, False)
    assert env.observation_space.contains(observation)
    assert observation["time"].tolist() == [110.0, 490.0, 690.0, 2.0]
    assert observation["disrupted"] == 1
    assert observation["candidates"][0, 2] == pytest.approx(40.0)
    event = info["recovery_event"]
    assert event.segment_index == 0
    assert event.cancelled_flight_id == "cancel"
    assert event.airport == "AAA"
    assert event.cancellation_utc == 1_060
    assert event.notification_ready_utc == 1_080
    assert event.restart_ready_utc == 1_110
    assert event.attempts_used == 1
    assert event.attempts_remaining == 2
    assert event.rebookings_before == 0
    assert event.decision == "rebooked"
    assert data.candidate_calls == [
        ("AAA", 1_030, 1_800, 4),
        ("AAA", 1_140, 1_800, 4),
    ]

    _, reward, terminated, truncated, info = env.step(0)
    assert (terminated, truncated) == (True, False)
    assert reward == pytest.approx(0.9625)
    assert info["metrics"] == recovery_metrics(env.record)
    return env.record


@pytest.mark.parametrize(
    "kwargs",
    [
        {"notification_delay_min": -1},
        {"notification_delay_min": True},
        {"rebooking_delay_min": 1.5},
        {"max_rebookings": -1},
    ],
)
def test_recovery_config_requires_nonnegative_integers(kwargs) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        RecoveryConfig(**kwargs)


def test_recovery_preserves_budgets_and_global_observation() -> None:
    record = successful_recovery_record()

    assert record.termination_reason == "arrived"
    assert record.final_airport == "BBB"
    assert record.clock_utc == 1_300
    assert len(record.segments) == 2
    assert record.segments[0].request == record.request
    assert record.segments[1].request == TripRequest(
        "AAA",
        "BBB",
        1_110,
        1_600,
        1_800,
        max_attempts=2,
        min_connection_min=30,
        delay_budget_min=600,
    )
    metrics = recovery_metrics(record)
    assert metrics == {
        "validity": True,
        "errors": [],
        "arrived": True,
        "on_time_arrival": True,
        "elapsed_min": 300,
        "disrupted": True,
        "attempts": 2,
        "rebookings": 1,
        "cancellation_segments": 1,
        "failed": False,
        "termination_reason": "arrived",
    }
    assert verify_recovery(record).aggregate_score == pytest.approx(0.9625)


def test_restart_at_horizon_is_disallowed_without_fabricating_clock_movement() -> None:
    cancelled_flight = flight("deadline", "AAA", "BBB", 1_500, 1_620)
    data = StubData((cancelled_flight,), {"deadline": cancelled()})
    env = RecoveryEnv(request(), data, RecoveryConfig(150, 150, 2))
    env.reset(seed=8)

    observation, reward, terminated, truncated, _ = env.step(0)

    assert (reward, terminated, truncated) == (0.0, True, False)
    assert env.record.termination_reason == "recovery_horizon_exhausted"
    assert env.record.clock_utc == 1_500
    assert env.record.final_airport == "AAA"
    assert env.record.events[0].restart_ready_utc == 1_800
    assert observation["time"].tolist() == [500.0, 100.0, 300.0, 2.0]
    assert recovery_metrics(env.record)["validity"] is True


def test_attempt_exhaustion_precedes_rebooking_limit() -> None:
    cancelled_flight = flight("last-attempt", "AAA", "BBB", 1_100, 1_220)
    data = StubData((cancelled_flight,), {"last-attempt": cancelled()})
    env = RecoveryEnv(request(attempts=1), data, RecoveryConfig(0, 0, 2))
    env.reset(seed=9)

    env.step(0)

    assert env.record.termination_reason == "recovery_attempts_exhausted"
    assert env.record.events[0].attempts_remaining == 0
    assert recovery_metrics(env.record)["validity"] is True


@pytest.mark.parametrize("outcome", [ordinary(), cancelled()])
def test_zero_rebookings_matches_the_core_episode_exactly(outcome: SampledOutcome) -> None:
    selected = flight("same", "AAA", "BBB", 1_100, 1_250)
    core = FlightRouteEnv(request(), StubData((selected,), {"same": outcome}), max_candidates=2)
    recovery = RecoveryEnv(
        request(),
        StubData((selected,), {"same": outcome}),
        RecoveryConfig(max_rebookings=0),
        max_candidates=2,
    )

    core_observation, _ = core.reset(seed=44)
    recovery_observation, _ = recovery.reset(seed=44)
    assert_observations_equal(core_observation, recovery_observation)
    core_result = core.step(0)
    recovery_result = recovery.step(0)

    assert_observations_equal(core_result[0], recovery_result[0])
    assert core_result[1:4] == recovery_result[1:4]
    assert recovery.record.segments == (core.record,)
    core_metrics = core_result[4]["metrics"]
    outer_metrics = recovery_result[4]["metrics"]
    for name in ("validity", "arrived", "on_time_arrival", "elapsed_min", "disrupted"):
        assert outer_metrics[name] == core_metrics[name]
    assert verify_recovery(recovery.record).aggregate_score == pytest.approx(
        core_result[4]["verification"].aggregate_score
    )
    if outcome.cancelled:
        assert recovery.record.termination_reason == "recovery_rebooking_limit"
        assert core.record.termination_reason == "cancelled"
    else:
        assert recovery.record.termination_reason == core.record.termination_reason == "arrived"


def test_partial_movement_evidence_makes_cancellation_ineligible() -> None:
    selected = flight("ambiguous", "AAA", "BBB", 1_100, 1_250)
    ambiguous = replace(cancelled(), dep_delay_min=0.0)
    env = RecoveryEnv(
        request(), StubData((selected,), {"ambiguous": ambiguous}), RecoveryConfig(0, 0, 2)
    )
    env.reset(seed=3)

    env.step(0)

    assert env.record.termination_reason == "recovery_ineligible_cancellation"
    assert env.record.events[0].decision == "recovery_ineligible_cancellation"
    assert recovery_metrics(env.record)["validity"] is True


def test_zero_connection_recovery_excludes_cancelled_id_and_refills_cap() -> None:
    cancelled_flight = flight("same-time", "AAA", "BBB", 1_100, 1_220)
    replacement = flight("replacement", "AAA", "BBB", 1_100, 1_240)
    data = StubData(
        (cancelled_flight, replacement),
        {"same-time": cancelled(), "replacement": ordinary()},
    )
    env = RecoveryEnv(request(connection=0), data, RecoveryConfig(0, 0, 2), max_candidates=1)
    env.reset(seed=12)

    observation, reward, terminated, truncated, _ = env.step(0)

    assert (reward, terminated, truncated) == (0.0, False, False)
    assert tuple(item.flight_id for item in env.available_flights) == ("replacement",)
    assert observation["action_mask"].tolist() == [1]
    assert data.candidate_calls == [
        ("AAA", 1_000, 1_800, 1),
        ("AAA", 1_100, 1_800, 1),
        ("AAA", 1_100, 1_800, 2),
    ]
    env.step(0)
    assert recovery_metrics(env.record)["validity"] is True


def test_unresolved_diversion_never_creates_a_recovery_boundary() -> None:
    selected = flight("diversion", "AAA", "BBB", 1_100, 1_250)
    outcome = SampledOutcome(
        "diverted",
        diverted=True,
        dep_delay_min=0.0,
        arr_delay_min=None,
        div_reached_dest=False,
        div_actual_elapsed_min=90.0,
        div_airport="CCC",
    )
    env = RecoveryEnv(
        request(), StubData((selected,), {"diversion": outcome}), RecoveryConfig(0, 0, 2)
    )
    env.reset(seed=5)

    env.step(0)

    assert env.record.termination_reason == "unresolved_diversion"
    assert env.record.events == ()
    assert recovery_metrics(env.record)["validity"] is True


def test_verifier_rejects_tampered_core_segment() -> None:
    record = successful_recovery_record()
    changed_segment = replace(record.segments[-1], clock_utc=record.clock_utc + 1)
    tampered = replace(record, segments=(*record.segments[:-1], changed_segment))

    assert verify_recovery(tampered).aggregate_score == 0.0
    metrics = recovery_metrics(tampered)
    assert metrics["validity"] is False
    assert metrics["errors"] == ["segment_1_invalid"]


def test_verifier_rejects_tampered_boundary_event() -> None:
    record = successful_recovery_record()
    changed_event = replace(
        record.events[0], restart_ready_utc=record.events[0].restart_ready_utc + 1
    )
    tampered = replace(record, events=(changed_event,))

    assert verify_recovery(tampered).aggregate_score == 0.0
    metrics = recovery_metrics(tampered)
    assert metrics["validity"] is False
    assert metrics["errors"] == ["segment_0_event"]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("attempts_used", True), ("cancellation_utc", True)],
)
def test_verifier_rejects_bool_event_integers(field_name: str, value: object) -> None:
    record = successful_recovery_record()
    changed_event = replace(record.events[0], **{field_name: value})
    tampered = replace(record, events=(changed_event,))

    assert verify_recovery(tampered).aggregate_score == 0.0
    assert recovery_metrics(tampered)["errors"] == ["events"]


def test_verifier_rejects_repeated_flight_id_across_recovery_segments() -> None:
    record = successful_recovery_record()
    final_segment = record.segments[1]
    repeated_flight = replace(final_segment.legs[0].flight, flight_id="cancel")
    repeated_leg = replace(final_segment.legs[0], flight=repeated_flight)
    changed_segment = replace(final_segment, legs=(repeated_leg,))
    tampered = replace(record, segments=(record.segments[0], changed_segment))

    assert verify_recovery(tampered).aggregate_score == 0.0
    assert recovery_metrics(tampered)["errors"] == ["repeated_flight_id"]


def test_verifier_rejects_attempt_budget_reset_between_segments() -> None:
    record = successful_recovery_record()
    reset_request = replace(record.segments[1].request, max_attempts=record.request.max_attempts)
    changed_segment = replace(record.segments[1], request=reset_request)
    tampered = replace(record, segments=(record.segments[0], changed_segment))

    assert verify_recovery(tampered).aggregate_score == 0.0
    assert recovery_metrics(tampered)["errors"] == ["segment_1_request"]


def test_evaluator_counts_valid_zero_score_failures_as_valid() -> None:
    selected = flight("cancel", "AAA", "BBB", 1_100, 1_250)
    data = StubData((selected,), {"cancel": cancelled()})

    result = evaluate_recovery_policy(
        lambda _scenario_seed: RecoveryEnv(
            request(), data, RecoveryConfig(max_rebookings=0), max_candidates=2
        ),
        lambda _env, _policy_seed: NonstopFirstPolicy(),
        episodes=3,
        seed=77,
        trace_count=1,
    )

    assert result["invalid_records"] == 0
    assert result["failures"] == 3
    assert result["score_profile"] == "deadline_first_v1"
    assert result["mean_score"] == 0.0
    assert result["mean_legacy_six_v1"] == 0.0
    assert result["total_attempts"] == 3
    assert result["total_rebookings"] == 0
    assert result["event_decision_counts"] == {"recovery_rebooking_limit": 3}
    assert [row["scenario_seed"] for row in result["episode_outcomes"]] == [77, 78, 79]


def test_cancellation_exposure_changes_successful_recovery_score_only():
    from flight_rl.verifier import verify_legacy_six_v1

    record = successful_recovery_record()
    recovery = verify_recovery_legacy_six_v1(record)
    clean = verify_legacy_six_v1(replace(record.segments[-1], request=record.request))
    recovery_raw = {item.name: item.raw_score for item in recovery.breakdown}
    clean_raw = {item.name: item.raw_score for item in clean.breakdown}
    assert recovery_raw.pop("cancellation_exposure") == 0
    assert clean_raw.pop("cancellation_exposure") == 1
    assert recovery_raw == clean_raw
    # The cancelled attempt is not counted as a flown connection.
    assert recovery_raw["connections_count"] == 1
    assert clean.aggregate_score - recovery.aggregate_score == pytest.approx(0.05)


def test_recovery_primary_uses_original_request_and_global_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flight_rl.recovery as recovery_module

    record = successful_recovery_record()
    original = recovery_module.validated_deadline_first_scores
    calls: list[tuple[TripRequest, bool, int]] = []

    def record_call(
        trip_request: TripRequest,
        *,
        arrived: bool,
        clock_utc: int,
    ) -> dict[str, float]:
        calls.append((trip_request, arrived, clock_utc))
        return original(trip_request, arrived=arrived, clock_utc=clock_utc)

    monkeypatch.setattr(recovery_module, "validated_deadline_first_scores", record_call)

    result = verify_recovery(record)
    assert calls == [(record.request, True, record.clock_utc)]
    assert result.aggregate_score == pytest.approx(0.9625)


@pytest.mark.parametrize("restart", [1600, 1640])
@pytest.mark.parametrize(
    "reward_mode", ["deadline_first", "legacy_six_v1", "rubric", "on_time_arrival"]
)
def test_recovery_can_finish_after_expired_deadline_until_horizon(restart, reward_mode):
    flights = (
        flight("cancel", "AAA", "BBB", 1500, 1620),
        flight("late-replacement", "AAA", "BBB", 1690, 1750),
    )
    data = StubData(flights, {"cancel": cancelled(), "late-replacement": ordinary()})
    env = RecoveryEnv(
        request(), data, RecoveryConfig(50, restart - 1550, 2), reward_mode=reward_mode
    )
    observation, _ = env.reset(seed=8)
    observation, reward, terminated, _, _ = env.step(0)
    assert not terminated
    assert reward == 0
    assert observation["time"][1] == 1600 - restart
    assert env.observation_space.contains(observation)
    observation, reward, terminated, _, info = env.step(0)
    assert terminated
    assert info["metrics"]["validity"]
    assert info["metrics"]["arrived"]
    assert not info["metrics"]["on_time_arrival"]
    assert env.record.segments[-1].request.deadline_utc == 1600
    assert env.record.clock_utc == 1750
    assert verify_recovery(env.record).aggregate_score == pytest.approx(0.10625)
    expected_reward = {
        "deadline_first": 0.10625,
        "legacy_six_v1": 0.70,
        "rubric": 0.70,
        "on_time_arrival": 0.0,
    }[reward_mode]
    assert reward == pytest.approx(expected_reward)
    assert env.observation_space.contains(observation)
    env.close()
