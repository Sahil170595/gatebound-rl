from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from flight_rl.env import FlightRouteEnv
from flight_rl.models import (
    CANDIDATE_FEATURES,
    FlightCandidate,
    OutcomeSummary,
    SampledOutcome,
    TripRequest,
    utc_minutes,
)


def make_flight(
    flight_id: str,
    origin: str,
    dest: str,
    departure: int,
    arrival: int,
    *,
    carrier: str = "AA",
) -> FlightCandidate:
    return FlightCandidate(
        flight_id=flight_id,
        origin=origin,
        dest=dest,
        carrier=carrier,
        scheduled_departure_utc=departure,
        scheduled_arrival_utc=arrival,
        flight_date="2026-01-01",
        dep_hour_bucket=2,
        season="DJF",
        operating_carrier=carrier,
        operating_flight_number=flight_id,
    )


def normal_outcome(donor_id: str = "normal", *, dep: float = 0, arr: float = 0) -> SampledOutcome:
    return SampledOutcome(
        donor_id=donor_id,
        dep_delay_min=dep,
        arr_delay_min=arr,
        actual_elapsed_min=None,
        support=50,
        fallback_level="route",
    )


@dataclass
class StubData:
    flights: tuple[FlightCandidate, ...]
    outcomes: dict[str, tuple[SampledOutcome, ...]]
    summaries: dict[str, OutcomeSummary] = field(default_factory=dict)
    airports: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD")
    carriers: tuple[str, ...] = ("AA", "BB")
    sample_calls: int = 0
    candidate_calls: list[tuple[str, int, int, int]] = field(default_factory=list)

    def candidates(
        self, origin: str, earliest_utc: int, horizon_utc: int, limit: int = 64
    ) -> tuple[FlightCandidate, ...]:
        self.candidate_calls.append((origin, earliest_utc, horizon_utc, limit))
        # Deliberately return every matching-origin row. The environment must enforce
        # its own boarding window, deterministic ordering, horizon, and cap.
        return tuple(flight for flight in self.flights if flight.origin == origin)

    def sample_outcome(self, flight: FlightCandidate, rng: np.random.Generator) -> SampledOutcome:
        self.sample_calls += 1
        pool = self.outcomes[flight.flight_id]
        return pool[int(rng.integers(len(pool)))]

    def outcome_summary(self, flight: FlightCandidate) -> OutcomeSummary:
        return self.summaries.get(
            flight.flight_id,
            OutcomeSummary(
                p_cancelled=0.1,
                p_diverted=0.05,
                mean_arrival_delay_min=7.5,
                support=50,
                fallback_level="route",
            ),
        )

    def historical_outcomes(self, flight: FlightCandidate) -> tuple[SampledOutcome, ...]:
        return self.outcomes[flight.flight_id]


def make_request(
    *,
    origin: str = "AAA",
    destination: str = "BBB",
    ready: int = 1_000,
    deadline: int = 1_300,
    horizon: int = 1_600,
    attempts: int = 3,
    connection: int = 30,
    delay_budget: int = 1_000,
) -> TripRequest:
    return TripRequest(
        origin=origin,
        destination=destination,
        ready_utc=ready,
        deadline_utc=deadline,
        horizon_utc=horizon,
        max_attempts=attempts,
        min_connection_min=connection,
        delay_budget_min=delay_budget,
    )


def run_one(env: FlightRouteEnv, *, seed: int = 7, action: int = 0):
    env.reset(seed=seed)
    return env.step(action)


def test_reset_observation_is_numeric_masked_sorted_capped_and_outcome_blind() -> None:
    too_soon = make_flight("soon", "AAA", "BBB", 1_029, 1_150)
    too_late = make_flight("late", "AAA", "BBB", 1_601, 1_700)
    second = make_flight("second", "AAA", "CCC", 1_100, 1_240, carrier="BB")
    first = make_flight("first", "AAA", "BBB", 1_060, 1_200)
    ignored_by_cap = make_flight("third", "AAA", "DDD", 1_200, 1_300)
    outcomes = {
        flight.flight_id: (normal_outcome(flight.flight_id),)
        for flight in (too_soon, too_late, second, first, ignored_by_cap)
    }
    summaries = {
        "first": OutcomeSummary(0.2, 0.1, 11.5, 80, "fit-only"),
    }
    data = StubData(
        flights=(too_late, second, ignored_by_cap, first, too_soon),
        outcomes=outcomes,
        summaries=summaries,
    )
    env = FlightRouteEnv(make_request(), data, max_candidates=2)

    obs, info = env.reset(seed=123)

    assert env.observation_space.contains(obs)
    assert tuple(flight.flight_id for flight in env.available_flights) == ("first", "second")
    assert obs["action_mask"].tolist() == [1, 1]
    assert obs["time"].tolist() == [0.0, 300.0, 600.0, 3.0]
    assert obs["current_airport"] == 0
    assert obs["destination"] == 1
    assert obs["disrupted"] == 0
    assert obs["candidates"].dtype == np.float32
    assert obs["candidates"].shape == (2, len(CANDIDATE_FEATURES))
    assert obs["candidates"][0].tolist() == pytest.approx([1, 0, 60, 200, 140, 0.2, 0.1, 11.5, 80])
    assert data.sample_calls == 0
    assert data.candidate_calls == [("AAA", 1_030, 1_600, 2)]
    assert info["episode"].termination_reason == "in_progress"

    changed_future = StubData(
        flights=data.flights,
        outcomes={key: (SampledOutcome(key, cancelled=True),) for key in outcomes},
        summaries=summaries,
    )
    changed_obs, _ = FlightRouteEnv(make_request(), changed_future, max_candidates=2).reset(
        seed=999
    )
    for key in obs:
        np.testing.assert_array_equal(obs[key], changed_obs[key])


def test_success_uses_realized_utc_times_and_terminal_verifier_reward() -> None:
    flight = make_flight("direct", "AAA", "BBB", 1_050, 1_200)
    data = StubData(
        flights=(flight,),
        outcomes={"direct": (normal_outcome(dep=5.4, arr=10.6),)},
    )
    env = FlightRouteEnv(make_request(), data, max_candidates=4)

    obs, reward, terminated, truncated, info = run_one(env)

    assert terminated is True and truncated is False
    assert env.record.termination_reason == "arrived"
    assert env.clock_utc == 1_211
    assert env.record.final_airport == "BBB"
    assert env.record.legs[0].departure_utc == 1_055
    assert env.record.legs[0].arrival_utc == 1_211
    assert obs["time"].tolist() == [211.0, 89.0, 389.0, 2.0]
    assert obs["action_mask"].tolist() == [0, 0, 0, 0]
    assert info["metrics"]["arrived"] is True
    assert info["metrics"]["on_time_arrival"] is True
    assert info["metrics"]["validity"] is True
    assert reward == pytest.approx(info["verification"].aggregate_score)
    assert 0.0 < reward <= 1.0


def test_cancellation_precedes_diversion_and_missing_timing() -> None:
    flight = make_flight("cancel", "AAA", "BBB", 1_080, 1_200)
    outcome = SampledOutcome(
        "cancelled-donor",
        cancelled=True,
        diverted=True,
        dep_delay_min=None,
        arr_delay_min=None,
    )
    env = FlightRouteEnv(
        make_request(), StubData((flight,), {"cancel": (outcome,)}), max_candidates=2
    )

    obs, reward, terminated, truncated, info = run_one(env)

    assert (terminated, truncated, reward) == (True, False, 0.0)
    assert env.record.termination_reason == "cancelled"
    assert env.clock_utc == flight.scheduled_departure_utc
    assert env.record.final_airport == "AAA"
    assert env.record.legs[0].departure_utc is None
    assert env.record.legs[0].arrival_utc is None
    assert env.record.legs[0].resolved_airport == "AAA"
    assert obs["disrupted"] == 1
    assert info["metrics"]["disrupted"] is True


@pytest.mark.parametrize(
    ("outcome", "expected_arrival"),
    [
        (
            SampledOutcome(
                "div-delay",
                diverted=True,
                dep_delay_min=10,
                div_reached_dest=True,
                div_arr_delay_min=25,
                div_actual_elapsed_min=999,
            ),
            1_225,
        ),
        (
            SampledOutcome(
                "div-elapsed",
                diverted=True,
                dep_delay_min=10,
                div_reached_dest=True,
                div_arr_delay_min=None,
                div_actual_elapsed_min=180,
            ),
            1_240,
        ),
    ],
)
def test_destination_reaching_diversion_uses_documented_timing_precedence(
    outcome: SampledOutcome, expected_arrival: int
) -> None:
    flight = make_flight("diverted", "AAA", "BBB", 1_050, 1_200)
    env = FlightRouteEnv(make_request(), StubData((flight,), {"diverted": (outcome,)}))

    obs, _, terminated, truncated, info = run_one(env)

    assert terminated is True and truncated is False
    assert env.record.termination_reason == "arrived"
    assert env.record.legs[0].departure_utc == 1_060
    assert env.record.legs[0].arrival_utc == expected_arrival
    assert env.record.legs[0].resolved_airport == "BBB"
    assert obs["disrupted"] == 1
    assert info["metrics"]["validity"] is True


def test_unresolved_diversion_preserves_donor_detail_without_claiming_movement() -> None:
    flight = make_flight("diverted", "AAA", "BBB", 1_050, 1_200)
    outcome = SampledOutcome(
        "unresolved",
        diverted=True,
        dep_delay_min=5,
        div_reached_dest=False,
        div_arr_delay_min=None,
        div_actual_elapsed_min=90,
        div_airport="CCC",
    )
    env = FlightRouteEnv(make_request(), StubData((flight,), {"diverted": (outcome,)}))

    _, reward, terminated, truncated, _ = run_one(env)

    assert (terminated, truncated, reward) == (True, False, 0.0)
    assert env.record.termination_reason == "unresolved_diversion"
    assert env.record.final_airport == "AAA"
    assert env.record.legs[0].departure_utc == 1_055
    assert env.record.legs[0].arrival_utc == 1_145
    assert env.record.legs[0].resolved_airport == "CCC"
    assert env.clock_utc == 1_145


@pytest.mark.parametrize(
    "outcome",
    [
        SampledOutcome("missing-dep", dep_delay_min=None, arr_delay_min=0),
        SampledOutcome("missing-arr", dep_delay_min=0, arr_delay_min=None),
        SampledOutcome("nan", dep_delay_min=float("nan"), arr_delay_min=0),
        SampledOutcome("string", dep_delay_min="0", arr_delay_min=0),  # type: ignore[arg-type]
        SampledOutcome(
            "missing-div-arrival",
            diverted=True,
            dep_delay_min=0,
            div_reached_dest=True,
            div_arr_delay_min=None,
            div_actual_elapsed_min=None,
        ),
    ],
)
def test_missing_or_invalid_required_timing_terminates_invalid_outcome(
    outcome: SampledOutcome,
) -> None:
    flight = make_flight("unknown", "AAA", "BBB", 1_050, 1_200)
    env = FlightRouteEnv(make_request(), StubData((flight,), {"unknown": (outcome,)}))

    _, reward, terminated, truncated, _ = run_one(env)

    assert (terminated, truncated, reward) == (True, False, 0.0)
    assert env.record.termination_reason == "invalid_outcome"
    assert env.record.final_airport == "AAA"
    assert env.record.legs[0].outcome is outcome
    assert env.record.legs[0].departure_utc is None
    assert env.record.legs[0].arrival_utc is None
    assert env.clock_utc == flight.scheduled_departure_utc


@pytest.mark.parametrize(("arr_delay", "expected_arrival"), [(-100, 1_100), (-50, 1_150)])
def test_nonpositive_realized_duration_is_invalid_outcome(
    arr_delay: float, expected_arrival: int
) -> None:
    flight = make_flight("impossible", "AAA", "BBB", 1_050, 1_200)
    outcome = normal_outcome(dep=100, arr=arr_delay)
    env = FlightRouteEnv(make_request(), StubData((flight,), {"impossible": (outcome,)}))

    run_one(env)

    assert env.record.termination_reason == "invalid_outcome"
    assert env.record.legs[0].departure_utc == 1_150
    assert env.record.legs[0].arrival_utc == expected_arrival
    assert env.clock_utc == 1_150
    assert env.record.final_airport == "AAA"


def test_realized_departure_before_prior_clock_is_missed() -> None:
    flight = make_flight("early", "AAA", "BBB", 1_050, 1_200)
    outcome = normal_outcome(dep=-60, arr=0)
    env = FlightRouteEnv(make_request(), StubData((flight,), {"early": (outcome,)}))

    _, reward, _, _, _ = run_one(env)

    assert reward == 0.0
    assert env.record.termination_reason == "missed_departure"
    assert env.record.legs[0].departure_utc == 990
    assert env.record.legs[0].arrival_utc == 1_200
    assert env.clock_utc == 1_000
    assert env.record.final_airport == "AAA"


def test_boarding_and_connection_buffers_use_absolute_utc_across_dst() -> None:
    # US clocks jump locally during this UTC interval, but the environment only
    # performs absolute UTC-minute arithmetic.
    ready = utc_minutes("2025-03-09T06:30:00+00:00")
    first_departure = utc_minutes("2025-03-09T07:15:00+00:00")
    first_arrival = utc_minutes("2025-03-09T08:00:00+00:00")
    tight_connection = make_flight("tight", "CCC", "BBB", first_arrival + 44, first_arrival + 120)
    valid_connection = make_flight("valid", "CCC", "BBB", first_arrival + 45, first_arrival + 130)
    first = make_flight("first", "AAA", "CCC", first_departure, first_arrival)
    request = make_request(
        ready=ready,
        deadline=first_arrival + 180,
        horizon=first_arrival + 300,
        connection=45,
    )
    data = StubData(
        flights=(tight_connection, valid_connection, first),
        outcomes={
            "first": (normal_outcome("first"),),
            "tight": (normal_outcome("tight"),),
            "valid": (normal_outcome("valid"),),
        },
    )
    env = FlightRouteEnv(request, data)

    obs, _, terminated, _, _ = run_one(env)

    assert terminated is False
    assert env.clock_utc == first_arrival
    assert tuple(flight.flight_id for flight in env.available_flights) == ("valid",)
    assert data.candidate_calls == [
        ("AAA", ready + 45, request.horizon_utc, 64),
        ("CCC", first_arrival + 45, request.horizon_utc, 64),
    ]
    assert obs["time"][0] == pytest.approx(90.0)


@pytest.mark.parametrize(
    ("arrival", "destination", "expected_reason", "expected_final"),
    [
        (1_601, "BBB", "horizon", "AAA"),
        (1_600, "BBB", "arrived", "BBB"),
        (1_600, "CCC", "horizon", "CCC"),
    ],
)
def test_horizon_is_intrinsic_inclusive_only_for_destination_success(
    arrival: int, destination: str, expected_reason: str, expected_final: str
) -> None:
    flight = make_flight("boundary", "AAA", destination, 1_400, 1_500)
    outcome = normal_outcome(arr=arrival - 1_500)
    env = FlightRouteEnv(make_request(), StubData((flight,), {"boundary": (outcome,)}))

    obs, reward, terminated, truncated, _ = run_one(env)

    assert terminated is True and truncated is False
    assert env.record.termination_reason == expected_reason
    assert env.record.final_airport == expected_final
    assert env.clock_utc == 1_600
    assert env.record.legs[0].arrival_utc == arrival
    assert obs["time"][2] == 0
    assert (reward > 0) is (expected_reason == "arrived")


def test_max_attempts_wins_after_non_destination_arrival() -> None:
    flight = make_flight("one-shot", "AAA", "CCC", 1_050, 1_200)
    onward = make_flight("onward", "CCC", "BBB", 1_300, 1_400)
    data = StubData(
        (flight, onward),
        {"one-shot": (normal_outcome(),), "onward": (normal_outcome(),)},
    )
    env = FlightRouteEnv(make_request(attempts=1), data)

    _, reward, terminated, truncated, _ = run_one(env)

    assert (terminated, truncated, reward) == (True, False, 0.0)
    assert env.record.termination_reason == "max_attempts"
    assert env.record.final_airport == "CCC"
    assert len(env.record.legs) == 1


def test_two_leg_arrival_preserves_continuity_and_is_verifier_valid() -> None:
    first = make_flight("first", "AAA", "CCC", 1_050, 1_150)
    second = make_flight("second", "CCC", "BBB", 1_180, 1_280)
    data = StubData(
        (second, first),
        {"first": (normal_outcome("first"),), "second": (normal_outcome("second"),)},
    )
    env = FlightRouteEnv(make_request(attempts=2), data)

    env.reset(seed=31)
    _, first_reward, first_terminated, first_truncated, _ = env.step(0)
    assert (first_terminated, first_truncated, first_reward) == (False, False, 0.0)
    assert tuple(flight.flight_id for flight in env.available_flights) == ("second",)

    _, reward, terminated, truncated, info = env.step(0)
    assert terminated is True and truncated is False
    assert reward > 0.0
    assert env.record.termination_reason == "arrived"
    assert env.record.final_airport == "BBB"
    assert env.clock_utc == 1_280
    assert len(env.record.legs) == 2
    assert info["metrics"]["validity"] is True


def test_no_candidates_at_reset_and_after_intermediate_arrival() -> None:
    empty = FlightRouteEnv(make_request(), StubData((), {}), max_candidates=3)
    obs, _ = empty.reset(seed=1)
    assert obs["action_mask"].tolist() == [0, 0, 0]
    assert empty.available_flights == ()

    _, reward, terminated, truncated, _ = empty.step(0)
    assert (terminated, truncated, reward) == (True, False, 0.0)
    assert empty.record.termination_reason == "no_candidates"

    first = make_flight("first", "AAA", "CCC", 1_050, 1_200)
    data = StubData((first,), {"first": (normal_outcome(),)})
    onward_empty = FlightRouteEnv(make_request(), data)
    _, reward, terminated, truncated, _ = run_one(onward_empty)
    assert (terminated, truncated, reward) == (True, False, 0.0)
    assert onward_empty.record.termination_reason == "no_candidates"
    assert onward_empty.record.final_airport == "CCC"


def test_invalid_actions_and_step_after_done_have_distinct_behavior() -> None:
    flight = make_flight("only", "AAA", "BBB", 1_050, 1_200)
    data = StubData((flight,), {"only": (normal_outcome(),)})
    env = FlightRouteEnv(make_request(), data, max_candidates=3)
    env.reset(seed=2)

    for action in (-1, 3, 1.5, True, np.array(0)):
        with pytest.raises(ValueError):
            env.step(action)  # type: ignore[arg-type]
    assert data.sample_calls == 0

    _, reward, terminated, truncated, _ = env.step(2)
    assert (terminated, truncated, reward) == (True, False, 0.0)
    assert env.record.termination_reason == "invalid_action"
    assert data.sample_calls == 0
    with pytest.raises(RuntimeError):
        env.step(0)


def test_seeded_sampling_is_repeatable() -> None:
    flight = make_flight("variable", "AAA", "BBB", 1_050, 1_200)
    pool = tuple(normal_outcome(f"donor-{index}", arr=index * 5) for index in range(20))
    data = StubData((flight,), {"variable": pool})
    env = FlightRouteEnv(make_request(), data)

    first = run_one(env, seed=8675309)
    first_record = env.record
    second = run_one(env, seed=8675309)

    assert env.record == first_record
    assert second[1] == pytest.approx(first[1])
    np.testing.assert_array_equal(second[0]["time"], first[0]["time"])


def test_reward_modes_have_identical_dynamics_and_separate_objectives() -> None:
    flight = make_flight("late", "AAA", "BBB", 1_050, 1_200)
    outcome = normal_outcome("after-deadline", arr=150)
    data_a = StubData((flight,), {"late": (outcome,)})
    data_b = StubData((flight,), {"late": (outcome,)})
    data_c = StubData((flight,), {"late": (outcome,)})
    legacy = FlightRouteEnv(make_request(), data_a, reward_mode="rubric")
    deadline = FlightRouteEnv(make_request(), data_b, reward_mode="on_time_arrival")
    deadline_first = FlightRouteEnv(make_request(), data_c, reward_mode="deadline_first")

    result_a = run_one(legacy, seed=22)
    result_b = run_one(deadline, seed=22)
    result_c = run_one(deadline_first, seed=22)

    assert legacy.record == deadline.record == deadline_first.record
    assert legacy.record.termination_reason == "arrived"
    assert result_a[1] > 0.0
    assert result_b[1] == 0.0
    assert 0.1 < result_c[1] < 0.2
    assert result_a[4]["metrics"] == result_b[4]["metrics"] == result_c[4]["metrics"]
    assert result_a[4]["verification"] == result_c[4]["verification"]
    assert result_a[4]["reward_mode"] == "legacy_six_v1"
    assert result_b[4]["reward_mode"] == "on_time_arrival"
    assert result_c[4]["reward_mode"] == "deadline_first"
    assert result_a[1] == pytest.approx(result_a[4]["legacy_six_v1_verification"].aggregate_score)
    assert result_c[1] == pytest.approx(result_c[4]["verification"].aggregate_score)


def test_default_reward_mode_is_primary_deadline_first() -> None:
    flight = make_flight("default", "AAA", "BBB", 1_050, 1_200)
    env = FlightRouteEnv(make_request(), StubData((flight,), {"default": (normal_outcome(),)}))

    result = run_one(env, seed=22)

    assert env.reward_mode == "deadline_first"
    assert result[4]["score_profile"] == "deadline_first_v1"
    assert result[1] == pytest.approx(result[4]["verification"].aggregate_score)


def test_gymnasium_check_env_and_public_state_is_read_only() -> None:
    flight = make_flight("check", "AAA", "BBB", 1_050, 1_200)
    env = FlightRouteEnv(
        make_request(), StubData((flight,), {"check": (normal_outcome(),)}), max_candidates=1
    )

    check_env(env, skip_render_check=True)

    with pytest.raises(AttributeError):
        env.clock_utc = 42  # type: ignore[misc]
    with pytest.raises(AttributeError):
        env.available_flights = ()  # type: ignore[misc]
    with pytest.raises(AttributeError):
        env.record = env.record  # type: ignore[misc]
    with pytest.raises(AttributeError):
        env.reward_mode = "on_time_arrival"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_candidates": 0}, "positive integer"),
        ({"max_candidates": True}, "positive integer"),
        ({"reward_mode": "unknown"}, "reward_mode"),
        ({"render_mode": "rgb_array"}, "render_mode"),
    ],
)
def test_invalid_environment_configuration_raises_value_error(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        FlightRouteEnv(make_request(), StubData((), {}), **kwargs)  # type: ignore[arg-type]


def test_candidate_summary_rejects_nonfinite_or_invalid_values() -> None:
    flight = make_flight("bad-summary", "AAA", "BBB", 1_050, 1_200)
    summaries: Iterable[OutcomeSummary] = (
        OutcomeSummary(float("nan"), 0, 0, 1, "bad"),
        OutcomeSummary(0, 1.1, 0, 1, "bad"),
        OutcomeSummary(0, 0, 0, -1, "bad"),
    )
    for summary in summaries:
        data = StubData(
            (flight,),
            {"bad-summary": (normal_outcome(),)},
            summaries={"bad-summary": summary},
        )
        with pytest.raises(ValueError):
            FlightRouteEnv(make_request(), data).reset(seed=1)
