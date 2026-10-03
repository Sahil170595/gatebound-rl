"""Cancellation recovery composed from independently verifiable core episodes.

Recovery is a counterfactual simulator assumption.  It does not establish that a
seat was available, authenticate a historical donor, or describe an observed
passenger itinerary.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

from flight_rl.env import FlightRouteEnv, canonical_reward_mode, score_profile_for_reward_mode
from flight_rl.models import EpisodeRecord, FlightDataSource, SampledOutcome, TripRequest
from flight_rl.verifier import (
    DEADLINE_FIRST_V1_WEIGHTS,
    LEGACY_SIX_V1_WEIGHTS,
    PRIMARY_SCORE_PROFILE,
    CriterionResult,
    VerificationResult,
    episode_metrics,
    validated_deadline_first_scores,
    validated_trip_scores,
)

_IN_PROGRESS = "in_progress"
_RECOVERY_DECISIONS = frozenset(
    {
        "rebooked",
        "recovery_ineligible_cancellation",
        "recovery_horizon_exhausted",
        "recovery_attempts_exhausted",
        "recovery_rebooking_limit",
    }
)


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


@dataclass(frozen=True)
class RecoveryConfig:
    """Fixed delay and retry assumptions applied after scheduled cancellation."""

    notification_delay_min: int = 0
    rebooking_delay_min: int = 0
    max_rebookings: int = 0

    def __post_init__(self) -> None:
        for name in (
            "notification_delay_min",
            "rebooking_delay_min",
            "max_rebookings",
        ):
            object.__setattr__(self, name, _nonnegative_integer(getattr(self, name), name))


@dataclass(frozen=True)
class RecoveryEvent:
    """One explicit decision at a terminal core-cancellation boundary."""

    segment_index: int
    cancelled_flight_id: str
    airport: str
    cancellation_utc: int
    notification_ready_utc: int
    restart_ready_utc: int
    attempts_used: int
    attempts_remaining: int
    rebookings_before: int
    decision: str


@dataclass(frozen=True)
class RecoveryRecord:
    """Immutable outer trace retaining every terminal core episode."""

    request: TripRequest
    config: RecoveryConfig
    segments: tuple[EpisodeRecord, ...] = field(default_factory=tuple)
    events: tuple[RecoveryEvent, ...] = field(default_factory=tuple)
    final_airport: str = ""
    clock_utc: int = 0
    termination_reason: str = _IN_PROGRESS


@dataclass(frozen=True)
class _RecoveryAnalysis:
    validity: bool
    errors: tuple[str, ...]
    arrived: bool
    on_time_arrival: bool
    elapsed_min: int | None
    disrupted: bool
    attempts: int
    rebookings: int
    cancellation_segments: int
    termination_reason: str


def _is_integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, Integral)


def _valid_config(config: object) -> bool:
    return isinstance(config, RecoveryConfig) and all(
        _is_integer(getattr(config, name)) and getattr(config, name) >= 0
        for name in (
            "notification_delay_min",
            "rebooking_delay_min",
            "max_rebookings",
        )
    )


def _valid_event(event: object) -> bool:
    if not isinstance(event, RecoveryEvent):
        return False
    integer_fields = (
        event.segment_index,
        event.cancellation_utc,
        event.notification_ready_utc,
        event.restart_ready_utc,
        event.attempts_used,
        event.attempts_remaining,
        event.rebookings_before,
    )
    return bool(
        all(_is_integer(value) for value in integer_fields)
        and event.segment_index >= 0
        and event.attempts_used >= 0
        and event.attempts_remaining >= 0
        and event.rebookings_before >= 0
        and isinstance(event.cancelled_flight_id, str)
        and event.cancelled_flight_id
        and isinstance(event.airport, str)
        and event.airport
        and isinstance(event.decision, str)
        and event.decision in _RECOVERY_DECISIONS
    )


def _has_partial_movement_evidence(outcome: object) -> bool:
    if not isinstance(outcome, SampledOutcome):
        return True
    return bool(
        outcome.diverted
        or outcome.div_reached_dest
        or outcome.dep_delay_min is not None
        or outcome.arr_delay_min is not None
        or outcome.actual_elapsed_min is not None
        or outcome.div_arr_delay_min is not None
        or outcome.div_actual_elapsed_min is not None
        or outcome.div_airport is not None
    )


def _eligible_cancellation(segment: object) -> bool:
    """Require a valid terminal cancellation with no evidence of movement."""

    if not isinstance(segment, EpisodeRecord):
        return False
    metrics = episode_metrics(segment)
    if not metrics["validity"] or segment.termination_reason != "cancelled" or not segment.legs:
        return False
    terminal = segment.legs[-1]
    return bool(
        terminal.outcome.cancelled
        and terminal.departure_utc is None
        and terminal.arrival_utc is None
        and terminal.resolved_airport == segment.final_airport
        and not _has_partial_movement_evidence(terminal.outcome)
    )


def _decision(
    segment: EpisodeRecord,
    config: RecoveryConfig,
    original: TripRequest,
    attempts_remaining: int,
    rebookings_before: int,
) -> str:
    terminal = segment.legs[-1]
    restart = (
        terminal.flight.scheduled_departure_utc
        + config.notification_delay_min
        + config.rebooking_delay_min
    )
    if not _eligible_cancellation(segment):
        return "recovery_ineligible_cancellation"
    if restart >= original.horizon_utc:
        return "recovery_horizon_exhausted"
    if attempts_remaining <= 0:
        return "recovery_attempts_exhausted"
    if rebookings_before >= config.max_rebookings:
        return "recovery_rebooking_limit"
    return "rebooked"


def _expected_event(
    segment: EpisodeRecord,
    segment_index: int,
    config: RecoveryConfig,
    original: TripRequest,
    attempts_used: int,
    rebookings_before: int,
) -> RecoveryEvent:
    terminal = segment.legs[-1]
    cancellation = terminal.flight.scheduled_departure_utc
    notification = cancellation + config.notification_delay_min
    restart = notification + config.rebooking_delay_min
    attempts_remaining = original.max_attempts - attempts_used
    return RecoveryEvent(
        segment_index=segment_index,
        cancelled_flight_id=terminal.flight.flight_id,
        airport=segment.final_airport,
        cancellation_utc=cancellation,
        notification_ready_utc=notification,
        restart_ready_utc=restart,
        attempts_used=attempts_used,
        attempts_remaining=attempts_remaining,
        rebookings_before=rebookings_before,
        decision=_decision(
            segment,
            config,
            original,
            attempts_remaining,
            rebookings_before,
        ),
    )


def _next_request(original: TripRequest, event: RecoveryEvent) -> TripRequest:
    return TripRequest(
        origin=event.airport,
        destination=original.destination,
        ready_utc=event.restart_ready_utc,
        deadline_utc=original.deadline_utc,
        horizon_utc=original.horizon_utc,
        max_attempts=event.attempts_remaining,
        min_connection_min=original.min_connection_min,
        delay_budget_min=original.delay_budget_min,
    )


def _invalid_analysis(record: object, error: str) -> _RecoveryAnalysis:
    reason = (
        record.termination_reason
        if isinstance(record, RecoveryRecord) and isinstance(record.termination_reason, str)
        else "invalid_record"
    )
    attempts = 0
    rebookings = 0
    cancellations = 0
    disrupted = False
    if isinstance(record, RecoveryRecord) and isinstance(record.segments, tuple):
        for segment in record.segments:
            if not isinstance(segment, EpisodeRecord) or not isinstance(segment.legs, tuple):
                continue
            attempts += len(segment.legs)
            for leg in segment.legs:
                outcome = getattr(leg, "outcome", None)
                if isinstance(outcome, SampledOutcome):
                    disrupted = disrupted or outcome.cancelled or outcome.diverted
            cancellations += int(segment.termination_reason == "cancelled")
    if isinstance(record, RecoveryRecord) and isinstance(record.events, tuple):
        rebookings = sum(
            isinstance(event, RecoveryEvent) and event.decision == "rebooked"
            for event in record.events
        )
    return _RecoveryAnalysis(
        validity=False,
        errors=(error,),
        arrived=False,
        on_time_arrival=False,
        elapsed_min=None,
        disrupted=bool(disrupted),
        attempts=attempts,
        rebookings=rebookings,
        cancellation_segments=cancellations,
        termination_reason=reason,
    )


def _analyze_recovery(record: object) -> _RecoveryAnalysis:
    """Validate every core segment and all cross-segment boundary claims."""

    if not isinstance(record, RecoveryRecord):
        return _invalid_analysis(record, "record_type")
    if not isinstance(record.request, TripRequest):
        return _invalid_analysis(record, "request_type")
    if not _valid_config(record.config):
        return _invalid_analysis(record, "config")
    if not isinstance(record.segments, tuple) or not record.segments:
        return _invalid_analysis(record, "segments")
    if not isinstance(record.events, tuple) or any(
        not _valid_event(event) for event in record.events
    ):
        return _invalid_analysis(record, "events")
    if not isinstance(record.final_airport, str) or not record.final_airport:
        return _invalid_analysis(record, "final_airport")
    if not _is_integer(record.clock_utc):
        return _invalid_analysis(record, "clock_utc")
    if not isinstance(record.termination_reason, str) or not record.termination_reason:
        return _invalid_analysis(record, "termination_reason")

    original = record.request
    expected_request = original
    attempts_used = 0
    rebookings = 0
    cancellation_segments = 0
    disrupted = False
    event_cursor = 0
    expected_outer_reason: str | None = None
    attempted_flight_ids: set[str] = set()

    for segment_index, segment in enumerate(record.segments):
        if not isinstance(segment, EpisodeRecord):
            return _invalid_analysis(record, f"segment_{segment_index}_type")
        if segment.request != expected_request:
            return _invalid_analysis(record, f"segment_{segment_index}_request")
        metrics = episode_metrics(segment)
        if not metrics["validity"]:
            return _invalid_analysis(record, f"segment_{segment_index}_invalid")
        for leg in segment.legs:
            if leg.flight.flight_id in attempted_flight_ids:
                return _invalid_analysis(record, "repeated_flight_id")
            attempted_flight_ids.add(leg.flight.flight_id)
        attempts_used += len(segment.legs)
        if attempts_used > original.max_attempts:
            return _invalid_analysis(record, "attempt_budget")
        disrupted = disrupted or bool(metrics["disrupted"])
        is_last = segment_index == len(record.segments) - 1

        if segment.termination_reason != "cancelled":
            if not is_last:
                return _invalid_analysis(record, f"segment_{segment_index}_nonterminal_chain")
            expected_outer_reason = segment.termination_reason
            continue

        cancellation_segments += 1
        if event_cursor >= len(record.events):
            return _invalid_analysis(record, f"segment_{segment_index}_missing_event")
        event = record.events[event_cursor]
        expected = _expected_event(
            segment,
            segment_index,
            record.config,
            original,
            attempts_used,
            rebookings,
        )
        if event != expected:
            return _invalid_analysis(record, f"segment_{segment_index}_event")
        event_cursor += 1

        if event.decision == "rebooked":
            if is_last:
                return _invalid_analysis(record, f"segment_{segment_index}_missing_restart")
            rebookings += 1
            expected_request = _next_request(original, event)
        else:
            if not is_last:
                return _invalid_analysis(record, f"segment_{segment_index}_continued_after_stop")
            expected_outer_reason = event.decision

    if event_cursor != len(record.events):
        return _invalid_analysis(record, "extra_events")
    if expected_outer_reason is None:
        return _invalid_analysis(record, "unfinished_chain")
    final_segment = record.segments[-1]
    if record.final_airport != final_segment.final_airport:
        return _invalid_analysis(record, "outer_final_airport")
    if record.clock_utc != final_segment.clock_utc:
        return _invalid_analysis(record, "outer_clock")
    if record.termination_reason != expected_outer_reason:
        return _invalid_analysis(record, "outer_termination")

    arrived = expected_outer_reason == "arrived" and record.final_airport == original.destination
    elapsed = record.clock_utc - original.ready_utc
    if elapsed < 0:
        return _invalid_analysis(record, "negative_elapsed")
    return _RecoveryAnalysis(
        validity=True,
        errors=(),
        arrived=arrived,
        on_time_arrival=bool(arrived and record.clock_utc <= original.deadline_utc),
        elapsed_min=elapsed,
        disrupted=disrupted,
        attempts=attempts_used,
        rebookings=rebookings,
        cancellation_segments=cancellation_segments,
        termination_reason=expected_outer_reason,
    )


def recovery_metrics(record: RecoveryRecord) -> dict[str, object]:
    """Return validation and outcome counters using the original request budgets."""

    analysis = _analyze_recovery(record)
    return {
        "validity": analysis.validity,
        "errors": list(analysis.errors),
        "arrived": analysis.arrived,
        "on_time_arrival": analysis.on_time_arrival,
        "elapsed_min": analysis.elapsed_min,
        "disrupted": analysis.disrupted,
        "attempts": analysis.attempts,
        "rebookings": analysis.rebookings,
        "cancellation_segments": analysis.cancellation_segments,
        "failed": not analysis.arrived,
        "termination_reason": analysis.termination_reason,
    }


def _result(raw: dict[str, float], weights: dict[str, float]) -> VerificationResult:
    breakdown = tuple(
        CriterionResult(
            name=name,
            raw_score=raw[name],
            weight=weight,
            weighted_contribution=raw[name] * weight,
        )
        for name, weight in weights.items()
    )
    return VerificationResult(
        aggregate_score=math.fsum(item.weighted_contribution for item in breakdown),
        breakdown=breakdown,
    )


def verify_recovery(record: RecoveryRecord) -> VerificationResult:
    """Validate and score the whole journey with deadline-first-v1."""

    analysis = _analyze_recovery(record)
    if analysis.validity:
        assert isinstance(record, RecoveryRecord)
        raw = validated_deadline_first_scores(
            record.request,
            arrived=analysis.arrived,
            clock_utc=record.clock_utc,
        )
    else:
        raw = dict.fromkeys(DEADLINE_FIRST_V1_WEIGHTS, 0.0)
    return _result(raw, dict(DEADLINE_FIRST_V1_WEIGHTS))


def verify_recovery_legacy_six_v1(record: RecoveryRecord) -> VerificationResult:
    """Validate a recovery chain and apply the frozen six-signal profile."""

    analysis = _analyze_recovery(record)
    if analysis.validity:
        assert isinstance(record, RecoveryRecord)
        raw = validated_trip_scores(
            record.request,
            tuple(leg for segment in record.segments for leg in segment.legs),
            arrived=analysis.arrived,
            clock_utc=record.clock_utc,
        )
    else:
        raw = dict.fromkeys(LEGACY_SIX_V1_WEIGHTS, 0.0)
    return _result(raw, dict(LEGACY_SIX_V1_WEIGHTS))


class _RecoveryData:
    """Filter already attempted IDs while refilling the requested candidate cap."""

    def __init__(self, data: FlightDataSource, attempted_flight_ids: set[str]) -> None:
        self._data = data
        self._attempted_flight_ids = attempted_flight_ids
        self.airports = tuple(data.airports)
        self.carriers = tuple(data.carriers)

    def candidates(self, origin: str, earliest_utc: int, horizon_utc: int, limit: int = 64):
        requested = limit
        while True:
            supplied = tuple(self._data.candidates(origin, earliest_utc, horizon_utc, requested))
            retained = [
                flight for flight in supplied if flight.flight_id not in self._attempted_flight_ids
            ]
            if len(retained) >= limit or len(supplied) < requested:
                return tuple(retained[:limit])
            requested *= 2

    def sample_outcome(self, flight, rng):
        return self._data.sample_outcome(flight, rng)

    def outcome_summary(self, flight):
        return self._data.outcome_summary(flight)

    def historical_outcomes(self, flight):
        return self._data.historical_outcomes(flight)


class RecoveryEnv(gym.Env[dict[str, np.ndarray | int], int]):
    """Compose core environments across eligible terminal cancellations."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        request: TripRequest,
        data: FlightDataSource,
        config: RecoveryConfig,
        *,
        max_candidates: int = 64,
        reward_mode: str = "deadline_first",
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, RecoveryConfig):
            raise TypeError("config must be a RecoveryConfig")
        if not _valid_config(config):
            raise ValueError("config contains invalid values")
        reward_mode = canonical_reward_mode(reward_mode)
        self._request = request
        self._data = data
        self._config = config
        self._max_candidates = max_candidates
        self._reward_mode = reward_mode
        self.render_mode = render_mode
        self._attempted_flight_ids: set[str] = set()
        self._filtered_data = _RecoveryData(data, self._attempted_flight_ids)
        self._inner = self._new_inner(request)
        self.observation_space = self._inner.observation_space
        self.action_space = self._inner.action_space
        self._segments: list[EpisodeRecord] = []
        self._events: list[RecoveryEvent] = []
        self._done = False
        self._record = RecoveryRecord(
            request=request,
            config=config,
            final_airport=request.origin,
            clock_utc=request.ready_utc,
        )

    @property
    def request(self) -> TripRequest:
        return self._request

    @property
    def config(self) -> RecoveryConfig:
        return self._config

    @property
    def data(self) -> FlightDataSource:
        return self._data

    @property
    def reward_mode(self) -> str:
        return self._reward_mode

    @property
    def available_flights(self):
        return self._inner.available_flights

    @property
    def clock_utc(self) -> int:
        return self._inner.clock_utc

    @property
    def record(self) -> RecoveryRecord:
        return self._record

    def _new_inner(self, request: TripRequest) -> FlightRouteEnv:
        return FlightRouteEnv(
            request,
            self._filtered_data,
            max_candidates=self._max_candidates,
            reward_mode=self._reward_mode,
            render_mode=None,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray | int], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self._inner.close()
        self._inner = self._new_inner(self.request)
        self._segments = []
        self._events = []
        self._attempted_flight_ids.clear()
        self._done = False
        observation, core_info = self._inner.reset(seed=seed)
        self._record = self._make_record(_IN_PROGRESS)
        return self._global_observation(observation), {
            "recovery": self.record,
            "core": core_info,
        }

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray | int], float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("step() called after episode termination")
        observation, reward, terminated, truncated, core_info = self._inner.step(action)
        self._sync_attempted_flight_ids()
        if not terminated and not truncated:
            self._record = self._make_record(_IN_PROGRESS)
            return (
                self._global_observation(observation),
                float(reward),
                False,
                False,
                {
                    "recovery": self.record,
                    "core": core_info,
                },
            )

        segment = self._inner.record
        self._segments.append(segment)
        if segment.termination_reason == "cancelled":
            attempts_used = sum(len(item.legs) for item in self._segments)
            event = _expected_event(
                segment,
                len(self._segments) - 1,
                self.config,
                self.request,
                attempts_used,
                len(self._events),
            )
            self._events.append(event)
            if event.decision == "rebooked":
                self._inner.close()
                self._inner = self._new_inner(_next_request(self.request, event))
                child_seed = int(self.np_random.integers(0, 2**32, dtype=np.uint32))
                next_observation, next_info = self._inner.reset(seed=child_seed)
                self._record = self._make_record(_IN_PROGRESS)
                return (
                    self._global_observation(next_observation),
                    0.0,
                    False,
                    False,
                    {
                        "recovery": self.record,
                        "core": next_info,
                        "recovery_event": event,
                    },
                )
            return self._finish(observation, event.decision, core_info)

        return self._finish(observation, segment.termination_reason, core_info)

    def _sync_attempted_flight_ids(self) -> None:
        self._attempted_flight_ids.update(leg.flight.flight_id for leg in self._inner.record.legs)

    def _attempts_used(self) -> int:
        complete = sum(len(segment.legs) for segment in self._segments)
        current = 0 if self._done else len(self._inner.record.legs)
        return complete + current

    def _disrupted(self) -> bool:
        segments = list(self._segments)
        if not self._done:
            segments.append(self._inner.record)
        return any(
            leg.outcome.cancelled or leg.outcome.diverted
            for segment in segments
            for leg in segment.legs
        )

    def _global_observation(
        self, observation: dict[str, np.ndarray | int]
    ) -> dict[str, np.ndarray | int]:
        result: dict[str, np.ndarray | int] = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in observation.items()
        }
        clock = self._inner.clock_utc
        result["time"] = np.asarray(
            [
                clock - self.request.ready_utc,
                self.request.deadline_utc - clock,
                self.request.horizon_utc - clock,
                self.request.max_attempts - self._attempts_used(),
            ],
            dtype=np.float32,
        )
        result["disrupted"] = int(self._disrupted())
        return result

    def _make_record(self, reason: str) -> RecoveryRecord:
        return RecoveryRecord(
            request=self.request,
            config=self.config,
            segments=tuple(self._segments),
            events=tuple(self._events),
            final_airport=self._inner.record.final_airport,
            clock_utc=self._inner.clock_utc,
            termination_reason=reason,
        )

    def _finish(
        self,
        observation: dict[str, np.ndarray | int],
        reason: str,
        core_info: dict[str, Any],
    ) -> tuple[dict[str, np.ndarray | int], float, bool, bool, dict[str, Any]]:
        self._done = True
        self._record = self._make_record(reason)
        verification = verify_recovery(self.record)
        legacy_verification = (
            verify_recovery_legacy_six_v1(self.record)
            if self.reward_mode == "legacy_six_v1"
            else None
        )
        metrics = recovery_metrics(self.record)
        if legacy_verification is not None:
            reward = float(legacy_verification.aggregate_score)
        elif self.reward_mode == "deadline_first":
            reward = float(verification.aggregate_score)
        else:
            reward = float(bool(metrics["on_time_arrival"]))
        info = {
            "reward_mode": self.reward_mode,
            "score_profile": score_profile_for_reward_mode(self.reward_mode),
            "recovery": self.record,
            "core": core_info,
            "verification": verification,
            "metrics": metrics,
        }
        if legacy_verification is not None:
            info["legacy_six_v1_verification"] = legacy_verification
        return (
            self._global_observation(observation),
            reward,
            True,
            False,
            info,
        )

    def render(self) -> str:
        text = (
            f"airport={self.record.final_airport} destination={self.request.destination} "
            f"clock_utc={self.clock_utc} attempts={self._attempts_used()}/"
            f"{self.request.max_attempts} rebookings="
            f"{sum(event.decision == 'rebooked' for event in self._events)} "
            f"status={self.record.termination_reason}"
        )
        if self.render_mode == "human":
            print(text)
        return text

    def close(self) -> None:
        self._inner.close()


def evaluate_recovery_policy(
    env_factory: Callable[[int], RecoveryEnv],
    policy_factory: Callable[[RecoveryEnv, int], Any],
    *,
    episodes: int,
    seed: int,
    trace_count: int = 2,
) -> dict[str, Any]:
    """Evaluate one policy on explicit scenario keys with recovery-specific metrics."""

    from flight_rl.evaluation import json_value, wilson_interval

    if episodes < 1 or trace_count < 0:
        raise ValueError("episodes must be positive and trace_count nonnegative")
    rows: list[dict[str, object]] = []
    traces: list[dict[str, Any]] = []
    scores: list[float] = []
    legacy_scores: list[float] = []
    returns: list[float] = []
    reasons: dict[str, int] = {}
    decisions: dict[str, int] = {}
    reward_modes: set[str] = set()

    for episode in range(episodes):
        scenario_seed = seed + episode
        env = env_factory(scenario_seed)
        try:
            policy = policy_factory(env, seed + 1_000_003 + episode)
            observation, _ = env.reset(seed=scenario_seed)
            actions: list[int] = []
            total_reward = 0.0
            for _ in range(env.request.max_attempts + 1):
                action = policy.act(observation)
                actions.append(action)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break
            else:
                raise RuntimeError("Recovery environment exceeded the original attempt budget")

            record = env.record
            verification = verify_recovery(record)
            legacy_verification = verify_recovery_legacy_six_v1(record)
            metrics = recovery_metrics(record)
            row = {
                "episode": episode,
                "scenario_seed": scenario_seed,
                "validity": bool(metrics["validity"]),
                "arrived": bool(metrics["arrived"]),
                "on_time_arrival": bool(metrics["on_time_arrival"]),
                "failed": bool(metrics["failed"]),
                "elapsed_min": metrics["elapsed_min"],
                "attempts": int(metrics["attempts"]),
                "rebookings": int(metrics["rebookings"]),
                "cancellation_segments": int(metrics["cancellation_segments"]),
                "termination_reason": str(metrics["termination_reason"]),
            }
            rows.append(row)
            scores.append(float(verification.aggregate_score))
            legacy_scores.append(float(legacy_verification.aggregate_score))
            returns.append(total_reward)
            reward_modes.add(env.reward_mode)
            reason = record.termination_reason
            reasons[reason] = reasons.get(reason, 0) + 1
            for event in record.events:
                decisions[event.decision] = decisions.get(event.decision, 0) + 1
            if episode < trace_count:
                traces.append(
                    json_value(
                        {
                            "episode": episode,
                            "scenario_seed": scenario_seed,
                            "actions": actions,
                            "record": record,
                            "verification": verification,
                            "legacy_six_v1_verification": legacy_verification,
                            "metrics": metrics,
                            "return": total_reward,
                        }
                    )
                )
        finally:
            env.close()

    arrivals = sum(bool(row["arrived"]) for row in rows)
    on_time = sum(bool(row["on_time_arrival"]) for row in rows)
    attempts = sum(int(row["attempts"]) for row in rows)
    rebookings = sum(int(row["rebookings"]) for row in rows)
    cancellations = sum(int(row["cancellation_segments"]) for row in rows)
    successful_elapsed = [row["elapsed_min"] for row in rows if row["arrived"]]
    if len(reward_modes) != 1:
        raise RuntimeError("Environment factory returned inconsistent reward modes")
    return {
        "episodes": episodes,
        "seed": seed,
        "score_profile": PRIMARY_SCORE_PROFILE,
        "reward_mode": next(iter(reward_modes)),
        "arrived": arrivals,
        "on_time_arrivals": on_time,
        "failures": episodes - arrivals,
        "invalid_records": sum(not bool(row["validity"]) for row in rows),
        "arrival_rate": arrivals / episodes,
        "on_time_arrival_rate": on_time / episodes,
        "arrival_rate_interval95": wilson_interval(arrivals, episodes),
        "on_time_arrival_rate_interval95": wilson_interval(on_time, episodes),
        "mean_score": float(np.mean(scores)),
        "mean_legacy_six_v1": float(np.mean(legacy_scores)),
        "mean_return": float(np.mean(returns)),
        "total_attempts": attempts,
        "mean_attempts": attempts / episodes,
        "total_rebookings": rebookings,
        "mean_rebookings": rebookings / episodes,
        "total_cancellation_segments": cancellations,
        "termination_counts": reasons,
        "event_decision_counts": decisions,
        "episode_outcomes": rows,
        "traces": traces,
        "mean_elapsed_given_arrival_min": (
            float(np.mean(successful_elapsed)) if successful_elapsed else None
        ),
        "interval_scope": "IID scenario keys conditional on this fixed request and data source; "
        "not historical-model, training, population, or causal uncertainty",
        "recoverability_scope": "Rebooking assumes a seat is available after each eligible "
        "cancellation; source membership and seat inventory are not authenticated.",
    }
