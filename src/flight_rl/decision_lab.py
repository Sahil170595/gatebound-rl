"""Interactive, flight-keyed replay and initial-action comparison engine.

The lab deliberately sits on top of :class:`FlightRouteEnv` instead of
reimplementing transition rules.  User choices are flight IDs, so a replay can
reconstruct a committed prefix from the origin while checking legality at every
step.  Once that prefix ends, one of the frozen policies selects the suffix from
the public observation only.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from flight_rl.baselines import NonstopFirstPolicy, ShortestScheduledArrivalPolicy
from flight_rl.env import FlightRouteEnv
from flight_rl.models import (
    EpisodeRecord,
    FlightCandidate,
    FlightDataSource,
    LegRecord,
    OutcomeSummary,
    TripRequest,
)
from flight_rl.planning import DeadlinePlannerPolicy
from flight_rl.scenarios import ScenarioData, ScenarioPoolCache
from flight_rl.source_auth import SourceBackedVerifier
from flight_rl.verifier import PRIMARY_SCORE_PROFILE, episode_metrics

POLICY_IDS = ("nonstop_first", "shortest_scheduled", "deadline_planner")

RUBRIC_LABELS: Mapping[str, str] = {
    "on_time_arrival": "Met deadline",
    "arrived": "Reached destination",
    "earliness": "Earlier arrival",
}

COMPARISON_SCOPE = (
    "Conditional simulator Monte Carlo across fixed scenario_seed+i keys; "
    "initial-flight choices only; deltas are alternative-minus-reference rates, "
    "not predictive-model confidence."
)

_ENV_SEED_MODULUS = 1 << 32


@dataclass(frozen=True)
class _Execution:
    selected_flight_ids: tuple[str, ...]
    steps: tuple[dict[str, Any], ...]
    record: EpisodeRecord
    metrics: dict[str, bool | int | str | None]
    aggregate_score: float
    rubric: tuple[dict[str, Any], ...]


class LabEngine:
    """Run deterministic decision-lab replays over one immutable data source."""

    def __init__(self, data: FlightDataSource, max_candidates: int = 64) -> None:
        if (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, Integral)
            or int(max_candidates) < 1
        ):
            raise ValueError("max_candidates must be a positive integer")
        self._max_candidates = int(max_candidates)
        self._data = data if isinstance(data, ScenarioPoolCache) else ScenarioPoolCache(data)
        self._source_verifier = SourceBackedVerifier(self._data)
        # Policies never receive ScenarioData: the planner sees fitted history only,
        # and the two schedule policies see only the environment observation.
        self._policy_cache: dict[tuple[TripRequest, str], Any] = {}

    def replay(
        self,
        request: TripRequest,
        scenario_seed: int,
        policy_id: str = "deadline_planner",
        choices: Iterable[str] = (),
    ) -> dict[str, Any]:
        """Replay a legal committed prefix, then finish with a frozen policy."""

        request = self._validated_request(request)
        seed = self._validated_seed(scenario_seed)
        policy_id = self._validated_policy_id(policy_id)
        prefix = self._validated_choices(choices)
        execution = self._execute(
            request,
            seed,
            policy_id,
            prefix,
            include_timeline=True,
        )
        metrics = execution.metrics
        return {
            "scenario_seed": seed,
            "policy_id": policy_id,
            "choices": list(execution.selected_flight_ids),
            "request": self._request_payload(request),
            "steps": list(execution.steps),
            "summary": {
                "arrived": bool(metrics["arrived"]),
                "on_time_arrival": bool(metrics["on_time_arrival"]),
                "validity": bool(metrics["validity"]),
                "source_authenticated": True,
                "elapsed_min": metrics["elapsed_min"],
                "termination_reason": execution.record.termination_reason,
                "final_airport": execution.record.final_airport,
                "clock_utc": execution.record.clock_utc,
                "score_profile": PRIMARY_SCORE_PROFILE,
                "rubric_score": execution.aggregate_score,
            },
            "rubric": list(execution.rubric),
        }

    def compare_first_actions(
        self,
        request: TripRequest,
        scenario_seed: int,
        policy_id: str,
        reference_flight_id: str,
        alternative_flight_id: str,
        trials: int = 32,
    ) -> dict[str, Any]:
        """Compare two legal initial flights on the same bounded scenario keys."""

        request = self._validated_request(request)
        seed = self._validated_seed(scenario_seed)
        policy_id = self._validated_policy_id(policy_id)
        count = self._validated_trials(trials)
        reference = self._validated_flight_id(reference_flight_id, "reference_flight_id")
        alternative = self._validated_flight_id(alternative_flight_id, "alternative_flight_id")

        legal = self._initial_flight_ids(request, seed)
        for label, flight_id in (
            ("reference_flight_id", reference),
            ("alternative_flight_id", alternative),
        ):
            if flight_id not in legal:
                raise ValueError(f"{label} {flight_id!r} is not legal at the initial state")

        reference_rows: list[_Execution] = []
        alternative_rows: list[_Execution] = []
        for offset in range(count):
            trial_seed = seed + offset
            reference_rows.append(
                self._execute(
                    request,
                    trial_seed,
                    policy_id,
                    (reference,),
                    include_timeline=False,
                )
            )
            alternative_rows.append(
                self._execute(
                    request,
                    trial_seed,
                    policy_id,
                    (alternative,),
                    include_timeline=False,
                )
            )

        reference_result = self._comparison_arm(reference, reference_rows)
        alternative_result = self._comparison_arm(alternative, alternative_rows)
        return {
            "score_profile": PRIMARY_SCORE_PROFILE,
            "source_authenticated_episodes": 2 * count,
            "trials": count,
            "reference": reference_result,
            "alternative": alternative_result,
            "deadline_delta": (
                alternative_result["on_time_arrivals"] - reference_result["on_time_arrivals"]
            )
            / count,
            "arrival_delta": (alternative_result["arrived"] - reference_result["arrived"]) / count,
            "scope": COMPARISON_SCOPE,
        }

    def _execute(
        self,
        request: TripRequest,
        scenario_seed: int,
        policy_id: str,
        prefix: tuple[str, ...],
        *,
        include_timeline: bool,
    ) -> _Execution:
        scenario_data = ScenarioData(self._data, scenario_seed)
        env = FlightRouteEnv(
            request,
            scenario_data,
            max_candidates=self._max_candidates,
            reward_mode="deadline_first",
        )
        policy = self._policy(request, policy_id)
        selected_ids: list[str] = []
        steps: list[dict[str, Any]] = []
        prefix_index = 0

        try:
            # ScenarioData owns the signed scenario key. Gymnasium requires a
            # nonnegative RNG seed, whose value cannot affect flight-keyed draws.
            observation, _ = env.reset(seed=scenario_seed % _ENV_SEED_MODULUS)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                candidates = env.available_flights
                if not candidates:
                    if prefix_index < len(prefix):
                        raise ValueError(
                            "Choice prefix continues after the itinerary has no legal flights"
                        )
                    observation, _, terminated, truncated, _ = env.step(0)
                    del observation
                    break

                recommended_index: int | None = None
                if include_timeline or prefix_index >= len(prefix):
                    recommended_index = int(policy.act(observation))
                    if not 0 <= recommended_index < len(candidates):
                        raise RuntimeError(f"Policy {policy_id!r} selected an illegal action index")

                if prefix_index < len(prefix):
                    selected_id = prefix[prefix_index]
                    action_index = next(
                        (
                            index
                            for index, candidate in enumerate(candidates)
                            if candidate.flight_id == selected_id
                        ),
                        None,
                    )
                    if action_index is None:
                        raise ValueError(
                            f"Choice {selected_id!r} is not legal at step {prefix_index}"
                        )
                    prefix_index += 1
                else:
                    assert recommended_index is not None
                    action_index = recommended_index
                    selected_id = candidates[action_index].flight_id

                if include_timeline:
                    assert recommended_index is not None
                    # act() has already populated these fitted-history action values.
                    # Reading the cache makes the recommendation inspectable without
                    # exposing the sampled world to the policy.
                    deadline_values = (
                        policy.action_values(observation)
                        if isinstance(policy, DeadlinePlannerPolicy)
                        else (None,) * len(candidates)
                    )
                    step_payload: dict[str, Any] = {
                        "index": len(selected_ids),
                        "airport": env.record.final_airport,
                        "clock_utc": env.clock_utc,
                        "candidates": [
                            self._candidate_payload(
                                candidate, scenario_data, deadline_values[index]
                            )
                            for index, candidate in enumerate(candidates)
                        ],
                        "selected_flight_id": selected_id,
                        "recommended_flight_id": candidates[recommended_index].flight_id,
                    }

                prior_leg_count = len(env.record.legs)
                observation, _, terminated, truncated, _ = env.step(action_index)
                selected_ids.append(selected_id)
                if len(env.record.legs) != prior_leg_count + 1:
                    raise RuntimeError("A legal flight action did not append exactly one leg")
                if include_timeline:
                    step_payload["outcome"] = self._outcome_payload(
                        env.record.legs[-1], env.record.termination_reason
                    )
                    steps.append(step_payload)

                if (terminated or truncated) and prefix_index < len(prefix):
                    raise ValueError("Choice prefix continues after the itinerary has terminated")
            if not (terminated or truncated):
                raise RuntimeError("Environment failed to terminate")

            record = env.record
            source_verified = self._source_verifier.verify(record)
            if not source_verified.authentication.authenticated:
                raise RuntimeError(
                    "Completed replay failed source authentication: "
                    f"{source_verified.authentication.reason}"
                )
            verification = source_verified.score
            metrics = episode_metrics(record)
            if not bool(metrics["validity"]):
                raise RuntimeError("Completed replay failed independent itinerary verification")
            rubric = tuple(
                {
                    "name": item.name,
                    "label": RUBRIC_LABELS[item.name],
                    "raw_score": float(item.raw_score),
                    "weight": float(item.weight),
                    "contribution": float(item.weighted_contribution),
                }
                for item in verification.breakdown
            )
            return _Execution(
                selected_flight_ids=tuple(selected_ids),
                steps=tuple(steps),
                record=record,
                metrics=metrics,
                aggregate_score=float(verification.aggregate_score),
                rubric=rubric,
            )
        finally:
            env.close()

    def _policy(self, request: TripRequest, policy_id: str) -> Any:
        key = (request, policy_id)
        cached = self._policy_cache.get(key)
        if cached is not None:
            return cached
        if policy_id == "nonstop_first":
            policy: Any = NonstopFirstPolicy()
        elif policy_id == "shortest_scheduled":
            policy = ShortestScheduledArrivalPolicy()
        else:
            policy = DeadlinePlannerPolicy(
                request,
                self._data,
                max_candidates=self._max_candidates,
            )
        self._policy_cache[key] = policy
        return policy

    def _initial_flight_ids(self, request: TripRequest, scenario_seed: int) -> frozenset[str]:
        env = FlightRouteEnv(
            request,
            ScenarioData(self._data, scenario_seed),
            max_candidates=self._max_candidates,
        )
        try:
            env.reset(seed=scenario_seed % _ENV_SEED_MODULUS)
            return frozenset(candidate.flight_id for candidate in env.available_flights)
        finally:
            env.close()

    @staticmethod
    def _candidate_payload(
        flight: FlightCandidate, data: ScenarioData, deadline_probability: float | None
    ) -> dict[str, str | int | float | None]:
        summary = data.outcome_summary(flight)
        if not isinstance(summary, OutcomeSummary):
            raise ValueError(  # noqa: TRY004 - stable request-facing error contract
                "data.outcome_summary must return OutcomeSummary"
            )
        return {
            "flight_id": flight.flight_id,
            "origin": flight.origin,
            "destination": flight.dest,
            "carrier": flight.carrier,
            "departure_utc": int(flight.scheduled_departure_utc),
            "arrival_utc": int(flight.scheduled_arrival_utc),
            "p_cancelled": float(summary.p_cancelled),
            "p_diverted": float(summary.p_diverted),
            "mean_delay_min": float(summary.mean_arrival_delay_min),
            "support": int(summary.support),
            "deadline_probability": deadline_probability,
        }

    @staticmethod
    def _outcome_payload(leg: LegRecord, termination_reason: str) -> dict[str, Any]:
        outcome = leg.outcome
        status = (
            "cancelled"
            if outcome.cancelled
            else termination_reason
            if termination_reason in {"invalid_outcome", "missed_departure", "unresolved_diversion"}
            else "diverted"
            if outcome.diverted
            else "completed"
        )
        return {
            "donor_id": outcome.donor_id,
            "cancelled": bool(outcome.cancelled),
            "diverted": bool(outcome.diverted),
            "departure_utc": leg.departure_utc,
            "arrival_utc": leg.arrival_utc,
            "resolved_airport": leg.resolved_airport,
            "status": status,
        }

    @staticmethod
    def _request_payload(request: TripRequest) -> dict[str, str | int]:
        return {
            "origin": request.origin,
            "destination": request.destination,
            "ready_utc": int(request.ready_utc),
            "deadline_utc": int(request.deadline_utc),
            "horizon_utc": int(request.horizon_utc),
            "max_attempts": int(request.max_attempts),
            "min_connection_min": int(request.min_connection_min),
            "delay_budget_min": int(request.delay_budget_min),
        }

    @staticmethod
    def _comparison_arm(flight_id: str, rows: list[_Execution]) -> dict[str, Any]:
        if not rows:
            raise RuntimeError("Comparison requires at least one completed trial")
        return {
            "flight_id": flight_id,
            "arrived": sum(bool(row.metrics["arrived"]) for row in rows),
            "on_time_arrivals": sum(bool(row.metrics["on_time_arrival"]) for row in rows),
            "mean_score": math.fsum(row.aggregate_score for row in rows) / len(rows),
        }

    @staticmethod
    def _validated_request(request: TripRequest) -> TripRequest:
        if not isinstance(request, TripRequest):
            raise ValueError("request must be a TripRequest")  # noqa: TRY004 - API contract
        return request

    @staticmethod
    def _validated_seed(scenario_seed: int) -> int:
        if isinstance(scenario_seed, bool) or not isinstance(scenario_seed, Integral):
            raise ValueError(  # noqa: TRY004 - stable request-facing error contract
                "scenario_seed must be a non-boolean integer"
            )
        return int(scenario_seed)

    @staticmethod
    def _validated_policy_id(policy_id: str) -> str:
        if not isinstance(policy_id, str) or policy_id not in POLICY_IDS:
            raise ValueError(f"Unknown policy_id {policy_id!r}; expected one of {list(POLICY_IDS)}")
        return policy_id

    @staticmethod
    def _validated_choices(choices: Iterable[str]) -> tuple[str, ...]:
        if isinstance(choices, (str, bytes)):
            raise ValueError(  # noqa: TRY004 - stable request-facing error contract
                "choices must be an iterable of flight IDs"
            )
        try:
            prefix = tuple(choices)
        except TypeError as exc:
            raise ValueError("choices must be an iterable of flight IDs") from exc
        if any(not isinstance(value, str) or not value for value in prefix):
            raise ValueError("choices must contain nonempty flight IDs")
        return prefix

    @staticmethod
    def _validated_flight_id(value: str, label: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be a nonempty flight ID")
        return value

    @staticmethod
    def _validated_trials(trials: int) -> int:
        if isinstance(trials, bool) or not isinstance(trials, Integral):
            raise ValueError(  # noqa: TRY004 - stable request-facing error contract
                "trials must be an integer from 1 to 64"
            )
        value = int(trials)
        if not 1 <= value <= 64:
            raise ValueError("trials must be an integer from 1 to 64")
        return value
