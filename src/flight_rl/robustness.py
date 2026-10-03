"""Paired scenario evaluation and post-hoc verifier-weight sensitivity."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any

import numpy as np

from .evaluation import json_value, wilson_interval
from .models import EpisodeRecord, TripRequest, utc_minutes
from .scenarios import ScenarioData
from .verifier import (
    LEGACY_SIX_V1_PROFILE,
    LEGACY_SIX_V1_WEIGHTS,
    episode_metrics,
    verify_legacy_six_v1,
)

CRITERION_NAMES = tuple(LEGACY_SIX_V1_WEIGHTS)
LEGACY_WEIGHT_PROFILES = MappingProxyType(
    {
        LEGACY_SIX_V1_PROFILE: LEGACY_SIX_V1_WEIGHTS,
        "deadline_priority": MappingProxyType(
            dict(zip(CRITERION_NAMES, (0.20, 0.60, 0.05, 0.05, 0.05, 0.05), strict=True))
        ),
        "arrival_priority": MappingProxyType(
            dict(zip(CRITERION_NAMES, (0.70, 0.10, 0.05, 0.05, 0.05, 0.05), strict=True))
        ),
        "journey_quality": MappingProxyType(
            dict(zip(CRITERION_NAMES, (0.25, 0.15, 0.20, 0.15, 0.10, 0.15), strict=True))
        ),
    }
)

PolicyFactory = Callable[[int], Any]


def fixed_2024_cases() -> dict[str, TripRequest]:
    """Return the fixed six-case 2024 schedule used for robustness axes."""

    cases: dict[str, TripRequest] = {}
    for name, origin, destination, month_day in (
        ("winter_west_east", "SFO", "JFK", "01-15"),
        ("summer_east_west", "JFK", "SFO", "07-15"),
        ("spring_atl_sea", "ATL", "SEA", "04-15"),
        ("autumn_dfw_bos", "DFW", "BOS", "10-15"),
        ("winter_mia_den", "MIA", "DEN", "01-22"),
        ("summer_ewr_slc", "EWR", "SLC", "07-22"),
    ):
        ready = utc_minutes(f"2024-{month_day}T12:00:00Z")
        cases[name] = TripRequest(
            origin,
            destination,
            ready,
            ready + 720,
            ready + 1440,
        )
    return cases


@dataclass(frozen=True)
class _EpisodeResult:
    scenario_key: int
    metrics: Mapping[str, bool | int | str | None]
    raw_scores: Mapping[str, float]
    profile_scores: Mapping[str, float]
    total_return: float
    termination_reason: str
    trace: Mapping[str, Any] | None


def _is_bool(value: object) -> bool:
    return isinstance(value, (bool, np.bool_))


def _finite_nonnegative(value: object, label: str) -> float:
    if not isinstance(value, Real) or _is_bool(value):
        raise ValueError(f"{label} must be a finite nonnegative number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return converted


def _validated_weights(weights: Mapping[str, object]) -> dict[str, float]:
    if not isinstance(weights, Mapping):
        raise TypeError("weights must be a mapping")
    if set(weights) != set(CRITERION_NAMES):
        raise ValueError(f"weights must contain exactly {list(CRITERION_NAMES)}")
    validated = {
        name: _finite_nonnegative(weights[name], f"weight {name!r}") for name in CRITERION_NAMES
    }
    try:
        total = math.fsum(validated.values())
    except OverflowError as exc:
        raise ValueError("weight sum must be finite and positive") from exc
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("weight sum must be finite and positive")
    return validated


def weight_profiles(
    extra_profiles: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, dict[str, float]]:
    """Return legacy six-signal profiles for post-hoc sensitivity analysis."""

    profiles: dict[str, Mapping[str, object]] = dict(LEGACY_WEIGHT_PROFILES)
    if extra_profiles is not None:
        if not isinstance(extra_profiles, Mapping):
            raise ValueError("extra_profiles must be a mapping")
        for name, weights in extra_profiles.items():
            if not isinstance(name, str) or not name:
                raise ValueError("weight profile names must be nonempty strings")
            if not isinstance(weights, Mapping):
                raise TypeError(f"weight profile {name!r} must be a mapping")
            if name == LEGACY_SIX_V1_PROFILE and dict(weights) != dict(LEGACY_SIX_V1_WEIGHTS):
                raise ValueError("legacy_six_v1 must equal the frozen compatibility profile")
            profiles[name] = weights
    return {name: _validated_weights(weights) for name, weights in profiles.items()}


def reweighted_score(raw_scores: Mapping[str, object], weights: Mapping[str, object]) -> float:
    """Aggregate one already observed criterion vector under alternate preferences."""

    if not isinstance(raw_scores, Mapping) or set(raw_scores) != set(CRITERION_NAMES):
        raise ValueError(f"raw_scores must contain exactly {list(CRITERION_NAMES)}")
    validated_weights = _validated_weights(weights)
    validated_scores: dict[str, float] = {}
    for name in CRITERION_NAMES:
        value = raw_scores[name]
        if not isinstance(value, Real) or _is_bool(value):
            raise ValueError(f"raw score {name!r} must be finite and in [0, 1]")
        score = float(value)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"raw score {name!r} must be finite and in [0, 1]")
        validated_scores[name] = score
    denominator = math.fsum(validated_weights.values())
    numerator = math.fsum(
        validated_scores[name] * validated_weights[name] for name in CRITERION_NAMES
    )
    result = numerator / denominator
    if not math.isfinite(result):
        raise ValueError("reweighted score must be finite")
    return result


def _approx_normal_mean_interval95(
    values: Sequence[float], *, lower: float, upper: float
) -> list[float] | None:
    if not values:
        raise ValueError("interval values must be nonempty")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not bool(np.all(np.isfinite(array))):
        raise ValueError("interval values must be a finite vector")
    mean = float(np.mean(array))
    if len(array) == 1:
        return None
    standard_deviation = float(np.std(array, ddof=1))
    if standard_deviation == 0.0:
        return None
    radius = 1.959963984540054 * standard_deviation / math.sqrt(len(array))
    return [max(lower, mean - radius), min(upper, mean + radius)]


def _runtime_seed(scenario_key: int, namespace: str) -> int:
    encoded = f"{scenario_key}:{namespace}".encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _scenario_keys(values: Sequence[int]) -> tuple[int, ...]:
    keys: list[int] = []
    for value in values:
        if not isinstance(value, Integral) or _is_bool(value):
            raise ValueError("scenario keys must be non-boolean integers")
        keys.append(int(value))
    if not keys:
        raise ValueError("scenario keys must be nonempty")
    if len(set(keys)) != len(keys):
        raise ValueError("scenario keys must be unique")
    return tuple(keys)


def _run_episode(
    *,
    request: TripRequest,
    scenario_data: ScenarioData,
    policy_factory: PolicyFactory,
    policy_seed: int,
    environment_seed: int,
    max_candidates: int,
    profiles: Mapping[str, Mapping[str, object]],
    scenario_key: int,
    include_trace: bool,
) -> _EpisodeResult:
    from .env import FlightRouteEnv

    env = FlightRouteEnv(
        request,
        scenario_data,
        max_candidates=max_candidates,
        reward_mode="legacy_six_v1",
    )
    try:
        policy = policy_factory(policy_seed)
        if not callable(getattr(policy, "act", None)):
            raise TypeError("policy factories must return an object with callable act()")
        observation, _ = env.reset(seed=environment_seed)
        total_return = 0.0
        actions: list[int] = []
        for _ in range(request.max_attempts + 1):
            action = policy.act(observation)
            actions.append(int(action))
            observation, reward, terminated, truncated, _ = env.step(action)
            total_return += float(reward)
            if terminated or truncated:
                break
        else:
            raise RuntimeError("Environment did not finish within the request attempt budget")
        record: EpisodeRecord = env.record
        verification = verify_legacy_six_v1(record)
        raw_scores = {item.name: float(item.raw_score) for item in verification.breakdown}
        if set(raw_scores) != set(CRITERION_NAMES):
            raise RuntimeError("Verifier criterion names differ from the robustness contract")
        profile_scores = {
            name: reweighted_score(raw_scores, weights) for name, weights in profiles.items()
        }
        metrics = episode_metrics(record)
        trace = None
        if include_trace:
            trace = json_value(
                {
                    "scenario_key": scenario_key,
                    "actions": actions,
                    "record": record,
                    "metrics": metrics,
                    "raw_scores": raw_scores,
                    "profile_scores": profile_scores,
                    "return": total_return,
                }
            )
        return _EpisodeResult(
            scenario_key=scenario_key,
            metrics=metrics,
            raw_scores=raw_scores,
            profile_scores=profile_scores,
            total_return=total_return,
            termination_reason=record.termination_reason,
            trace=trace,
        )
    finally:
        env.close()


def _summarize_policy(
    rows: Sequence[_EpisodeResult], profiles: Mapping[str, Mapping[str, object]]
) -> dict[str, Any]:
    count = len(rows)
    arrivals = [float(bool(row.metrics["arrived"])) for row in rows]
    on_time = [float(bool(row.metrics["on_time_arrival"])) for row in rows]
    disrupted = [float(bool(row.metrics["disrupted"])) for row in rows]
    valid = [float(bool(row.metrics["validity"])) for row in rows]
    elapsed = [
        float(row.metrics["elapsed_min"])
        for row in rows
        if bool(row.metrics["arrived"]) and row.metrics["elapsed_min"] is not None
    ]
    profile_summary = {}
    for profile_name in profiles:
        values = [row.profile_scores[profile_name] for row in rows]
        profile_summary[profile_name] = {
            "mean_score": float(np.mean(values)),
            "approx_normal_mean_interval95": _approx_normal_mean_interval95(
                values, lower=0.0, upper=1.0
            ),
        }
    raw_means = {
        name: float(np.mean([row.raw_scores[name] for row in rows])) for name in CRITERION_NAMES
    }
    return {
        "episodes": count,
        "arrived": int(sum(arrivals)),
        "on_time_arrivals": int(sum(on_time)),
        "invalid_records": count - int(sum(valid)),
        "arrival_rate": float(np.mean(arrivals)),
        "on_time_arrival_rate": float(np.mean(on_time)),
        "arrival_rate_interval95": wilson_interval(int(sum(arrivals)), count),
        "on_time_arrival_rate_interval95": wilson_interval(int(sum(on_time)), count),
        "score_profile": LEGACY_SIX_V1_PROFILE,
        "mean_score": profile_summary[LEGACY_SIX_V1_PROFILE]["mean_score"],
        "mean_legacy_six_v1": profile_summary[LEGACY_SIX_V1_PROFILE]["mean_score"],
        "mean_return": float(np.mean([row.total_return for row in rows])),
        "mean_elapsed_given_arrival_min": float(np.mean(elapsed)) if elapsed else None,
        "disruption_rate": float(np.mean(disrupted)),
        "termination_counts": dict(sorted(Counter(row.termination_reason for row in rows).items())),
        "mean_raw_criteria": raw_means,
        "weight_sensitivity": profile_summary,
        "scenario_outcomes": [
            {
                "scenario_key": row.scenario_key,
                "arrived": bool(row.metrics["arrived"]),
                "on_time_arrival": bool(row.metrics["on_time_arrival"]),
                "raw_scores": dict(row.raw_scores),
                "profile_scores": dict(row.profile_scores),
                "termination_reason": row.termination_reason,
            }
            for row in rows
        ],
        "traces": [row.trace for row in rows if row.trace is not None],
    }


def _paired_policy_differences(
    rows_by_policy: Mapping[str, Sequence[_EpisodeResult]],
    profiles: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for first_name, second_name in combinations(rows_by_policy, 2):
        first_rows = rows_by_policy[first_name]
        second_rows = rows_by_policy[second_name]
        if [row.scenario_key for row in first_rows] != [row.scenario_key for row in second_rows]:
            raise RuntimeError("Policy rows do not share the same ordered scenario keys")
        metric_values: dict[str, tuple[list[float], float, float]] = {
            "arrival": (
                [
                    float(bool(second.metrics["arrived"])) - float(bool(first.metrics["arrived"]))
                    for first, second in zip(first_rows, second_rows, strict=True)
                ],
                -1.0,
                1.0,
            ),
            "on_time_arrival": (
                [
                    float(bool(second.metrics["on_time_arrival"]))
                    - float(bool(first.metrics["on_time_arrival"]))
                    for first, second in zip(first_rows, second_rows, strict=True)
                ],
                -1.0,
                1.0,
            ),
        }
        for profile_name in profiles:
            metric_values[f"score:{profile_name}"] = (
                [
                    second.profile_scores[profile_name] - first.profile_scores[profile_name]
                    for first, second in zip(first_rows, second_rows, strict=True)
                ],
                -1.0,
                1.0,
            )
        comparison = {}
        for metric_name, (values, lower, upper) in metric_values.items():
            comparison[metric_name] = {
                "mean_second_minus_first": float(np.mean(values)),
                "paired_approx_normal_interval95": _approx_normal_mean_interval95(
                    values, lower=lower, upper=upper
                ),
            }
        result[f"{second_name}_minus_{first_name}"] = {
            "first_policy": first_name,
            "second_policy": second_name,
            "metrics": comparison,
        }
    return result


def run_paired_scenarios(
    *,
    request: TripRequest,
    data: object,
    policy_factories: Mapping[str, PolicyFactory],
    scenario_keys: Sequence[int],
    dependence: float = 0.0,
    max_candidates: int = 64,
    weight_profile_overrides: Mapping[str, Mapping[str, object]] | None = None,
    trace_count: int = 1,
) -> dict[str, Any]:
    """Evaluate policies against a shared deterministic universe of flight outcomes.

    Alternate verifier weights are applied to the same recorded raw criterion vectors.
    Policies are neither reoptimized nor rerun for those profiles. Policy factories receive
    only a deterministic seed; model-based factories should close over the fit-facing source.
    """

    if not isinstance(request, TripRequest):
        raise TypeError("request must be a TripRequest")
    if not isinstance(policy_factories, Mapping) or not policy_factories:
        raise ValueError("policy_factories must be a nonempty mapping")
    for name, factory in policy_factories.items():
        if not isinstance(name, str) or not name or not callable(factory):
            raise ValueError("policy names must be nonempty strings with callable factories")
    if not isinstance(max_candidates, Integral) or _is_bool(max_candidates) or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    if not isinstance(trace_count, Integral) or _is_bool(trace_count) or trace_count < 0:
        raise ValueError("trace_count must be a nonnegative integer")

    keys = _scenario_keys(scenario_keys)
    profiles = weight_profiles(weight_profile_overrides)
    rows_by_policy: dict[str, list[_EpisodeResult]] = {name: [] for name in policy_factories}
    regime_counts: Counter[str] = Counter()
    scenario_diagnostics: list[dict[str, object]] = []
    for episode_index, scenario_key in enumerate(keys):
        scenario_data = ScenarioData(data, scenario_seed=scenario_key, dependence=dependence)
        diagnostics = scenario_data.diagnostics()
        regime_counts[str(diagnostics["coupling_mode"])] += 1
        if episode_index < max(1, int(trace_count)):
            scenario_diagnostics.append(diagnostics)
        for policy_name, factory in policy_factories.items():
            row = _run_episode(
                request=request,
                scenario_data=scenario_data,
                policy_factory=factory,
                policy_seed=_runtime_seed(scenario_key, "policy"),
                environment_seed=_runtime_seed(scenario_key, "environment"),
                max_candidates=int(max_candidates),
                profiles=profiles,
                scenario_key=scenario_key,
                include_trace=episode_index < int(trace_count),
            )
            rows_by_policy[policy_name].append(row)

    encoded_keys = json.dumps(keys, separators=(",", ":")).encode("utf-8")
    return {
        "score_profile": LEGACY_SIX_V1_PROFILE,
        "scenario": {
            "keys": list(keys),
            "keys_sha256": hashlib.sha256(encoded_keys).hexdigest(),
            "dependence": float(dependence),
            "coupling_regime_counts": dict(sorted(regime_counts.items())),
            "diagnostic_examples": scenario_diagnostics,
            "pairing": "same scenario key defines every flight outcome for every policy; "
            "different chosen flights remain different interventions",
        },
        "weight_profiles": profiles,
        "weight_analysis": "post-hoc legacy six-signal scores from the same episode criterion "
        "vectors; policies were not reoptimized or rerun by profile",
        "policies": {
            name: _summarize_policy(rows, profiles) for name, rows in rows_by_policy.items()
        },
        "paired_policy_differences": _paired_policy_differences(rows_by_policy, profiles),
        "uncertainty_scope": "Wilson intervals for individual binary rates and normal intervals "
        "for paired Monte Carlo differences, conditional on this fixed request, local fitted "
        "data, policies, scenario-key generator and dependence construction; excludes historical "
        "model error, fit-year selection, policy training, causal effects and multiplicity. "
        "Approximate normal mean intervals are unavailable for one observation or zero empirical "
        "variance because those samples do not establish zero population uncertainty",
    }


def paired_condition_differences(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare two condition reports using their aligned compact scenario rows."""

    try:
        first_keys = first["scenario"]["keys"]
        second_keys = second["scenario"]["keys"]
        first_policies = first["policies"]
        second_policies = second["policies"]
        first_profiles = first["weight_profiles"]
        second_profiles = second["weight_profiles"]
    except (KeyError, TypeError) as exc:
        raise ValueError("condition reports lack paired robustness fields") from exc
    if first_keys != second_keys:
        raise ValueError("paired condition reports require identical ordered scenario keys")
    if set(first_policies) != set(second_policies):
        raise ValueError("paired condition reports require identical policy names")
    if first_profiles != second_profiles:
        raise ValueError("paired condition reports require identical weight profiles")

    result: dict[str, Any] = {}
    for policy_name in first_policies:
        first_rows = first_policies[policy_name].get("scenario_outcomes")
        second_rows = second_policies[policy_name].get("scenario_outcomes")
        if not isinstance(first_rows, list) or not isinstance(second_rows, list):
            raise ValueError(  # noqa: TRY004 - serialized report contract
                "condition reports lack compact scenario outcomes"
            )
        if [row.get("scenario_key") for row in first_rows] != first_keys or [
            row.get("scenario_key") for row in second_rows
        ] != second_keys:
            raise ValueError("compact scenario outcomes do not align with report keys")

        metrics: dict[str, dict[str, float | list[float] | None]] = {}
        for metric_name in ("arrived", "on_time_arrival"):
            differences = [
                float(bool(second_row[metric_name])) - float(bool(first_row[metric_name]))
                for first_row, second_row in zip(first_rows, second_rows, strict=True)
            ]
            metrics[metric_name] = {
                "mean_second_minus_first": float(np.mean(differences)),
                "paired_approx_normal_interval95": _approx_normal_mean_interval95(
                    differences, lower=-1.0, upper=1.0
                ),
            }
        for profile_name in first_profiles:
            differences = [
                float(second_row["profile_scores"][profile_name])
                - float(first_row["profile_scores"][profile_name])
                for first_row, second_row in zip(first_rows, second_rows, strict=True)
            ]
            metrics[f"score:{profile_name}"] = {
                "mean_second_minus_first": float(np.mean(differences)),
                "paired_approx_normal_interval95": _approx_normal_mean_interval95(
                    differences, lower=-1.0, upper=1.0
                ),
            }
        result[policy_name] = metrics
    return {
        "policies": result,
        "pairs": len(first_keys),
        "scope": "Paired scenario-key Monte Carlo difference conditional on both specified "
        "conditions, the fixed requests, local data models and policies; approximate normal "
        "intervals are unavailable with fewer than two pairs or zero empirical variance",
    }
