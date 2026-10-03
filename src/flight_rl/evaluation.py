"""Reproducible policy comparisons and explicitly scoped Monte Carlo intervals."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def wilson_interval(successes: int, total: int) -> list[float]:
    """95% Wilson interval for an IID Bernoulli Monte Carlo estimate."""
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("Require 0 <= successes <= total and total > 0")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def evaluate_policy(
    env_factory: Callable[[], Any],
    policy_factory: Callable[[Any, int], Any],
    *,
    episodes: int,
    seed: int,
    trace_count: int = 2,
) -> dict[str, Any]:
    """One fixed request/policy, independently seeded environment episodes.

    Intervals condition on the fixed simulator and request. They do not measure historical-model
    uncertainty or independent training runs; matching seeds is not a paired-disturbance guarantee.
    """
    from flight_rl.source_auth import SourceBackedVerifier
    from flight_rl.verifier import (
        PRIMARY_SCORE_PROFILE,
        episode_metrics,
        verify_legacy_six_v1,
    )

    if episodes < 1 or trace_count < 0:
        raise ValueError("episodes must be positive and trace_count nonnegative")
    scores: list[float] = []
    legacy_scores: list[float] = []
    metrics_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    rewards: list[float] = []
    reasons: dict[str, int] = {}
    reward_modes: set[str] = set()
    for i in range(episodes):
        env = env_factory()
        try:
            policy = policy_factory(env, seed + 1_000_003 + i)
            obs, _info = env.reset(seed=seed + i)
            total_reward = 0.0
            actions: list[int] = []
            for _ in range(env.request.max_attempts + 1):
                action = policy.act(obs)
                actions.append(action)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break
            else:
                raise RuntimeError("Environment failed to finish within the task attempt budget")
            record = env.record
            source_verified = SourceBackedVerifier(env.data).verify(record)
            if not source_verified.authentication.authenticated:
                raise RuntimeError(
                    f"Episode {i} failed source authentication: "
                    f"{source_verified.authentication.reason}"
                )
            verified = source_verified.score
            legacy_verified = verify_legacy_six_v1(record)
            metrics = episode_metrics(record)
            scores.append(float(verified.aggregate_score))
            legacy_scores.append(float(legacy_verified.aggregate_score))
            metrics_rows.append(metrics)
            rewards.append(total_reward)
            reward_modes.add(str(info["reward_mode"]))
            reason = record.termination_reason
            reasons[reason] = reasons.get(reason, 0) + 1
            if i < trace_count:
                trace = {
                    "episode": i,
                    "seed": seed + i,
                    "actions": actions,
                    "record": record,
                    "verification": verified,
                    "source_authentication": source_verified.authentication,
                    "legacy_six_v1_verification": legacy_verified,
                    "metrics": metrics,
                    "return": total_reward,
                }
                traces.append(json_value(trace))
        finally:
            env.close()
    arrivals = sum(bool(x["arrived"]) for x in metrics_rows)
    on_time = sum(bool(x["on_time_arrival"]) for x in metrics_rows)
    successful_elapsed = [x["elapsed_min"] for x in metrics_rows if x["arrived"]]
    if len(reward_modes) != 1:
        raise RuntimeError("Environment factory returned inconsistent reward modes")
    return {
        "episodes": episodes,
        "seed": seed,
        "score_profile": PRIMARY_SCORE_PROFILE,
        "source_authenticated_episodes": episodes,
        "verification_scope": "Canonical schedule and eligible transition-donor payloads; "
        "not RNG-draw provenance",
        "reward_mode": next(iter(reward_modes)),
        "arrived": arrivals,
        "on_time_arrivals": on_time,
        "invalid_records": sum(not bool(x["validity"]) for x in metrics_rows),
        "arrival_rate": arrivals / episodes,
        "on_time_arrival_rate": on_time / episodes,
        "arrival_rate_interval95": wilson_interval(arrivals, episodes),
        "on_time_arrival_rate_interval95": wilson_interval(on_time, episodes),
        "mean_score": float(np.mean(scores)),
        "mean_legacy_six_v1": float(np.mean(legacy_scores)),
        "mean_return": float(np.mean(rewards)),
        "mean_elapsed_given_arrival_min": float(np.mean(successful_elapsed))
        if successful_elapsed
        else None,
        "disruption_rate": sum(bool(x["disrupted"]) for x in metrics_rows) / episodes,
        "termination_counts": reasons,
        "traces": traces,
        "interval_scope": "IID simulator draws conditional on this fixed request and data source; "
        "not historical-model, training, or causal uncertainty",
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(json_value(payload), indent=2, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(path)
