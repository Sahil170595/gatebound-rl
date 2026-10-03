#!/usr/bin/env python
"""Train and evaluate a small policy on code-owned synthetic flights, entirely offline."""

from __future__ import annotations

import argparse
import sys
from numbers import Integral
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flight_rl.baselines import NonstopFirstPolicy
from flight_rl.env import FlightRouteEnv
from flight_rl.evaluation import evaluate_policy, json_value, write_report
from flight_rl.fixtures import make_demo_scenario
from flight_rl.learning import MaskedLinearPolicy, TrainingConfig, train_reinforce
from flight_rl.provenance import source_identity

EVALUATION_SEED_GAP = 100_000


def run_synthetic_experiment(
    *, training_episodes: int = 100, evaluation_episodes: int = 100, seed: int = 42
) -> dict:
    """Freeze a trained model before drawing fresh evaluation seeds from the same fixture.

    This tests the training/evaluation workflow, not generalization to historical data.
    """
    for name, value, minimum in (
        ("training_episodes", training_episodes, 1),
        ("evaluation_episodes", evaluation_episodes, 1),
        ("seed", seed, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    identity = source_identity(ROOT)
    request, data = make_demo_scenario()
    policy, training = train_reinforce(
        lambda _index, _seed: FlightRouteEnv(request, data, reward_mode="on_time_arrival"),
        seed=seed,
        config=TrainingConfig(episodes=training_episodes),
    )
    training["reward_mode"] = "on_time_arrival"
    frozen = policy.to_dict()
    evaluation_seed = seed + training_episodes + EVALUATION_SEED_GAP
    results = {}
    for name in ("nonstop_first", "zero_weights", "learned"):

        def factory(_env, policy_seed, name=name):
            if name == "nonstop_first":
                return NonstopFirstPolicy()
            if name == "zero_weights":
                return MaskedLinearPolicy(seed=policy_seed)
            return MaskedLinearPolicy.from_dict(frozen, seed=policy_seed)

        results[name] = evaluate_policy(
            lambda: FlightRouteEnv(request, data),
            factory,
            episodes=evaluation_episodes,
            seed=evaluation_seed,
            trace_count=1,
        )
    if policy.to_dict() != frozen:
        raise RuntimeError("Evaluation changed the frozen training policy")
    if source_identity(ROOT) != identity:
        raise RuntimeError("Source changed during the synthetic experiment")
    return {
        "schema_version": 1,
        "source_identity": identity,
        "provenance": {"source": "synthetic_fixture", "historical_coverage": False},
        "request": json_value(request),
        "training_seed": seed,
        "evaluation_seed": evaluation_seed,
        "training": training,
        "results": results,
        "frozen_model_after_evaluation": policy.to_dict(),
        "scope": "New evaluation seeds in the same invented five-flight distribution. "
        "No historical holdout, convergence, policy superiority or passenger-benefit claim. "
        "Shared episode seeds do not make action-dependent draws a paired comparison.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-episodes", type=int, default=100)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("results/synthetic_learning.json"))
    args = parser.parse_args()
    try:
        report = run_synthetic_experiment(
            training_episodes=args.training_episodes,
            evaluation_episodes=args.evaluation_episodes,
            seed=args.seed,
        )
    except ValueError as exc:
        parser.error(str(exc))
    write_report(args.out, report)
    for name, result in report["results"].items():
        print(f"{name}: deadline={result['on_time_arrival_rate']:.3f}", flush=True)
    print(f"Report: {args.out}", flush=True)


if __name__ == "__main__":
    main()
