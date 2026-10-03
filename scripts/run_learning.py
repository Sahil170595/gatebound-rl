#!/usr/bin/env python
"""Train three small policies on the fit simulator, then freeze and test on 2025."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flight_rl.baselines import NonstopFirstPolicy
from flight_rl.data import load_flight_data
from flight_rl.env import FlightRouteEnv
from flight_rl.evaluation import evaluate_policy, json_value, write_report
from flight_rl.experiments import default_cases, paired_comparisons
from flight_rl.learning import MaskedLinearPolicy, TrainingConfig, train_reinforce
from flight_rl.provenance import dataset_identity, source_identity
from flight_rl.scenarios import ScenarioData, ScenarioPoolCache
from flight_rl.split_data import SplitFlightData
from flight_rl.verifier import PRIMARY_SCORE_PROFILE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-data", type=Path, default=Path("data/processed/bts_v1_default_airports")
    )
    parser.add_argument(
        "--evaluation-data", type=Path, default=Path("data/processed/bts2025_default_airports")
    )
    parser.add_argument("--training-episodes", type=int, default=1500)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[71000, 81000, 91000])
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--evaluation-seed", type=int, default=53000)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path("results/learning_2025.json"))
    args = parser.parse_args()
    if len(set(args.training_seeds)) != len(args.training_seeds) or any(
        s < 0 for s in args.training_seeds
    ):
        parser.error("training seeds must be unique nonnegative integers")
    root = Path(__file__).resolve().parents[1]
    source = source_identity(root)
    fit_lineage = dataset_identity(args.fit_data, "2020-01-01", "2024-12-31")
    config = TrainingConfig(episodes=args.training_episodes)
    training_cases = default_cases(2024)
    requests = list(training_cases.values())
    print("Loading fit data and 2024 training schedule", flush=True)
    fitted = load_flight_data(
        args.fit_data,
        schedule_start="2024-01-01",
        schedule_end="2024-12-31",
        fit_start="2020-01-01",
        fit_end="2024-12-31",
    )
    training = {}
    for seed in args.training_seeds:

        def training_env(index, _environment_seed):
            return FlightRouteEnv(
                requests[index % len(requests)],
                fitted,
                max_candidates=args.max_candidates,
                reward_mode="on_time_arrival",
            )

        policy, report = train_reinforce(training_env, seed=seed, config=config)
        report["case_episode_counts"] = {
            name: sum(i % len(requests) == position for i in range(config.episodes))
            for position, name in enumerate(training_cases)
        }
        report["reward_mode"] = "on_time_arrival"
        report["source_identity"] = source
        report["fit_lineage"] = fit_lineage
        training[str(seed)] = report
        if source_identity(root) != source:
            raise RuntimeError("Source changed during training")
        write_report(args.out.with_name(f"{args.out.stem}_train_{seed}.json"), report)
        del policy

    # The first held-out read occurs only after every model is trained and saved.
    evaluation_lineage = dataset_identity(args.evaluation_data, "2025-01-01", "2025-12-31")
    print("All models frozen; loading 2025 outcomes", flush=True)
    observed = load_flight_data(
        args.evaluation_data,
        schedule_start="2025-01-01",
        schedule_end="2025-12-31",
        fit_start="2025-01-01",
        fit_end="2025-12-31",
    )
    split = SplitFlightData(observed, fitted, observed)
    scenario_pools = ScenarioPoolCache(split)
    cases = {}
    for case_index, (name, request) in enumerate(default_cases(2025).items()):
        seed = args.evaluation_seed + 10000 * case_index
        results = {}
        for policy_name in [
            "nonstop_first",
            "zero_weights",
            *[f"learned_{s}" for s in args.training_seeds],
        ]:
            counter = 0

            def env_factory(request=request, seed=seed):
                nonlocal counter
                world = ScenarioData(scenario_pools, seed + counter)
                counter += 1
                return FlightRouteEnv(request, world, max_candidates=args.max_candidates)

            def policy_factory(_env, policy_seed, policy_name=policy_name):
                if policy_name == "nonstop_first":
                    return NonstopFirstPolicy()
                if policy_name == "zero_weights":
                    return MaskedLinearPolicy(seed=policy_seed)
                model = training[policy_name.removeprefix("learned_")]["final_model"]
                return MaskedLinearPolicy.from_dict(model, seed=policy_seed)

            results[policy_name] = evaluate_policy(
                env_factory,
                policy_factory,
                episodes=args.evaluation_episodes,
                seed=seed,
                trace_count=args.evaluation_episodes,
            )
            print(
                f"{name}/{policy_name}: deadline={results[policy_name]['on_time_arrival_rate']:.3f}",
                flush=True,
            )
        cases[name] = {
            "request": json_value(request),
            "scenario_seed": seed,
            "max_candidates": args.max_candidates,
            "results": results,
            "paired_comparisons": paired_comparisons(results),
        }
    if source_identity(root) != source:
        raise RuntimeError("Source changed during learning evaluation")
    write_report(
        args.out,
        {
            "source_identity": source,
            "evaluation_score_profile": PRIMARY_SCORE_PROFILE,
            "command_arguments": sys.argv[1:],
            "lineage": {"fit": fit_lineage, "evaluation": evaluation_lineage},
            "training_cases": json_value(training_cases),
            "training": training,
            "cases": cases,
            "max_candidates": args.max_candidates,
            "scope": "Fixed 2024 training cases with 2020-2024 empirical donors; all models frozen before2025 reads. "
            "Evaluation uses2025 transitions with fit-only observations and matched flight-keyed scenarios. "
            "Three separate training seeds, conditional simulator uncertainty; no convergence or superiority claim.",
        },
    )
    print(f"Report: {args.out.resolve()}")


if __name__ == "__main__":
    main()
