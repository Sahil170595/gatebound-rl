#!/usr/bin/env python
"""Compare fixed cancellation-recovery assumptions on held-out 2025 scenarios."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flight_rl.baselines import NonstopFirstPolicy, RandomPolicy
from flight_rl.evaluation import json_value, write_report
from flight_rl.experiments import default_cases, load_split_data
from flight_rl.provenance import source_identity
from flight_rl.recovery import RecoveryConfig, RecoveryEnv, evaluate_recovery_policy
from flight_rl.scenarios import ScenarioData, ScenarioPoolCache
from flight_rl.verifier import PRIMARY_SCORE_PROFILE

CONFIGS = {
    "no_recovery": RecoveryConfig(0, 0, 0),
    "recovery_0_0": RecoveryConfig(0, 0, 2),
    "recovery_30_30": RecoveryConfig(30, 30, 2),
    "recovery_60_60": RecoveryConfig(60, 60, 2),
}
POLICIES = ("random", "nonstop_first")


def _policy_factory(name: str):
    if name == "random":
        return lambda _env, policy_seed: RandomPolicy(policy_seed)
    if name == "nonstop_first":
        return lambda _env, _policy_seed: NonstopFirstPolicy()
    raise ValueError(f"Unsupported recovery policy {name!r}")


def _paired_config_comparisons(results: dict[str, dict]) -> dict[str, dict]:
    """Describe aligned config differences without a population-inference claim."""

    reference = results["no_recovery"]["episode_outcomes"]
    reference_seeds = [row["scenario_seed"] for row in reference]
    comparisons = {}
    for name, result in results.items():
        if name == "no_recovery":
            continue
        current = result["episode_outcomes"]
        if [row["scenario_seed"] for row in current] != reference_seeds:
            raise ValueError("Recovery comparison requires identical scenario keys")
        deadline_differences = np.asarray(
            [
                int(row["on_time_arrival"]) - int(base["on_time_arrival"])
                for row, base in zip(current, reference, strict=True)
            ],
            dtype=float,
        )
        arrival_differences = np.asarray(
            [
                int(row["arrived"]) - int(base["arrived"])
                for row, base in zip(current, reference, strict=True)
            ],
            dtype=float,
        )
        attempt_differences = np.asarray(
            [
                int(row["attempts"]) - int(base["attempts"])
                for row, base in zip(current, reference, strict=True)
            ],
            dtype=float,
        )
        n = len(current)
        comparisons[f"{name}_minus_no_recovery"] = {
            "pairs": n,
            "mean_deadline_arrival_difference": float(deadline_differences.mean()),
            "mean_arrival_difference": float(arrival_differences.mean()),
            "mean_attempt_difference": float(attempt_differences.mean()),
            "deadline_improved_pairs": int(np.count_nonzero(deadline_differences > 0)),
            "deadline_worsened_pairs": int(np.count_nonzero(deadline_differences < 0)),
            "deadline_discordant_pairs": int(np.count_nonzero(deadline_differences)),
            "deadline_difference_monte_carlo_se": (
                float(deadline_differences.std(ddof=1) / math.sqrt(n)) if n > 1 else None
            ),
            "scope": "Matched flight-keyed scenario worlds for one fixed request; recovery "
            "assumes a seat is available after each eligible cancellation.",
        }
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-data", type=Path, default=Path("data/processed/bts_v1_default_airports")
    )
    parser.add_argument(
        "--evaluation-data", type=Path, default=Path("data/processed/bts2025_default_airports")
    )
    parser.add_argument("--fit-start", default="2020-01-01")
    parser.add_argument("--fit-end", default="2024-12-31")
    parser.add_argument("--evaluation-year", type=int, default=2025)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=53000)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--policies", nargs="+", default=list(POLICIES))
    parser.add_argument("--dependence", type=float, default=0.0)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--trace-count", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results/recovery_2025.json"))
    args = parser.parse_args()
    if args.episodes < 1 or args.trace_count < 0 or args.max_candidates < 1:
        parser.error("episodes/max-candidates must be positive and trace-count nonnegative")
    if not 0.0 <= args.dependence <= 1.0:
        parser.error("dependence must be in [0, 1]")
    if len(args.policies) != len(set(args.policies)) or any(
        name not in POLICIES for name in args.policies
    ):
        parser.error(f"Policies must be unique members of {list(POLICIES)}")

    root = Path(__file__).resolve().parents[1]
    source = source_identity(root)
    cases = default_cases(args.evaluation_year)
    selected = args.cases or list(cases)
    if len(selected) != len(set(selected)) or any(name not in cases for name in selected):
        parser.error("Case names must be unique members of the fixed case set")

    data, lineage = load_split_data(
        args.fit_data,
        args.evaluation_data,
        fit_start=args.fit_start,
        fit_end=args.fit_end,
        evaluation_year=args.evaluation_year,
    )
    scenario_pools = ScenarioPoolCache(data)
    reports = {}
    for case_name in selected:
        print(f"CASE {case_name}", flush=True)
        case_request = cases[case_name]
        case_seed = args.seed + 10_000 * list(cases).index(case_name)
        policy_reports = {}
        for policy_name in args.policies:
            print(f"  POLICY {policy_name}", flush=True)
            config_results = {}
            for config_name, config in CONFIGS.items():

                def env_factory(
                    scenario_seed: int,
                    config=config,
                    case_request=case_request,
                ):
                    scenario = ScenarioData(
                        scenario_pools,
                        scenario_seed=scenario_seed,
                        dependence=args.dependence,
                    )
                    return RecoveryEnv(
                        case_request,
                        scenario,
                        config,
                        max_candidates=args.max_candidates,
                    )

                result = evaluate_recovery_policy(
                    env_factory,
                    _policy_factory(policy_name),
                    episodes=args.episodes,
                    seed=case_seed,
                    trace_count=args.trace_count,
                )
                config_results[config_name] = result
                print(
                    f"    {config_name}: deadline={result['on_time_arrival_rate']:.3f} "
                    f"arrival={result['arrival_rate']:.3f} failures={result['failures']} "
                    f"attempts={result['total_attempts']} "
                    f"rebookings={result['total_rebookings']} "
                    f"invalid={result['invalid_records']}",
                    flush=True,
                )
            policy_reports[policy_name] = {
                "configs": config_results,
                "paired_config_comparisons": _paired_config_comparisons(config_results),
            }
        reports[case_name] = {
            "request": json_value(case_request),
            "seed": case_seed,
            "policies": policy_reports,
        }

    if source_identity(root) != source:
        raise RuntimeError(
            "Source changed during recovery evaluation; rerun from a stable snapshot"
        )
    write_report(
        args.out,
        {
            "source_identity": source,
            "score_profile": PRIMARY_SCORE_PROFILE,
            "lineage": lineage,
            "data_metadata": dict(data.metadata),
            "scenario_pool_cache": scenario_pools.cache_info(),
            "configs": {name: asdict(config) for name, config in CONFIGS.items()},
            "episodes_per_case_policy_config": args.episodes,
            "trace_count_per_case_policy_config": args.trace_count,
            "dependence": args.dependence,
            "max_candidates": args.max_candidates,
            "selected_cases": selected,
            "selected_policies": args.policies,
            "cases": reports,
            "command_arguments": sys.argv[1:],
            "evaluation_scope": f"{args.evaluation_year} empirical transition pools with "
            "2020-2024 fitted priors/history. Six fixed requests and fixed delay assumptions, "
            "with no tuning from future outcomes. Recoveries assume an available seat after "
            "eligible cancellation; results are not factual passenger itineraries, source "
            "authentication, population estimates, or causal effects.",
        },
    )
    print(f"Report: {args.out.resolve()}")


if __name__ == "__main__":
    main()
