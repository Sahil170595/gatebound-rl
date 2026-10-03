#!/usr/bin/env python
"""Run a fixture or historically fitted flight policy and write auditable JSON."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flight_rl.baselines import NonstopFirstPolicy, RandomPolicy
from flight_rl.env import FlightRouteEnv, canonical_reward_mode
from flight_rl.evaluation import evaluate_policy, json_value, write_report
from flight_rl.fixtures import make_demo_scenario
from flight_rl.models import DEFAULT_AIRPORTS, TripRequest, iso_utc, utc_minutes
from flight_rl.provenance import dataset_identity, source_identity
from flight_rl.verifier import PRIMARY_SCORE_PROFILE


def main() -> None:
    source_root = Path(__file__).resolve().parents[1]
    initial_source = source_identity(source_root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", action="store_true", help="Use synthetic example; no downloads"
    )
    parser.add_argument("--data", type=Path, default=Path("data/processed/bts_v1_default_airports"))
    parser.add_argument("--origin", default="SFO")
    parser.add_argument("--destination", default="JFK")
    parser.add_argument("--ready", default="2024-01-15T12:00:00Z")
    parser.add_argument("--deadline-hours", type=float, default=12)
    parser.add_argument("--horizon-hours", type=float, default=24)
    parser.add_argument("--delay-budget-min", type=int, default=180)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--connection-min", type=int, default=45)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--fit-start", default="2020-01-01")
    parser.add_argument("--fit-end", default="2024-12-31")
    parser.add_argument("--airports", default=",".join(DEFAULT_AIRPORTS))
    parser.add_argument("--min-support", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--policy",
        choices=[
            "random",
            "nonstop_first",
            "shortest_scheduled",
            "deadline_planner",
            "basic",
            "all",
        ],
        default="basic",
    )
    parser.add_argument("--planner-branches", type=int, default=12)
    parser.add_argument("--planner-time-bin", type=int, default=15)
    parser.add_argument("--planner-outcome-bins", type=int, default=8)
    parser.add_argument(
        "--reward-mode",
        choices=["deadline_first", "on_time_arrival", "legacy_six_v1", "rubric"],
        default="deadline_first",
    )
    parser.add_argument("--trace-count", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("results/baseline.json"))
    args = parser.parse_args()
    args.reward_mode = canonical_reward_mode(args.reward_mode)
    if args.fixture:
        fixed_options = {
            "--data",
            "--origin",
            "--destination",
            "--ready",
            "--deadline-hours",
            "--horizon-hours",
            "--delay-budget-min",
            "--max-attempts",
            "--connection-min",
            "--fit-start",
            "--fit-end",
            "--airports",
            "--min-support",
        }
        supplied = {arg.split("=", 1)[0] for arg in sys.argv[1:]} & fixed_options
        if supplied:
            parser.error(
                f"--fixture uses a fixed scenario; incompatible options: {sorted(supplied)}"
            )
        request, data = make_demo_scenario()
        provenance = {"source": "synthetic_fixture", "historical_coverage": False}
    else:
        from flight_rl.data import load_flight_data

        ready = utc_minutes(args.ready)
        request = TripRequest(
            args.origin,
            args.destination,
            ready,
            ready + round(60 * args.deadline_hours),
            ready + round(60 * args.horizon_hours),
            args.max_attempts,
            args.connection_min,
            args.delay_budget_min,
        )
        # Include adjoining local dates when UTC request bounds straddle time zones.
        first = datetime.fromtimestamp((ready - 1440) * 60, UTC).date().isoformat()
        last = datetime.fromtimestamp((request.horizon_utc + 1440) * 60, UTC).date().isoformat()
        lineage = dataset_identity(args.data, args.fit_start, args.fit_end)
        data = load_flight_data(
            args.data,
            schedule_start=first,
            schedule_end=last,
            fit_start=args.fit_start,
            fit_end=args.fit_end,
            airports=tuple(x.strip() for x in args.airports.split(",")),
            min_support=args.min_support,
        )
        provenance = {
            "source": "BTS_Marketing_Carrier",
            "path": str(args.data.resolve()),
            "fit_start": args.fit_start,
            "fit_end": args.fit_end,
            "schedule_start": first,
            "schedule_end": last,
            "data_metadata": json_value(getattr(data, "metadata", {})),
            "lineage": lineage,
            "evaluation_scope": "retrospective fitted simulator; transitions sample the fit "
            "window, not held-out outcomes",
            "fit_includes_request_date_or_later": args.fit_end
            >= datetime.fromtimestamp(ready * 60, UTC).date().isoformat(),
        }
    policies = ["random", "nonstop_first"] if args.policy in {"basic", "all"} else [args.policy]
    if args.policy == "all":
        policies.extend(["shortest_scheduled", "deadline_planner"])
    results = {}
    for policy_name in policies:
        shared_policy = None
        if policy_name == "shortest_scheduled":
            from flight_rl.baselines import ShortestScheduledArrivalPolicy

            shared_policy = ShortestScheduledArrivalPolicy()
        elif policy_name == "deadline_planner":
            from flight_rl.planning import DeadlinePlannerPolicy

            shared_policy = DeadlinePlannerPolicy(
                request,
                data,
                max_candidates=args.max_candidates,
                max_branches=args.planner_branches,
                time_bin_min=args.planner_time_bin,
                outcome_bins=args.planner_outcome_bins,
            )

        def env_factory():
            return FlightRouteEnv(
                request, data, max_candidates=args.max_candidates, reward_mode=args.reward_mode
            )

        def policy_factory(_env, policy_seed, policy_name=policy_name, shared_policy=shared_policy):
            if shared_policy is not None:
                return shared_policy
            return (
                RandomPolicy(seed=policy_seed) if policy_name == "random" else NonstopFirstPolicy()
            )

        results[policy_name] = evaluate_policy(
            env_factory,
            policy_factory,
            episodes=args.episodes,
            seed=args.seed,
            trace_count=args.trace_count,
        )
        result = results[policy_name]
        if shared_policy is not None and hasattr(shared_policy, "diagnostics"):
            result["policy_details"] = shared_policy.diagnostics()
        print(
            f"{policy_name}: arrival={result['arrival_rate']:.3f} "
            f"deadline={result['on_time_arrival_rate']:.3f} "
            f"score={result['mean_score']:.3f} "
            f"legacy_six_v1={result['mean_legacy_six_v1']:.3f} "
            f"return={result['mean_return']:.3f}",
            flush=True,
        )
    if source_identity(source_root) != initial_source:
        raise RuntimeError("Source changed during evaluation; rerun from a stable local snapshot")
    payload = {
        "source_identity": initial_source,
        "command_arguments": sys.argv[1:],
        "request": json_value(request),
        "ready_iso": iso_utc(request.ready_utc),
        "score_profile": PRIMARY_SCORE_PROFILE,
        "reward_mode": args.reward_mode,
        "provenance": provenance,
        "max_candidates": args.max_candidates,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["gymnasium", "numpy", "pandas", "pyarrow"]
        },
        "results": results,
    }
    write_report(args.out, payload)
    print(f"Report: {args.out.resolve()}")


if __name__ == "__main__":
    main()
