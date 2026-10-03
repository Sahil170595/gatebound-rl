#!/usr/bin/env python
"""Evaluate frozen fitted policies against a disjoint historical transition year."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flight_rl.evaluation import write_report
from flight_rl.experiments import benchmark_case, default_cases, load_split_data
from flight_rl.provenance import source_identity
from flight_rl.verifier import PRIMARY_SCORE_PROFILE


def main():
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
    parser.add_argument(
        "--deadline-hours",
        type=float,
        default=12,
        help="Hours from ready time to deadline; horizon remains 24 hours",
    )
    parser.add_argument("--seed", type=int, default=53000)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["random", "nonstop_first", "shortest_scheduled", "deadline_planner"],
    )
    parser.add_argument("--dependence", type=float, default=0.0)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path("results/heldout_2025.json"))
    args = parser.parse_args()
    if not math.isfinite(args.deadline_hours) or not 0 < args.deadline_hours <= 24:
        parser.error("--deadline-hours must be finite, positive and at most 24")
    root = Path(__file__).resolve().parents[1]
    source = source_identity(root)
    cases = {
        name: replace(request, deadline_utc=request.ready_utc + round(args.deadline_hours * 60))
        for name, request in default_cases(args.evaluation_year).items()
    }
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
    reports = {}
    for name in selected:
        print(f"CASE {name}", flush=True)
        case_seed = args.seed + 10000 * list(cases).index(name)
        reports[name] = benchmark_case(
            data,
            cases[name],
            episodes=args.episodes,
            seed=case_seed,
            policies=tuple(args.policies),
            dependence=args.dependence,
            max_candidates=args.max_candidates,
        )
    if source_identity(root) != source:
        raise RuntimeError("Source changed during evaluation; rerun from a stable snapshot")
    write_report(
        args.out,
        {
            "source_identity": source,
            "score_profile": PRIMARY_SCORE_PROFILE,
            "lineage": lineage,
            "data_metadata": dict(data.metadata),
            "cases": reports,
            "command_arguments": sys.argv[1:],
            "evaluation_scope": f"{args.evaluation_year} empirical transition pools; policy priors/history fitted only "
            "on the preceding fit window. Fixed cases; independent or synthetic coupled "
            "flight-keyed draws. Not factual passenger itineraries, seat availability or a causal estimate.",
        },
    )
    print(f"Report: {args.out.resolve()}")


if __name__ == "__main__":
    main()
