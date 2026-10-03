#!/usr/bin/env python
"""Run fixed-schedule fit-year, weight, and synthetic-dependence robustness studies."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flight_rl.baselines import (
    NonstopFirstPolicy,
    RandomPolicy,
    ShortestScheduledArrivalPolicy,
)
from flight_rl.data import load_flight_data
from flight_rl.evaluation import json_value, write_report
from flight_rl.models import DEFAULT_AIRPORTS, FlightCandidate, TripRequest
from flight_rl.planning import DeadlinePlannerPolicy
from flight_rl.provenance import dataset_identity, source_identity
from flight_rl.robustness import (
    fixed_2024_cases,
    paired_condition_differences,
    run_paired_scenarios,
)
from flight_rl.scenarios import ScenarioPoolCache
from flight_rl.split_data import SplitFlightData
from flight_rl.verifier import LEGACY_SIX_V1_PROFILE

_SCHEDULE_START = "2024-01-01"
_SCHEDULE_END = "2024-12-31"
_POOLED_FIT_START = "2020-01-01"
_POOLED_FIT_END = "2024-12-31"


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def _csv_floats(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one number")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a nonnegative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return parsed


def _selected_cases(value: str, available: dict[str, TripRequest]) -> dict[str, TripRequest]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if not names or names == ("all",):
        return available
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f"Unknown robustness cases: {unknown}")
    if len(set(names)) != len(names):
        raise ValueError("Robustness case names must be unique")
    return {name: available[name] for name in names}


def _policy_names(selection: str) -> tuple[str, ...]:
    if selection == "basic":
        return ("random", "nonstop_first")
    if selection == "schedule":
        return ("random", "nonstop_first", "shortest_scheduled")
    if selection == "all":
        return ("random", "nonstop_first", "shortest_scheduled", "deadline_planner")
    return (selection,)


def _policy_factories(
    names: tuple[str, ...],
    request: TripRequest,
    data: object,
    *,
    max_candidates: int,
    planner_branches: int,
    planner_time_bin: int,
    planner_outcome_bins: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    factories: dict[str, Any] = {}
    policy_objects: dict[str, Any] = {}
    for name in names:
        if name == "random":
            factories[name] = lambda seed: RandomPolicy(seed)
        elif name == "nonstop_first":
            factories[name] = lambda _seed: NonstopFirstPolicy()
        elif name == "shortest_scheduled":
            factories[name] = lambda _seed: ShortestScheduledArrivalPolicy()
        elif name == "deadline_planner":
            planner = DeadlinePlannerPolicy(
                request,
                data,
                max_candidates=max_candidates,
                max_branches=planner_branches,
                time_bin_min=planner_time_bin,
                outcome_bins=planner_outcome_bins,
            )
            policy_objects[name] = planner
            factories[name] = lambda _seed, planner=planner: planner
        else:
            raise ValueError(f"Unsupported policy {name!r}")
    return factories, policy_objects


def _run_cases(
    *,
    data: object,
    cases: dict[str, TripRequest],
    policy_names: tuple[str, ...],
    scenario_keys: tuple[int, ...],
    dependence: float,
    max_candidates: int,
    planner_branches: int,
    planner_time_bin: int,
    planner_outcome_bins: int,
    trace_count: int,
) -> dict[str, Any]:
    case_results: dict[str, Any] = {}
    for case_name, request in cases.items():
        factories, policy_objects = _policy_factories(
            policy_names,
            request,
            data,
            max_candidates=max_candidates,
            planner_branches=planner_branches,
            planner_time_bin=planner_time_bin,
            planner_outcome_bins=planner_outcome_bins,
        )
        result = run_paired_scenarios(
            request=request,
            data=data,
            policy_factories=factories,
            scenario_keys=scenario_keys,
            dependence=dependence,
            max_candidates=max_candidates,
            trace_count=trace_count,
        )
        result["request"] = json_value(request)
        for name, policy in policy_objects.items():
            result["policies"][name]["policy_details"] = policy.diagnostics()
        case_results[case_name] = result
        print(
            f"{case_name} dependence={dependence:g}: "
            + ", ".join(
                f"{name} deadline={values['on_time_arrival_rate']:.3f}"
                for name, values in result["policies"].items()
            ),
            flush=True,
        )
    return case_results


def _load_model(
    path: Path,
    *,
    fit_start: str,
    fit_end: str,
    airports: tuple[str, ...],
    min_support: int,
    schedule_start: str = _SCHEDULE_START,
    schedule_end: str = _SCHEDULE_END,
) -> object:
    return load_flight_data(
        path,
        schedule_start=schedule_start,
        schedule_end=schedule_end,
        fit_start=fit_start,
        fit_end=fit_end,
        airports=airports,
        min_support=min_support,
    )


def _catalog_route_representatives(
    data: object, airports: tuple[str, ...]
) -> dict[tuple[str, str], FlightCandidate]:
    """Return one fixed-schedule flight for every route in the loaded catalog."""

    result: dict[tuple[str, str], FlightCandidate] = {}
    for origin in airports:
        flights = data.candidates(origin, -(1 << 62), 1 << 62, limit=(1 << 31) - 1)
        for flight in flights:
            result.setdefault((flight.origin, flight.dest), flight)
    if not result:
        raise RuntimeError("The fixed schedule contains no candidate routes")
    return result


def _common_fit_routes(
    path: Path,
    *,
    fit_years: tuple[int, ...],
    airports: tuple[str, ...],
    min_support: int,
    catalog: dict[tuple[str, str], FlightCandidate],
) -> tuple[frozenset[tuple[str, str]], dict[str, Any]]:
    """Find the fixed-schedule routes with donor coverage in every fit year."""

    common = set(catalog)
    diagnostics: dict[str, Any] = {}
    for year in fit_years:
        fit_start, fit_end = f"{year}-01-01", f"{year}-12-31"
        fitted = _load_model(
            path,
            fit_start=fit_start,
            fit_end=fit_end,
            airports=airports,
            min_support=min_support,
            schedule_start=fit_start,
            schedule_end=fit_start,
        )
        supported = {
            route for route, flight in catalog.items() if fitted.outcome_summary(flight).support > 0
        }
        unsupported = sorted(set(catalog) - supported)
        diagnostics[str(year)] = {
            "supported_fixed_schedule_routes": len(supported),
            "unsupported_fixed_schedule_routes": [list(route) for route in unsupported],
        }
        common.intersection_update(supported)
        del fitted
        gc.collect()
    if not common:
        raise RuntimeError("No fixed-schedule route has donor coverage in every requested fit year")
    return frozenset(common), diagnostics


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    initial_source = source_identity(root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/bts_v1_default_airports"))
    parser.add_argument("--out", type=Path, default=Path("results/robustness.json"))
    parser.add_argument("--episodes", type=_positive_int, default=50)
    parser.add_argument("--seed", type=int, default=47_000)
    parser.add_argument("--fit-years", type=_csv_ints, default=(2020, 2021, 2022, 2023, 2024))
    parser.add_argument("--dependences", type=_csv_floats, default=(0.0, 0.5, 1.0))
    parser.add_argument("--cases", default="all")
    parser.add_argument("--airports", default=",".join(DEFAULT_AIRPORTS))
    parser.add_argument("--min-support", type=_positive_int, default=30)
    parser.add_argument("--max-candidates", type=_positive_int, default=64)
    parser.add_argument("--pool-cache-size", type=_positive_int, default=512)
    parser.add_argument(
        "--policy",
        choices=(
            "random",
            "nonstop_first",
            "shortest_scheduled",
            "deadline_planner",
            "basic",
            "schedule",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--planner-branches", type=_positive_int, default=12)
    parser.add_argument("--planner-time-bin", type=_positive_int, default=15)
    parser.add_argument("--planner-outcome-bins", type=_nonnegative_int, default=8)
    parser.add_argument("--trace-count", type=_nonnegative_int, default=1)
    parser.add_argument("--skip-fit-year", action="store_true")
    parser.add_argument("--skip-dependence", action="store_true")
    args = parser.parse_args()

    if args.skip_fit_year and args.skip_dependence:
        parser.error("at least one robustness axis must run")
    if len(set(args.fit_years)) != len(args.fit_years) or any(
        year < 2020 or year > 2024 for year in args.fit_years
    ):
        parser.error("--fit-years must contain unique years in 2020..2024")
    if len(set(args.dependences)) != len(args.dependences) or any(
        not 0.0 <= dependence <= 1.0 for dependence in args.dependences
    ):
        parser.error("--dependences must contain unique finite values in [0,1]")
    airports = tuple(code.strip().upper() for code in args.airports.split(",") if code.strip())
    if not airports or len(set(airports)) != len(airports):
        parser.error("--airports must contain unique nonempty codes")
    try:
        cases = _selected_cases(args.cases, fixed_2024_cases())
    except ValueError as exc:
        parser.error(str(exc))

    data_path = args.data.resolve()
    scenario_keys = tuple(args.seed + index for index in range(args.episodes))
    policy_names = _policy_names(args.policy)
    started = perf_counter()
    fixed_evaluation = _load_model(
        data_path,
        fit_start=_POOLED_FIT_START,
        fit_end=_POOLED_FIT_END,
        airports=airports,
        min_support=args.min_support,
    )
    provenance = {
        "source": "BTS_Marketing_Carrier",
        "path": str(data_path),
        "lineage": dataset_identity(data_path, _POOLED_FIT_START, _POOLED_FIT_END),
        "schedule_window": [_SCHEDULE_START, _SCHEDULE_END],
        "schedule_fixed_across_fit_years": True,
        "fit_year_transition_window": [_POOLED_FIT_START, _POOLED_FIT_END],
        "fit_year_transition_note": "Every fit-year condition uses the same pooled empirical "
        "transition pools. They overlap the fitted years and are a fixed retrospective reference, "
        "not held-out outcomes.",
        "scope": "Retrospective fitted simulator over the configured airport subset; no seat "
        "inventory, causal passenger outcome, or population-random case claim.",
    }

    fit_year_runs: dict[str, Any] = {}
    fit_year_catalog: dict[str, Any] | None = None
    if not args.skip_fit_year:
        catalog = _catalog_route_representatives(fixed_evaluation, airports)
        common_routes, support_by_year = _common_fit_routes(
            data_path,
            fit_years=args.fit_years,
            airports=airports,
            min_support=args.min_support,
            catalog=catalog,
        )
        fit_year_catalog = {
            "rule": "2024 schedule restricted to routes with donor coverage in every requested "
            "fit year; schedules, ordering and legal actions are then fixed across conditions.",
            "fixed_schedule_routes": len(catalog),
            "common_support_route_count": len(common_routes),
            "common_support_routes": [list(route) for route in sorted(common_routes)],
            "excluded_from_common_support_routes": [
                list(route) for route in sorted(set(catalog) - common_routes)
            ],
            "support_by_fit_year": support_by_year,
        }
        for year in args.fit_years:
            fit_start, fit_end = f"{year}-01-01", f"{year}-12-31"
            fitted = _load_model(
                data_path,
                fit_start=fit_start,
                fit_end=fit_end,
                airports=airports,
                min_support=args.min_support,
                schedule_start=fit_start,
                schedule_end=fit_start,
            )
            split = SplitFlightData(
                fixed_evaluation,
                fitted,
                fixed_evaluation,
                candidate_routes=common_routes,
            )
            cached = ScenarioPoolCache(split, maxsize=args.pool_cache_size)
            case_results = _run_cases(
                data=cached,
                cases=cases,
                policy_names=policy_names,
                scenario_keys=scenario_keys,
                dependence=0.0,
                max_candidates=args.max_candidates,
                planner_branches=args.planner_branches,
                planner_time_bin=args.planner_time_bin,
                planner_outcome_bins=args.planner_outcome_bins,
                trace_count=args.trace_count,
            )
            fit_year_runs[str(year)] = {
                "fit_window": [fit_start, fit_end],
                "data_metadata": json_value(split.metadata),
                "cases": case_results,
                "ordered_pool_cache": cached.cache_info(),
            }
            cached.clear()
            del cached, split, fitted
            gc.collect()

    fit_year_comparisons: dict[str, Any] = {}
    if fit_year_runs and "2024" in fit_year_runs:
        for year, run in fit_year_runs.items():
            if year == "2024":
                continue
            fit_year_comparisons[f"fit_{year}_minus_fit_2024"] = {
                case_name: paired_condition_differences(
                    fit_year_runs["2024"]["cases"][case_name],
                    run["cases"][case_name],
                )
                for case_name in cases
            }

    dependence_runs: dict[str, Any] = {}
    dependence_comparisons: dict[str, Any] = {}
    if not args.skip_dependence:
        cached = ScenarioPoolCache(fixed_evaluation, maxsize=args.pool_cache_size)
        for dependence in args.dependences:
            label = format(dependence, "g")
            dependence_runs[label] = {
                "dependence": dependence,
                "fit_window": [_POOLED_FIT_START, _POOLED_FIT_END],
                "cases": _run_cases(
                    data=cached,
                    cases=cases,
                    policy_names=policy_names,
                    scenario_keys=scenario_keys,
                    dependence=dependence,
                    max_candidates=args.max_candidates,
                    planner_branches=args.planner_branches,
                    planner_time_bin=args.planner_time_bin,
                    planner_outcome_bins=args.planner_outcome_bins,
                    trace_count=args.trace_count,
                ),
            }
        if "0" in dependence_runs:
            for label, run in dependence_runs.items():
                if label == "0":
                    continue
                dependence_comparisons[f"dependence_{label}_minus_0"] = {
                    case_name: paired_condition_differences(
                        dependence_runs["0"]["cases"][case_name],
                        run["cases"][case_name],
                    )
                    for case_name in cases
                }
        dependence_cache_info = cached.cache_info()
        data_metadata = json_value(getattr(fixed_evaluation, "metadata", {}))
        cached.clear()
        del cached
        gc.collect()
    else:
        dependence_cache_info = None
        data_metadata = None

    final_source = source_identity(root)
    if final_source != initial_source:
        raise RuntimeError("Source changed during robustness execution; rerun from a stable source")
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "score_profile": LEGACY_SIX_V1_PROFILE,
        "source_identity": initial_source,
        "command_arguments": sys.argv[1:],
        "configuration": {
            "episodes_per_case_condition": args.episodes,
            "scenario_keys": list(scenario_keys),
            "scenario_key_reuse": "The same ordered keys are reused across policies, fit years "
            "and dependence values.",
            "cases": list(cases),
            "policies": list(policy_names),
            "airports": list(airports),
            "min_support": args.min_support,
            "max_candidates": args.max_candidates,
            "planner": {
                "max_branches": args.planner_branches,
                "time_bin_min": args.planner_time_bin,
                "outcome_bins": args.planner_outcome_bins,
            },
        },
        "provenance": provenance,
        "fit_year_sensitivity": {
            "runs": fit_year_runs,
            "paired_comparisons_to_2024": fit_year_comparisons,
            "candidate_catalog": fit_year_catalog,
            "interpretation": "The 2024 schedules, common-support candidate routes, pooled "
            "2020-2024 transition pools and scenario quantiles are fixed. Only policy-facing "
            "summaries and donor histories change by fit year. Planner policies are refitted by "
            "year; post-hoc weight profiles never change a policy or episode.",
        },
        "dependence_stress": {
            "runs": dependence_runs,
            "paired_comparisons_to_independence": dependence_comparisons,
            "fit_data_metadata": data_metadata,
            "ordered_pool_cache": dependence_cache_info,
            "interpretation": "Dependence is a synthetic independence/comonotonic mixture "
            "weight, not an estimated weather or operational correlation.",
        },
        "uncertainty_scope": "All intervals condition on the six fixed requests, local "
        "BTS-derived simulator, policy implementations and deterministic scenario construction. "
        "They omit historical-model, case-selection and causal uncertainty; comparisons are "
        "descriptive sensitivity analyses with no multiplicity adjustment.",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("gymnasium", "numpy", "pandas", "pyarrow")
        },
        "duration_seconds": perf_counter() - started,
    }
    write_report(args.out, payload)
    print(f"Report: {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
