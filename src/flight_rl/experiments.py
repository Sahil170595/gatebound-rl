"""Fixed cases and reusable fitted/held-out experiment setup."""

from __future__ import annotations

import gc
import math
from datetime import date
from pathlib import Path

import numpy as np

from flight_rl.baselines import NonstopFirstPolicy, RandomPolicy, ShortestScheduledArrivalPolicy
from flight_rl.data import load_flight_data
from flight_rl.env import FlightRouteEnv
from flight_rl.evaluation import evaluate_policy, json_value
from flight_rl.models import DEFAULT_AIRPORTS, TripRequest, utc_minutes
from flight_rl.planning import DeadlinePlannerPolicy
from flight_rl.provenance import dataset_identity
from flight_rl.split_data import SplitFlightData


def default_cases(year: int = 2025) -> dict[str, TripRequest]:
    """Six illustrative route/date requests; not a population sample."""
    cases = {}
    for name, origin, destination, month_day in (
        ("winter_west_east", "SFO", "JFK", "01-15"),
        ("summer_east_west", "JFK", "SFO", "07-15"),
        ("spring_atl_sea", "ATL", "SEA", "04-15"),
        ("autumn_dfw_bos", "DFW", "BOS", "10-15"),
        ("winter_mia_den", "MIA", "DEN", "01-22"),
        ("summer_ewr_slc", "EWR", "SLC", "07-22"),
    ):
        ready = utc_minutes(f"{year}-{month_day}T12:00:00Z")
        cases[name] = TripRequest(origin, destination, ready, ready + 720, ready + 1440)
    return cases


def load_split_data(
    fit_path: Path,
    evaluation_path: Path,
    *,
    fit_start="2020-01-01",
    fit_end="2024-12-31",
    evaluation_year=2025,
    min_support=30,
    airports=DEFAULT_AIRPORTS,
) -> tuple[SplitFlightData, dict]:
    evaluation_start, evaluation_end = f"{evaluation_year}-01-01", f"{evaluation_year}-12-31"
    if date.fromisoformat(fit_end) >= date.fromisoformat(evaluation_start):
        raise ValueError("The fit window must end before the evaluation year")
    lineage = {
        "fit": dataset_identity(Path(fit_path), fit_start, fit_end),
        "evaluation": dataset_identity(Path(evaluation_path), evaluation_start, evaluation_end),
    }
    # Keep one fitted donor model, with only a tiny unused schedule slice, across all cases.
    fitted = load_flight_data(
        Path(fit_path),
        schedule_start=fit_start,
        schedule_end=fit_start,
        fit_start=fit_start,
        fit_end=fit_end,
        airports=airports,
        min_support=min_support,
    )
    # The complete evaluation catalog lets cases share a model rather than duplicate fit frames.
    observed = load_flight_data(
        Path(evaluation_path),
        schedule_start=evaluation_start,
        schedule_end=evaluation_end,
        fit_start=evaluation_start,
        fit_end=evaluation_end,
        airports=airports,
        min_support=min_support,
    )
    return SplitFlightData(observed, fitted, observed), lineage


def benchmark_case(
    data,
    request: TripRequest,
    *,
    episodes: int,
    seed: int,
    policies=("random", "nonstop_first", "shortest_scheduled", "deadline_planner"),
    dependence=0.0,
    max_candidates=64,
    trace_count=None,
) -> dict:
    from flight_rl.scenarios import ScenarioData, ScenarioPoolCache

    results = {}
    scenario_pools = ScenarioPoolCache(data)
    for name in policies:
        shared = None
        if name == "nonstop_first":
            shared = NonstopFirstPolicy()
        elif name == "shortest_scheduled":
            shared = ShortestScheduledArrivalPolicy()
        elif name == "deadline_planner":
            shared = DeadlinePlannerPolicy(request, data, max_candidates=max_candidates)
        elif name != "random":
            raise ValueError(f"Unsupported policy {name}")
        counter = 0

        def env_factory():
            nonlocal counter
            scenario = ScenarioData(
                scenario_pools, scenario_seed=seed + counter, dependence=dependence
            )
            counter += 1
            return FlightRouteEnv(request, scenario, max_candidates=max_candidates)

        def policy_factory(_env, policy_seed, shared=shared):
            return shared if shared is not None else RandomPolicy(policy_seed)

        result = evaluate_policy(
            env_factory,
            policy_factory,
            episodes=episodes,
            seed=seed,
            trace_count=episodes if trace_count is None else trace_count,
        )
        if shared is not None and hasattr(shared, "diagnostics"):
            result["policy_details"] = shared.diagnostics()
        results[name] = result
        print(
            f"{name}: deadline={result['on_time_arrival_rate']:.3f} "
            f"arrival={result['arrival_rate']:.3f} invalid={result['invalid_records']}",
            flush=True,
        )
        shared = None
        del policy_factory
        gc.collect()
    return {
        "request": json_value(request),
        "seed": seed,
        "dependence": dependence,
        "max_candidates": max_candidates,
        "results": results,
        "paired_comparisons": paired_comparisons(results),
    }


def paired_comparisons(results: dict) -> dict:
    """Monte Carlo SE for paired differences, conditional on one fixed case/model.

    The standard error is descriptive; no confidence interval or significance claim is made.
    Full traces are required rather than pretending a truncated trace is the full experiment.
    """
    if "nonstop_first" not in results:
        return {}
    reference = results["nonstop_first"]
    result = {}
    for name, current in results.items():
        if name == "nonstop_first":
            continue
        n = current["episodes"]
        if (
            n != reference["episodes"]
            or len(current["traces"]) != n
            or len(reference["traces"]) != n
        ):
            continue
        if [t["seed"] for t in current["traces"]] != [t["seed"] for t in reference["traces"]]:
            raise ValueError("Paired comparison requires aligned scenario seeds")
        differences = np.asarray(
            [
                int(a["metrics"]["on_time_arrival"]) - int(b["metrics"]["on_time_arrival"])
                for a, b in zip(current["traces"], reference["traces"], strict=True)
            ],
            dtype=float,
        )
        result[name + "_minus_nonstop_first"] = {
            "mean_deadline_difference": float(differences.mean()),
            "monte_carlo_standard_error": float(differences.std(ddof=1) / math.sqrt(n))
            if n > 1
            else None,
            "discordant_pairs": int(np.count_nonzero(differences)),
            "pairs": n,
            "scope": "Flight-keyed common scenarios; fixed request and empirical transition model",
        }
    return result
