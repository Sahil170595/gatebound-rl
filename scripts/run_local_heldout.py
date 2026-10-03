#!/usr/bin/env python
"""Evaluate a local Ollama policy on two fixed 2025 cases."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flight_rl.baselines import NonstopFirstPolicy
from flight_rl.env import FlightRouteEnv
from flight_rl.evaluation import evaluate_policy, write_report
from flight_rl.experiments import default_cases, load_split_data, paired_comparisons
from flight_rl.llm_policy import OllamaPolicy
from flight_rl.provenance import source_identity
from flight_rl.scenarios import ScenarioData, ScenarioPoolCache
from flight_rl.verifier import PRIMARY_SCORE_PROFILE

_DEFAULT_MODEL = "qwen2.5:1.5b-instruct-q4_K_M"
_CASE_SEEDS = {
    "winter_west_east": 53_000,
    "summer_east_west": 63_000,
}
_POLICY_LABEL = "Ollama with deterministic observation-derived decision aid"


def _parse_args() -> argparse.Namespace:
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
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-s", type=float, default=60)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("results/local_policy_heldout_2025.json"))
    args = parser.parse_args()
    if not 1 <= args.episodes <= 100:
        parser.error("--episodes must be in 1..100")
    if args.evaluation_year != 2025:
        parser.error("This fixed comparison supports evaluation year 2025 only")
    return args


def _evaluate_on_scenarios(
    *,
    request: Any,
    pool_cache: ScenarioPoolCache,
    scenario_keys: tuple[int, ...],
    policy: Any,
    max_candidates: int,
) -> dict[str, Any]:
    next_scenario = 0

    def env_factory() -> FlightRouteEnv:
        nonlocal next_scenario
        scenario = ScenarioData(
            pool_cache,
            scenario_seed=scenario_keys[next_scenario],
            dependence=0.0,
        )
        next_scenario += 1
        return FlightRouteEnv(request, scenario, max_candidates=max_candidates)

    result = evaluate_policy(
        env_factory,
        lambda _env, _policy_seed: policy,
        episodes=len(scenario_keys),
        seed=scenario_keys[0],
        trace_count=len(scenario_keys),
    )
    if next_scenario != len(scenario_keys):
        raise RuntimeError("Evaluation did not consume the fixed scenario sequence")
    for trace, scenario_seed in zip(result["traces"], scenario_keys, strict=True):
        if trace["seed"] != scenario_seed:
            raise RuntimeError("Environment and scenario seed sequences are not aligned")
        trace["scenario_seed"] = scenario_seed
    return result


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    source = source_identity(root)
    cases = default_cases(2025)
    selected = {name: cases[name] for name in _CASE_SEEDS}
    data, lineage = load_split_data(
        args.fit_data,
        args.evaluation_data,
        fit_start=args.fit_start,
        fit_end=args.fit_end,
        evaluation_year=args.evaluation_year,
    )
    pool_cache = ScenarioPoolCache(data)
    case_reports: dict[str, Any] = {}
    model_artifact: dict[str, Any] | None = None

    for case_name, request in selected.items():
        case_seed = _CASE_SEEDS[case_name]
        scenario_keys = tuple(range(case_seed, case_seed + args.episodes))
        print(f"CASE {case_name} scenarios={scenario_keys[0]}..{scenario_keys[-1]}", flush=True)
        policy = OllamaPolicy(
            airports=tuple(data.airports),
            carriers=tuple(data.carriers),
            model=args.model,
            base_url=args.base_url,
            timeout_s=args.timeout_s,
            seed=args.model_seed,
        )
        initial = policy.diagnostics()
        if initial["metadata_lookup"]["status"] != "resolved":
            policy.close()
            raise RuntimeError(
                f"Requested model is not installed in the local Ollama store: {args.model!r}"
            )
        try:
            local = _evaluate_on_scenarios(
                request=request,
                pool_cache=pool_cache,
                scenario_keys=scenario_keys,
                policy=policy,
                max_candidates=args.max_candidates,
            )
        finally:
            policy.close()
        diagnostics = policy.diagnostics()
        local["policy_details"] = diagnostics
        local["policy_label"] = _POLICY_LABEL
        local["interpretation"] = (
            f"{_POLICY_LABEL}; all requested model actions were valid."
            if diagnostics["fallbacks"] == 0
            else f"{_POLICY_LABEL} plus NonstopFirst fallback; outcomes combine both paths."
        )

        baseline = _evaluate_on_scenarios(
            request=request,
            pool_cache=pool_cache,
            scenario_keys=scenario_keys,
            policy=NonstopFirstPolicy(),
            max_candidates=args.max_candidates,
        )
        results = {"ollama_policy": local, "nonstop_first": baseline}
        case_reports[case_name] = {
            "request": request,
            "scenario_keys": list(scenario_keys),
            "max_candidates": args.max_candidates,
            "same_flight_keyed_scenario_worlds": True,
            "scenario": ScenarioData(
                pool_cache, scenario_seed=case_seed, dependence=0.0
            ).diagnostics(),
            "results": results,
            "paired_comparisons": paired_comparisons(results),
        }
        current_artifact = {
            "name": diagnostics["model"],
            "digest": diagnostics["model_digest"],
            "size_bytes": diagnostics["model_size_bytes"],
            "details": diagnostics["model_details"],
        }
        if model_artifact is not None and current_artifact != model_artifact:
            raise RuntimeError("Local model identity changed between held-out cases")
        model_artifact = current_artifact
        print(
            f"  ollama: deadline={local['on_time_arrival_rate']:.3f} "
            f"raw_valid={diagnostics['raw_valid']}/{diagnostics['calls']} "
            f"fallbacks={diagnostics['fallbacks']}",
            flush=True,
        )
        print(
            f"  nonstop_first: deadline={baseline['on_time_arrival_rate']:.3f}",
            flush=True,
        )

    if source_identity(root) != source:
        raise RuntimeError("Source changed during evaluation; rerun from a stable local snapshot")
    report = {
        "score_profile": PRIMARY_SCORE_PROFILE,
        "source_identity": source,
        "command_arguments": sys.argv[1:],
        "policy_label": _POLICY_LABEL,
        "policy_method": "Fixed prompt with a deterministic decision aid from visible flight "
        "features; no model updates during evaluation. Actual prompts, outputs and generation "
        "settings are recorded for each run.",
        "max_candidates": args.max_candidates,
        "required_local_model_artifact": model_artifact,
        "lineage": lineage,
        "data_metadata": dict(data.metadata),
        "scenario_pool_cache": pool_cache.cache_info(),
        "cases": case_reports,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["gymnasium", "numpy", "pandas", "pyarrow", "requests"]
        },
        "evaluation_scope": (
            "Two fixed historical 2025 schedule cases with empirical 2025 transition "
            "pools; policy observations and deterministic decision aids use only current "
            "features fitted on 2020-2024. Each policy sees the same flight-keyed scenario "
            "seeds, though different selected flights need not share a donor. This small case "
            "and Monte Carlo sample is retrospective simulator evidence, not a causal passenger "
            "claim or population estimate."
        ),
    }
    write_report(args.out, report)
    pool_cache.clear()
    print(f"Report: {args.out.resolve()}")


if __name__ == "__main__":
    main()
