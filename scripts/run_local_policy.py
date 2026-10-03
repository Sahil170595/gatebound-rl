#!/usr/bin/env python
"""Compare one local Ollama policy with NonstopFirst on identical episode seeds."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flight_rl.baselines import NonstopFirstPolicy
from flight_rl.env import FlightRouteEnv, canonical_reward_mode
from flight_rl.evaluation import evaluate_policy, json_value, write_report
from flight_rl.fixtures import make_demo_scenario
from flight_rl.llm_policy import OllamaPolicy
from flight_rl.models import DEFAULT_AIRPORTS, TripRequest, iso_utc, utc_minutes
from flight_rl.provenance import dataset_identity, source_identity
from flight_rl.verifier import PRIMARY_SCORE_PROFILE

_DEFAULT_MODEL = "qwen2.5:1.5b-instruct-q4_K_M"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="store_true", help="Use the synthetic example")
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
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-s", type=float, default=60)
    parser.add_argument(
        "--reward-mode",
        choices=["deadline_first", "on_time_arrival", "legacy_six_v1", "rubric"],
        default="deadline_first",
    )
    parser.add_argument("--trace-count", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("results/local_policy.json"))
    return parser


def _scenario(args: argparse.Namespace) -> tuple[TripRequest, object, dict]:
    if args.fixture:
        request, data = make_demo_scenario()
        return request, data, {"source": "synthetic_fixture", "historical_coverage": False}

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
    first = datetime.fromtimestamp((ready - 1440) * 60, UTC).date().isoformat()
    last = datetime.fromtimestamp((request.horizon_utc + 1440) * 60, UTC).date().isoformat()
    lineage = dataset_identity(args.data, args.fit_start, args.fit_end)
    data = load_flight_data(
        args.data,
        schedule_start=first,
        schedule_end=last,
        fit_start=args.fit_start,
        fit_end=args.fit_end,
        airports=tuple(code.strip() for code in args.airports.split(",")),
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
        "evaluation_scope": "retrospective fitted simulator; transitions sample the fit window, "
        "not held-out outcomes",
        "fit_includes_request_date_or_later": args.fit_end
        >= datetime.fromtimestamp(ready * 60, UTC).date().isoformat(),
    }
    return request, data, provenance


def main() -> None:
    source_root = Path(__file__).resolve().parents[1]
    initial_source = source_identity(source_root)
    parser = _parser()
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
        supplied = {argument.split("=", 1)[0] for argument in sys.argv[1:]} & fixed_options
        if supplied:
            parser.error(
                f"--fixture uses a fixed scenario; incompatible options: {sorted(supplied)}"
            )

    request, data, provenance = _scenario(args)

    def env_factory() -> FlightRouteEnv:
        return FlightRouteEnv(
            request,
            data,
            max_candidates=args.max_candidates,
            reward_mode=args.reward_mode,
        )

    policy = OllamaPolicy(
        airports=tuple(data.airports),
        carriers=tuple(data.carriers),
        model=args.model,
        base_url=args.base_url,
        timeout_s=args.timeout_s,
        seed=args.seed,
    )
    initial_diagnostics = policy.diagnostics()
    if initial_diagnostics["metadata_lookup"]["status"] != "resolved":
        policy.close()
        raise RuntimeError(
            f"Requested model is not installed in the local Ollama store: {args.model!r}"
        )
    try:
        local_result = evaluate_policy(
            env_factory,
            lambda _env, _policy_seed: policy,
            episodes=args.episodes,
            seed=args.seed,
            trace_count=args.trace_count,
        )
    finally:
        policy.close()
    diagnostics = policy.diagnostics()
    local_result["policy_details"] = diagnostics
    local_result["interpretation"] = (
        "Ollama with deterministic observation-derived decision aid: every requested model "
        "action parsed and passed the current mask."
        if diagnostics["fallbacks"] == 0
        else "Ollama with deterministic observation-derived decision aid plus NonstopFirst "
        "fallback; do not attribute all outcomes to the model."
    )

    baseline_result = evaluate_policy(
        env_factory,
        lambda _env, _policy_seed: NonstopFirstPolicy(),
        episodes=args.episodes,
        seed=args.seed,
        trace_count=args.trace_count,
    )
    results = {"ollama_policy": local_result, "nonstop_first": baseline_result}
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
        "comparison_scope": "Same request and episode seed sequence. Different actions can select "
        "different empirical donor pools, so this is not paired flight-outcome evidence.",
        "required_local_model_artifact": {
            "name": diagnostics["model"],
            "digest": diagnostics["model_digest"],
            "size_bytes": diagnostics["model_size_bytes"],
            "details": diagnostics["model_details"],
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["gymnasium", "numpy", "pandas", "pyarrow", "requests"]
        },
        "results": results,
    }
    write_report(args.out, payload)
    print(
        f"ollama_policy: arrival={local_result['arrival_rate']:.3f} "
        f"deadline={local_result['on_time_arrival_rate']:.3f} "
        f"score={local_result['mean_score']:.3f} "
        f"raw_valid={diagnostics['raw_valid']}/{diagnostics['calls']} "
        f"fallbacks={diagnostics['fallbacks']}",
        flush=True,
    )
    print(
        f"nonstop_first: arrival={baseline_result['arrival_rate']:.3f} "
        f"deadline={baseline_result['on_time_arrival_rate']:.3f} "
        f"score={baseline_result['mean_score']:.3f}",
        flush=True,
    )
    print(f"Report: {args.out.resolve()}")


if __name__ == "__main__":
    main()
