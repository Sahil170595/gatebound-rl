#!/usr/bin/env python
"""Run a fixed prebooked itinerary on a fixture or real BTS-backed schedules."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flight_rl.env import FlightRouteEnv
from flight_rl.evaluation import json_value, write_report
from flight_rl.fixtures import DemoFlightData
from flight_rl.models import (
    DEFAULT_AIRPORTS,
    FlightCandidate,
    SampledOutcome,
    TripRequest,
    iso_utc,
    utc_minutes,
)
from flight_rl.prebooked import (
    PrebookedRouteEnv,
    plan_scheduled_itinerary,
    verify_prebooked,
)
from flight_rl.provenance import dataset_identity, source_identity
from flight_rl.verifier import episode_metrics


def _fixture() -> tuple[TripRequest, DemoFlightData, tuple[str, ...]]:
    request = TripRequest(
        "AAA",
        "CCC",
        ready_utc=0,
        deadline_utc=250,
        horizon_utc=320,
        max_attempts=2,
        min_connection_min=45,
    )
    flights = (
        FlightCandidate("inbound", "AAA", "BBB", "DEMO", 60, 120, "1970-01-01", 0, "DJF"),
        FlightCandidate("early", "BBB", "CCC", "DEMO", 165, 220, "1970-01-01", 0, "DJF"),
        FlightCandidate("later", "BBB", "CCC", "DEMO", 210, 280, "1970-01-01", 0, "DJF"),
    )
    data = DemoFlightData(
        flights,
        {
            "inbound": (SampledOutcome("inbound-late", arr_delay_min=30.0),),
            "early": (SampledOutcome("early-on-time"),),
            "later": (SampledOutcome("later-on-time"),),
        },
    )
    return request, data, ("AAA", "BBB", "CCC")


def _parse_route(parser: argparse.ArgumentParser, value: str) -> tuple[str, ...]:
    route = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    if len(route) < 2:
        parser.error("--route must contain at least origin,destination")
    return route


def _run_prebooked(
    request: TripRequest,
    data: object,
    itinerary: tuple[FlightCandidate, ...],
    *,
    episodes: int,
    seed: int,
    trace_count: int,
) -> tuple[dict[str, object], tuple[object, ...]]:
    scores: list[float] = []
    records: list[object] = []
    traces: list[dict[str, object]] = []
    terminations: Counter[str] = Counter()
    actual_prefixes: Counter[str] = Counter()
    misses = 0
    on_time = 0
    authenticated = 0
    valid = 0
    for episode_index in range(episodes):
        with PrebookedRouteEnv(request, data, itinerary) as env:
            env.reset(seed=seed + episode_index)
            for _ in range(request.max_attempts + 1):
                _, _, terminated, truncated, _ = env.step(0)
                if terminated or truncated:
                    break
            else:
                raise RuntimeError("prebooked environment exceeded the request attempt budget")
            record = env.record
        verification = verify_prebooked(record, data)
        metrics = episode_metrics(record.episode)
        records.append(record)
        scores.append(float(verification.score.aggregate_score))
        terminations[record.termination_reason] += 1
        actual_prefixes[" -> ".join(leg.flight.flight_id for leg in record.episode.legs)] += 1
        misses += int(record.termination_reason == "missed_connection")
        authenticated += int(verification.source_authenticated)
        valid += int(verification.validity)
        on_time += int(
            verification.validity
            and verification.source_authenticated
            and bool(metrics["on_time_arrival"])
        )
        if episode_index < trace_count:
            traces.append(
                json_value(
                    {
                        "episode": episode_index,
                        "seed": seed + episode_index,
                        "actual_prefix": tuple(leg.flight for leg in record.episode.legs),
                        "record": record,
                        "verification": verification,
                        "core_metrics": metrics,
                    }
                )
            )
    result: dict[str, object] = {
        "episodes": episodes,
        "seed": seed,
        "score_profile": "deadline_first_v1",
        "committed_route": json_value(itinerary),
        "committed_flight_ids": [flight.flight_id for flight in itinerary],
        "missed_connections": misses,
        "missed_connection_rate": misses / episodes,
        "on_time_arrivals": on_time,
        "deadline_rate": on_time / episodes,
        "mean_score": fmean(scores),
        "valid_records": valid,
        "source_authenticated_records": authenticated,
        "source_authenticated": authenticated == episodes,
        "termination_counts": dict(terminations),
        "actual_prefix_counts": dict(actual_prefixes),
        "traces": traces,
        "boarding_assumption": "Published scheduled departure is the fixed boarding cutoff; "
        "equality is catchable, and an unobserved future outbound delay cannot rescue a miss. "
        "This is not observed passenger boarding or seat inventory.",
    }
    return result, tuple(records)


def _adaptive_fixture_comparison(
    request: TripRequest, data: DemoFlightData, *, seed: int
) -> dict[str, object]:
    with FlightRouteEnv(request, data, max_candidates=8, reward_mode="deadline_first") as env:
        env.reset(seed=seed)
        for _ in range(request.max_attempts + 1):
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
        else:
            raise RuntimeError("adaptive fixture exceeded the request attempt budget")
        record = env.record

    from flight_rl.source_auth import SourceBackedVerifier

    verification = SourceBackedVerifier(data).verify(record, profile="deadline_first")
    return json_value(
        {
            "actual_route": tuple(leg.flight for leg in record.legs),
            "donor_ids": [leg.outcome.donor_id for leg in record.legs],
            "termination_reason": record.termination_reason,
            "deadline_arrival": bool(episode_metrics(record)["on_time_arrival"]),
            "source_authenticated": verification.authentication.authenticated,
            "score": verification.score,
        }
    )


def main() -> None:
    source_root = Path(__file__).resolve().parents[1]
    initial_source = source_identity(source_root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="store_true", help="Use the fixed synthetic miss case")
    parser.add_argument("--data", type=Path, default=Path("data/processed/bts_v1_default_airports"))
    parser.add_argument("--route", default="SFO,JFK", help="Comma-separated committed airport path")
    parser.add_argument(
        "--time",
        "--ready",
        dest="ready",
        default="2024-01-15T12:00:00Z",
        help="Passenger-ready timestamp with UTC offset",
    )
    parser.add_argument("--deadline-hours", type=float, default=12.0)
    parser.add_argument("--horizon-hours", type=float, default=24.0)
    parser.add_argument("--delay-budget-min", type=int, default=180)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--connection-min", type=int, default=45)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--max-search-nodes", type=int, default=10_000)
    parser.add_argument("--fit-start", default="2020-01-01")
    parser.add_argument("--fit-end", default="2024-12-31")
    parser.add_argument("--airports", default=",".join(DEFAULT_AIRPORTS))
    parser.add_argument("--min-support", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace-count", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("results/prebooked.json"))
    args = parser.parse_args()
    if args.episodes < 1 or args.trace_count < 0:
        parser.error("--episodes must be positive and --trace-count nonnegative")

    if args.fixture:
        incompatible = {
            "--data",
            "--route",
            "--time",
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
        supplied = {argument.split("=", 1)[0] for argument in sys.argv[1:]}
        conflicts = sorted(supplied & incompatible)
        if conflicts:
            parser.error(f"--fixture uses a fixed scenario; incompatible options: {conflicts}")
        request, data, route = _fixture()
        provenance: dict[str, object] = {
            "source": "synthetic_fixture",
            "historical_coverage": False,
        }
    else:
        from flight_rl.data import load_flight_data

        route = _parse_route(parser, args.route)
        ready = utc_minutes(args.ready)
        request = TripRequest(
            route[0],
            route[-1],
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
        selected_airports = {
            part.strip().upper() for part in args.airports.split(",") if part.strip()
        }
        selected_airports.update(route)
        data = load_flight_data(
            args.data,
            schedule_start=first,
            schedule_end=last,
            fit_start=args.fit_start,
            fit_end=args.fit_end,
            airports=tuple(sorted(selected_airports)),
            min_support=args.min_support,
        )
        provenance = {
            "source": "BTS_Marketing_Carrier",
            "path": str(args.data.resolve()),
            "schedule_start": first,
            "schedule_end": last,
            "fit_start": args.fit_start,
            "fit_end": args.fit_end,
            "data_metadata": json_value(getattr(data, "metadata", {})),
            "lineage": lineage,
            "evaluation_scope": "retrospective fitted simulator; transitions sample the fit "
            "window, not held-out outcomes",
        }

    itinerary = plan_scheduled_itinerary(
        request,
        data,
        route=route,
        max_candidates=args.max_candidates,
        max_search_nodes=args.max_search_nodes,
    )
    results, records = _run_prebooked(
        request,
        data,
        itinerary,
        episodes=args.episodes,
        seed=args.seed,
        trace_count=args.trace_count,
    )
    if args.fixture:
        adaptive = _adaptive_fixture_comparison(request, data, seed=args.seed)
        first_prebooked = records[0]
        adaptive["same_inbound_draw"] = bool(
            first_prebooked.episode.legs
            and adaptive["donor_ids"]
            and first_prebooked.episode.legs[0].outcome.donor_id == adaptive["donor_ids"][0]
        )
        results["adaptive_comparison"] = adaptive

    if source_identity(source_root) != initial_source:
        raise RuntimeError("Source changed during evaluation; rerun from a stable local snapshot")
    payload = {
        "source_identity": initial_source,
        "command_arguments": sys.argv[1:],
        "request": json_value(request),
        "ready_iso": iso_utc(request.ready_utc),
        "provenance": provenance,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["gymnasium", "numpy", "pandas", "pyarrow"]
        },
        "results": results,
    }
    write_report(args.out, payload)
    print(
        f"route={' -> '.join(flight.flight_id for flight in itinerary)} "
        f"misses={results['missed_connection_rate']:.3f} "
        f"deadline={results['deadline_rate']:.3f} "
        f"score={results['mean_score']:.3f} "
        f"source_authenticated={results['source_authenticated']}",
        flush=True,
    )
    print(f"Report: {args.out.resolve()}")


if __name__ == "__main__":
    main()
