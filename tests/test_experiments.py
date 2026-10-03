from pathlib import Path

import pytest

from flight_rl.experiments import benchmark_case, default_cases, load_split_data, paired_comparisons
from flight_rl.fixtures import make_demo_scenario


def report(values, *, seeds=None, episodes=None):
    return {
        "episodes": len(values) if episodes is None else episodes,
        "traces": [
            {"seed": seed, "metrics": {"on_time_arrival": value}}
            for seed, value in zip(
                range(len(values)) if seeds is None else seeds, values, strict=True
            )
        ],
    }


def test_fit_window_overlap_is_rejected_before_loading_files():
    with pytest.raises(ValueError, match="fit window"):
        load_split_data(Path("missing-fit"), Path("missing-eval"), fit_end="2025-01-01")


def test_cases_cover_directions_seasons_and_have_original_budgets():
    cases = default_cases()
    assert len(cases) == 6
    assert {(x.origin, x.destination) for x in cases.values()} >= {("SFO", "JFK"), ("JFK", "SFO")}
    assert all(x.deadline_utc - x.ready_utc == 720 for x in cases.values())
    assert all(x.horizon_utc - x.ready_utc == 1440 for x in cases.values())
    assert cases == default_cases(2025)


def test_paired_difference_counts_discordance_and_uncertainty():
    value = paired_comparisons(
        {
            "nonstop_first": report([True, False, False, True]),
            "learned": report([True, True, False, False]),
        }
    )["learned_minus_nonstop_first"]
    assert value["mean_deadline_difference"] == 0
    assert value["discordant_pairs"] == 2
    assert value["monte_carlo_standard_error"] == pytest.approx((2 / 3) ** 0.5 / 2)


def test_paired_comparison_refuses_misaligned_or_truncated_results():
    with pytest.raises(ValueError, match="aligned"):
        paired_comparisons({"nonstop_first": report([True]), "random": report([False], seeds=[2])})
    assert (
        paired_comparisons(
            {"nonstop_first": report([True], episodes=10), "random": report([False], episodes=10)}
        )
        == {}
    )


def test_actual_benchmark_is_reproducible_and_uses_full_paired_scenarios():
    request, data = make_demo_scenario()
    config = {"episodes": 12, "seed": 87, "policies": ("random", "nonstop_first")}
    first = benchmark_case(data, request, **config)
    second = benchmark_case(data, request, **config)
    assert first == second
    assert first["paired_comparisons"]["random_minus_nonstop_first"]["pairs"] == 12
    for value in first["results"].values():
        assert value["invalid_records"] == 0
    # Every shared selected flight has the same outcome, regardless of policy's action order.
    for a, b in zip(
        first["results"]["random"]["traces"],
        first["results"]["nonstop_first"]["traces"],
        strict=True,
    ):
        pool = {leg["flight"]["flight_id"]: leg["outcome"] for leg in a["record"]["legs"]}
        for leg in b["record"]["legs"]:
            if leg["flight"]["flight_id"] in pool:
                assert leg["outcome"] == pool[leg["flight"]["flight_id"]]
