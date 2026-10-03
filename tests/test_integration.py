"""End-to-end fixture evidence and result-report invariants."""

import json
from types import MappingProxyType

import pytest
from gymnasium.utils.env_checker import check_env

from flight_rl.baselines import NonstopFirstPolicy, RandomPolicy
from flight_rl.env import FlightRouteEnv
from flight_rl.evaluation import evaluate_policy, wilson_interval, write_report
from flight_rl.fixtures import make_demo_scenario


def test_gymnasium_contract():
    request, data = make_demo_scenario()
    check_env(FlightRouteEnv(request, data), skip_render_check=True)


@pytest.mark.parametrize("mode", ["deadline_first", "on_time_arrival", "legacy_six_v1", "rubric"])
def test_repeatable_end_to_end_report(mode, tmp_path):
    request, data = make_demo_scenario()

    def env_factory():
        return FlightRouteEnv(request, data, reward_mode=mode)

    def policy_factory(_env, seed):
        return RandomPolicy(seed=seed)

    report = evaluate_policy(env_factory, policy_factory, episodes=20, seed=42, trace_count=20)
    repeated = evaluate_policy(env_factory, policy_factory, episodes=20, seed=42, trace_count=20)
    assert report == repeated
    assert 0 <= report["on_time_arrival_rate"] <= report["arrival_rate"] <= 1
    assert sum(report["termination_counts"].values()) == 20
    assert report["score_profile"] == "deadline_first_v1"
    assert report["source_authenticated_episodes"] == 20
    assert all(trace["source_authentication"]["authenticated"] for trace in report["traces"])
    assert all(len(trace["verification"]["breakdown"]) == 3 for trace in report["traces"])
    assert all(
        len(trace["legacy_six_v1_verification"]["breakdown"]) == 6 for trace in report["traces"]
    )
    if mode == "on_time_arrival":
        expected_return = report["on_time_arrival_rate"]
    elif mode in {"legacy_six_v1", "rubric"}:
        expected_return = report["mean_legacy_six_v1"]
        assert report["reward_mode"] == "legacy_six_v1"
    else:
        expected_return = report["mean_score"]
        assert report["reward_mode"] == "deadline_first"
    assert report["mean_return"] == pytest.approx(expected_return)
    path = tmp_path / "report.json"
    write_report(path, report)
    assert json.loads(path.read_text()) == report
    assert not path.with_suffix(".json.part").exists()


def test_objective_change_does_not_change_dynamics():
    request, data = make_demo_scenario()

    def run(mode):
        return evaluate_policy(
            lambda: FlightRouteEnv(request, data, reward_mode=mode),
            lambda _env, _seed: NonstopFirstPolicy(),
            episodes=20,
            seed=8,
            trace_count=20,
        )

    legacy, deadline, deadline_first = (
        run("legacy_six_v1"),
        run("on_time_arrival"),
        run("deadline_first"),
    )
    records = [x["record"] for x in legacy["traces"]]
    assert records == [x["record"] for x in deadline["traces"]]
    assert records == [x["record"] for x in deadline_first["traces"]]
    assert legacy["termination_counts"] == deadline["termination_counts"]
    assert legacy["termination_counts"] == deadline_first["termination_counts"]


def test_intervals_cover_boundary_rates_without_false_certainty():
    lower, upper = wilson_interval(0, 100)
    assert lower == pytest.approx(0)
    assert 0 < upper < 0.05
    lower, upper = wilson_interval(100, 100)
    assert 0.95 < lower < 1
    assert upper == pytest.approx(1)


def test_data_metadata_mapping_is_serializable(tmp_path):
    payload = {"metadata": MappingProxyType({"donor_rows": 50987, "window": [2020, 2020]})}
    path = tmp_path / "report.json"
    write_report(path, payload)
    assert json.loads(path.read_text())["metadata"]["donor_rows"] == 50987


def test_evaluation_rejects_outcome_absent_from_trusted_source(monkeypatch):
    request, data = make_demo_scenario()
    monkeypatch.setattr(type(data), "source_transition_outcome", lambda *args: None)
    with pytest.raises(RuntimeError, match="failed source authentication"):
        evaluate_policy(
            lambda: FlightRouteEnv(request, data),
            lambda _env, _seed: NonstopFirstPolicy(),
            episodes=1,
            seed=42,
        )
