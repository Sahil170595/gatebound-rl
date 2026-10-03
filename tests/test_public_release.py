"""Offline reproduction contracts for the public synthetic training example."""

from pathlib import Path

import pytest

from scripts.run_decision_lab import RequestError, build_runtime, load_cases, parse_args, replay
from scripts.run_synthetic import run_synthetic_experiment


def test_synthetic_training_and_frozen_evaluation_are_repeatable():
    arguments = {"training_episodes": 12, "evaluation_episodes": 8, "seed": 42}
    first = run_synthetic_experiment(**arguments)
    second = run_synthetic_experiment(**arguments)
    assert first == second
    assert first["schema_version"] == 1
    assert first["provenance"]["source"] == "synthetic_fixture"
    assert first["provenance"]["historical_coverage"] is False
    training = first["training"]
    assert training["initial_model"] != training["final_model"]
    assert len(training["episode_returns"]) == 12
    assert training["reward_mode"] == "on_time_arrival"
    assert training["final_model"] == first["frozen_model_after_evaluation"]
    assert first["evaluation_seed"] > first["training_seed"] + 12
    for result in first["results"].values():
        assert result["source_authenticated_episodes"] == 8
        assert result["score_profile"] == "deadline_first_v1"
        assert 0 <= result["on_time_arrival_rate"] <= result["arrival_rate"] <= 1
        assert all(trace["source_authentication"]["authenticated"] for trace in result["traces"])


@pytest.mark.parametrize(
    "arguments", [{"training_episodes": 0}, {"evaluation_episodes": 0}, {"seed": -1}]
)
def test_synthetic_example_rejects_invalid_counts_and_seeds(arguments):
    with pytest.raises(ValueError):
        run_synthetic_experiment(**arguments)


def test_public_lab_runtime_replays_without_a_server_or_historical_data():
    runtime = build_runtime(parse_args(["--fixture"]))
    assert runtime.catalog["title"] == "Gatebound Decision Lab"
    assert runtime.catalog["mode"] == "synthetic_fixture"
    assert b"<title>Gatebound Decision Lab</title>" in runtime.html
    payload = {
        "case_id": "synthetic_demo",
        "scenario_seed": 42,
        "policy_id": "nonstop_first",
        "choices": [],
    }
    first = replay(runtime, payload)
    assert first == replay(runtime, payload)
    assert first["summary"]["source_authenticated"]
    assert first["summary"]["validity"]
    with pytest.raises(RequestError, match="Unknown fields"):
        replay(runtime, {**payload, "unknown": True})


def test_public_request_catalog_is_invented_and_parses():
    root = Path(__file__).resolve().parents[1]
    cases = load_cases(root / "examples" / "decision_lab_cases.json")
    assert len(cases) == 2
    assert all(case.case_id.startswith("synthetic-request-") for case in cases)
    assert all("invented" in case.label.lower() for case in cases)
