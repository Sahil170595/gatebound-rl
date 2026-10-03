"""Focused contract proofs for the interactive decision lab."""

import pytest

from flight_rl.decision_lab import LabEngine
from flight_rl.fixtures import make_demo_scenario


def test_replay_reconstructs_legal_prefix_and_preserves_scenario_outcomes() -> None:
    request, data = make_demo_scenario()
    engine = LabEngine(data)

    first = engine.replay(
        request,
        scenario_seed=0,
        policy_id="deadline_planner",
        choices=("demo-SFO-DEN",),
    )
    reconstructed = engine.replay(
        request,
        scenario_seed=0,
        policy_id="deadline_planner",
        choices=first["choices"],
    )

    assert first["choices"] == ["demo-SFO-DEN", "demo-DEN-JFK"]
    assert reconstructed["steps"] == first["steps"]
    assert first["summary"]["validity"] is True
    assert first["summary"]["source_authenticated"] is True
    assert first["summary"]["score_profile"] == "deadline_first_v1"
    assert [row["name"] for row in first["rubric"]] == [
        "on_time_arrival",
        "arrived",
        "earliness",
    ]
    assert all(step["recommended_flight_id"] for step in first["steps"])
    assert all(candidate["support"] > 0 for candidate in first["steps"][0]["candidates"])
    assert all(
        0 <= candidate["deadline_probability"] <= 1 for candidate in first["steps"][0]["candidates"]
    )
    assert all(step["outcome"]["status"] == "completed" for step in first["steps"])

    signed_seed = engine.replay(request, -1, "nonstop_first")
    assert signed_seed["scenario_seed"] == -1
    assert signed_seed == engine.replay(request, -1, "nonstop_first")
    assert all(
        candidate["deadline_probability"] is None
        for candidate in signed_seed["steps"][0]["candidates"]
    )


def test_replay_rejects_impossible_and_extra_prefix_choices() -> None:
    request, data = make_demo_scenario()
    engine = LabEngine(data)

    with pytest.raises(ValueError, match="not legal at step 1"):
        engine.replay(
            request,
            0,
            choices=("demo-SFO-DEN", "demo-SFO-JFK"),
        )
    with pytest.raises(ValueError, match="continues after.*terminated"):
        engine.replay(
            request,
            0,
            choices=("demo-SFO-JFK", "demo-SFO-DEN"),
        )


def test_comparison_is_bounded_to_legal_initial_flights() -> None:
    request, data = make_demo_scenario()
    engine = LabEngine(data)

    result = engine.compare_first_actions(
        request,
        10,
        "nonstop_first",
        "demo-SFO-DEN",
        "demo-SFO-JFK",
        trials=4,
    )
    assert result["trials"] == 4
    assert result["source_authenticated_episodes"] == 8
    assert result["score_profile"] == "deadline_first_v1"
    assert result["reference"]["flight_id"] == "demo-SFO-DEN"
    assert result["alternative"]["flight_id"] == "demo-SFO-JFK"
    assert 0 <= result["reference"]["mean_score"] <= 1
    assert (
        result["deadline_delta"]
        == (result["alternative"]["on_time_arrivals"] - result["reference"]["on_time_arrivals"]) / 4
    )

    with pytest.raises(ValueError, match="not legal at the initial state"):
        engine.compare_first_actions(
            request,
            10,
            "nonstop_first",
            "demo-DEN-JFK",
            "demo-SFO-JFK",
        )
    with pytest.raises(ValueError, match="1 to 64"):
        engine.compare_first_actions(
            request,
            10,
            "nonstop_first",
            "demo-SFO-DEN",
            "demo-SFO-JFK",
            trials=65,
        )


def test_lab_rejects_outcome_absent_from_trusted_source(monkeypatch) -> None:
    request, data = make_demo_scenario()
    monkeypatch.setattr(type(data), "source_transition_outcome", lambda *args: None)
    with pytest.raises(RuntimeError, match="failed source authentication"):
        LabEngine(data).replay(request, 0, "nonstop_first")
