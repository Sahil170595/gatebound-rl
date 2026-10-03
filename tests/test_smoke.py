"""Smoke tests: package imports, Verifier aggregation is correct, and
FlightRouteEnv is a proper gymnasium.Env subclass.
"""

from __future__ import annotations

import gymnasium as gym
import pytest

import flight_rl
from flight_rl.env import FlightRouteEnv
from flight_rl.fixtures import make_demo_scenario
from flight_rl.verifier import Criterion, Verifier


def test_package_imports():
    assert flight_rl.__version__


def test_verifier_weighted_aggregate_and_breakdown():
    # Two toy criteria with known, unequal weights and fixed raw scores, so
    # the expected weighted average can be computed by hand:
    #   criterion "a": raw 1.0, weight 3
    #   criterion "b": raw 0.0, weight 1
    #   aggregate = (1.0*3 + 0.0*1) / (3 + 1) = 0.75
    criterion_a = Criterion(
        name="a",
        weight=3.0,
        description="toy criterion that always scores 1.0",
        score_fn=lambda episode: 1.0,
    )
    criterion_b = Criterion(
        name="b",
        weight=1.0,
        description="toy criterion that always scores 0.0",
        score_fn=lambda episode: 0.0,
    )

    verifier = Verifier(criteria=[criterion_a, criterion_b])
    result = verifier.score(episode=object())

    assert result.aggregate_score == pytest.approx(0.75)
    assert len(result.breakdown) == 2

    by_name = {entry.name: entry for entry in result.breakdown}
    assert by_name["a"].raw_score == pytest.approx(1.0)
    assert by_name["a"].weight == pytest.approx(3.0)
    assert by_name["a"].weighted_contribution == pytest.approx(3.0)
    assert by_name["b"].raw_score == pytest.approx(0.0)
    assert by_name["b"].weight == pytest.approx(1.0)
    assert by_name["b"].weighted_contribution == pytest.approx(0.0)


def test_verifier_rejects_out_of_range_score():
    bad_criterion = Criterion(
        name="bad",
        weight=1.0,
        description="toy criterion that violates the [0, 1] contract",
        score_fn=lambda episode: 2.0,
    )
    verifier = Verifier(criteria=[bad_criterion])
    with pytest.raises(ValueError):
        verifier.score(episode=object())


def test_flight_route_env_is_gymnasium_env_subclass():
    assert issubclass(FlightRouteEnv, gym.Env)

    request, data = make_demo_scenario()
    env = FlightRouteEnv(request=request, data=data)
    assert isinstance(env, gym.Env)
