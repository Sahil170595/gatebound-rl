from dataclasses import replace

import numpy as np
import pytest

from flight_rl.env import FlightRouteEnv
from flight_rl.evaluation import evaluate_policy
from flight_rl.fixtures import DemoFlightData, make_demo_scenario
from flight_rl.learning import (
    FEATURE_NAMES,
    MaskedLinearPolicy,
    TrainingConfig,
    features,
    train_reinforce,
)
from flight_rl.models import SampledOutcome


def observation():
    request, data = make_demo_scenario()
    return FlightRouteEnv(request, data).reset(seed=42)[0]


def binary_choice_problem():
    """A direct flight succeeds and an otherwise identical flight is always cancelled."""
    request, source = make_demo_scenario()
    reliable = next(
        flight
        for flight in source.flights
        if flight.origin == request.origin and flight.dest == request.destination
    )
    cancelled = replace(reliable, flight_id="z-cancelled")
    data = DemoFlightData(
        (reliable, cancelled),
        {
            reliable.flight_id: (source.historical_outcomes(reliable)[0],),
            cancelled.flight_id: (
                SampledOutcome(
                    "cancelled-outcome",
                    cancelled=True,
                    dep_delay_min=None,
                    arr_delay_min=None,
                ),
            ),
        },
    )
    return request, data


def reliable_action_probability(policy, request, data):
    env = FlightRouteEnv(request, data, reward_mode="on_time_arrival")
    try:
        obs, _ = env.reset(seed=0)
        reliable_action = next(
            index
            for index, flight in enumerate(env.available_flights)
            if flight.flight_id == "demo-SFO-JFK"
        )
        legal, _, probabilities = policy.distribution(obs)
        return float(probabilities[np.flatnonzero(legal == reliable_action)[0]])
    finally:
        env.close()


def test_softmax_gradient_matches_finite_difference_for_every_feature():
    obs = observation()
    weights = np.linspace(-0.4, 0.4, len(FEATURE_NAMES))
    policy = MaskedLinearPolicy(weights)
    legal, _, _ = policy.distribution(obs)
    for position, action in enumerate(legal):
        expected = policy.log_gradient(obs, int(action))
        numeric = []
        for j in range(len(weights)):
            shift = np.eye(len(weights))[j] * 1e-6
            high = MaskedLinearPolicy(weights + shift).distribution(obs)[2][position]
            low = MaskedLinearPolicy(weights - shift).distribution(obs)[2][position]
            numeric.append((np.log(high) - np.log(low)) / 2e-6)
        np.testing.assert_allclose(expected, numeric, atol=1e-8, rtol=1e-6)


def test_mask_holes_padded_nan_and_empty_state_are_safe():
    obs = observation()
    obs["action_mask"][1] = 0
    obs["candidates"][obs["action_mask"] == 0] = np.nan
    policy = MaskedLinearPolicy(np.full(len(FEATURE_NAMES), 1000), seed=9)
    assert {policy.act(obs) for _ in range(100)} <= {0, 2}
    obs["action_mask"][:] = 0
    action, gradient = policy.act_and_gradient(obs)
    assert action == 0
    np.testing.assert_array_equal(gradient, np.zeros(len(FEATURE_NAMES)))


def test_model_roundtrip_preserves_seeded_actions_and_cannot_update_on_eval():
    obs = observation()
    policy = MaskedLinearPolicy(np.arange(len(FEATURE_NAMES)) / 10, seed=17)
    model = policy.to_dict()
    restored = MaskedLinearPolicy.from_dict(model, seed=17)
    assert [policy.act(obs) for _ in range(100)] == [restored.act(obs) for _ in range(100)]
    assert policy.to_dict() == model == restored.to_dict()
    exposed = policy.weights
    exposed[:] = 400
    assert policy.to_dict() == model


@pytest.mark.parametrize(
    "weights", [[1], [float("nan")] * 9, [float("inf")] * 9, [True] * 9, [1e8] * 9]
)
def test_malformed_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        MaskedLinearPolicy(weights)


def test_version_mismatch_and_nonfinite_legal_features_are_rejected():
    model = MaskedLinearPolicy().to_dict()
    model["feature_version"] = True
    with pytest.raises(ValueError):
        MaskedLinearPolicy.from_dict(model)
    obs = observation()
    obs["candidates"][0, 7] = np.nan
    with pytest.raises(ValueError):
        features(obs)


def test_baseline_uses_preceding_episodes_and_does_not_create_false_gradient():
    request, data = make_demo_scenario()
    flight = next(f for f in data.flights if f.origin == "SFO" and f.dest == "JFK")
    deterministic = DemoFlightData(
        (flight,), {flight.flight_id: (replace(data.historical_outcomes(flight)[0], support=1),)}
    )

    def factory(_index, _seed):
        return FlightRouteEnv(request, deterministic, reward_mode="on_time_arrival")

    model, result = train_reinforce(factory, seed=10, config=TrainingConfig(episodes=3))
    assert result["episode_returns"] == [1, 1, 1]
    np.testing.assert_allclose(result["baseline_before_episode"], [0, 0.05, 0.0975])
    np.testing.assert_array_equal(model.weights, np.zeros(len(FEATURE_NAMES)))


def test_rewarded_update_increases_the_action_probability():
    request, data = binary_choice_problem()

    def factory(_index, _seed):
        return FlightRouteEnv(request, data, reward_mode="on_time_arrival")

    initial_probability = reliable_action_probability(MaskedLinearPolicy(), request, data)
    model, result = train_reinforce(
        factory,
        seed=1,
        config=TrainingConfig(episodes=1, learning_rate=0.4),
    )

    assert result["episode_returns"] == [1.0]
    assert reliable_action_probability(model, request, data) > initial_probability


def test_training_improves_exact_expected_reward_on_a_binary_choice():
    request, data = binary_choice_problem()

    def factory(_index, _seed):
        return FlightRouteEnv(request, data, reward_mode="on_time_arrival")

    initial_expected_reward = reliable_action_probability(MaskedLinearPolicy(), request, data)
    model, _ = train_reinforce(
        factory,
        seed=123,
        config=TrainingConfig(episodes=200, learning_rate=0.1),
    )
    trained_expected_reward = reliable_action_probability(model, request, data)

    # Outcomes are deterministic: E[reward] is exactly P(selecting the reliable flight).
    assert initial_expected_reward == pytest.approx(0.5)
    assert trained_expected_reward > 0.8


def test_training_is_reproducible_and_records_every_episode():
    request, data = make_demo_scenario()

    def factory(_index, _seed):
        return FlightRouteEnv(request, data, reward_mode="on_time_arrival")

    _, a = train_reinforce(factory, seed=123, config=TrainingConfig(episodes=30))
    _, b = train_reinforce(factory, seed=123, config=TrainingConfig(episodes=30))
    assert a == b
    assert len(a["episode_returns"]) == len(a["baseline_before_episode"]) == 30
    assert a["invalid_records"] == 0
    assert a["final_model"]["weights"] != a["initial_model"]["weights"]


def test_real_evaluation_does_not_update_frozen_weights():
    request, data = make_demo_scenario()
    policy = MaskedLinearPolicy(np.arange(len(FEATURE_NAMES)) / 10, seed=7)
    before = policy.to_dict()
    result = evaluate_policy(
        lambda: FlightRouteEnv(request, data), lambda _env, _seed: policy, episodes=20, seed=100
    )
    assert result["invalid_records"] == 0
    assert policy.to_dict() == before


def test_primary_reward_mode_is_rejected_before_a_binary_training_episode(monkeypatch):
    request, data = make_demo_scenario()
    flight = next(f for f in data.flights if f.origin == "SFO" and f.dest == "JFK")
    deterministic = DemoFlightData(
        (flight,), {flight.flight_id: (data.historical_outcomes(flight)[0],)}
    )
    primary_env = FlightRouteEnv(request, deterministic)
    monkeypatch.setattr(
        primary_env,
        "reset",
        lambda **_kwargs: pytest.fail("primary environment should be rejected before reset"),
    )
    with pytest.raises(ValueError, match="binary on-time"):
        train_reinforce(
            lambda _index, _seed: primary_env,
            seed=10,
            config=TrainingConfig(episodes=1),
        )


@pytest.mark.parametrize("seed", [True, -1, 1.5])
def test_invalid_policy_seed_is_rejected(seed):
    with pytest.raises(ValueError):
        MaskedLinearPolicy(seed=seed)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"episodes": 0},
        {"episodes": True},
        {"learning_rate": float("nan")},
        {"baseline_rate": 1.1},
        {"gradient_norm_cap": -1},
    ],
)
def test_invalid_training_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)
