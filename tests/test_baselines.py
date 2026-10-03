from __future__ import annotations

import numpy as np
import pytest

from flight_rl.baselines import NonstopFirstPolicy, RandomPolicy, ShortestScheduledArrivalPolicy
from flight_rl.models import CANDIDATE_FEATURES


def observation(
    *,
    mask: tuple[int, ...] = (1, 1, 1, 0),
    destination: int = 2,
) -> dict[str, np.ndarray | np.integer]:
    candidates = np.zeros((len(mask), len(CANDIDATE_FEATURES)), dtype=np.float32)
    # dest index, carrier index, departure offset, arrival offset, then priors.
    candidates[0, :5] = (1, 0, 20, 100, 80)
    candidates[1, :5] = (2, 0, 40, 90, 50)
    candidates[2, :5] = (2, 1, 30, 80, 50)
    if len(mask) > 3:
        candidates[3, :5] = (2, 1, 1, 1, 1)
    return {
        "candidates": candidates,
        "action_mask": np.asarray(mask, dtype=np.int8),
        "destination": np.int64(destination),
    }


def test_random_policy_is_repeatable_and_never_uses_masked_rows() -> None:
    obs = observation(mask=(0, 1, 0, 1))
    first = RandomPolicy(seed=418)
    second = RandomPolicy(seed=418)
    first_actions = [first.act(obs) for _ in range(50)]
    second_actions = [second.act(obs) for _ in range(50)]

    assert first_actions == second_actions
    assert set(first_actions) == {1, 3}


def test_policies_return_zero_for_empty_mask() -> None:
    obs = observation(mask=(0, 0, 0, 0))
    assert RandomPolicy(seed=1).act(obs) == 0
    assert NonstopFirstPolicy().act(obs) == 0
    assert ShortestScheduledArrivalPolicy().act(obs) == 0


def test_nonstop_first_uses_earliest_arrival_among_destination_rows() -> None:
    assert NonstopFirstPolicy().act(observation()) == 2


def test_nonstop_first_ignores_masked_early_nonstop() -> None:
    obs = observation(mask=(1, 1, 0, 0))
    assert NonstopFirstPolicy().act(obs) == 1


def test_nonstop_first_falls_back_to_earliest_departure() -> None:
    obs = observation(destination=3)
    assert NonstopFirstPolicy().act(obs) == 0


def test_shortest_scheduled_arrival_is_destination_agnostic_and_mask_aware() -> None:
    obs = observation(mask=(1, 1, 1, 0))
    candidates = obs["candidates"]
    assert isinstance(candidates, np.ndarray)
    candidates[0, 3] = 60

    # The schedule baseline takes the earlier connecting arrival, while the
    # nonstop heuristic retains its destination preference.
    assert ShortestScheduledArrivalPolicy().act(obs) == 0
    assert NonstopFirstPolicy().act(obs) == 2

    obs["action_mask"] = np.array([0, 1, 1, 0], dtype=np.int8)
    assert ShortestScheduledArrivalPolicy().act(obs) == 2


def test_nonstop_first_ties_are_stable_by_other_time_then_index() -> None:
    obs = observation()
    candidates = obs["candidates"]
    assert isinstance(candidates, np.ndarray)
    candidates[1, 3] = candidates[2, 3]
    candidates[1, 2] = candidates[2, 2]
    assert NonstopFirstPolicy().act(obs) == 1


@pytest.mark.parametrize(
    "bad_mask",
    [
        np.array([[1, 0]], dtype=np.int8),
        np.array([1, 2, 0, 0], dtype=np.int8),
        np.array([1.0, np.nan, 0.0, 0.0]),
    ],
)
def test_policies_reject_malformed_masks(bad_mask: np.ndarray) -> None:
    obs = observation()
    obs["action_mask"] = bad_mask
    with pytest.raises(ValueError):
        RandomPolicy().act(obs)


def test_nonstop_first_rejects_nonfinite_valid_candidate_time() -> None:
    obs = observation()
    candidates = obs["candidates"]
    assert isinstance(candidates, np.ndarray)
    candidates[1, 3] = np.nan
    with pytest.raises(ValueError):
        NonstopFirstPolicy().act(obs)
