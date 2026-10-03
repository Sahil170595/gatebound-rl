"""Mask-aware, deterministic baseline policies for :mod:`flight_rl.env`."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

import numpy as np

from flight_rl.models import CANDIDATE_FEATURES

_DESTINATION = CANDIDATE_FEATURES.index("destination_index")
_DEPARTURE = CANDIDATE_FEATURES.index("departure_in_min")
_ARRIVAL = CANDIDATE_FEATURES.index("arrival_in_min")


def _valid_action_indices(observation: Any) -> np.ndarray:
    if not isinstance(observation, Mapping) or "action_mask" not in observation:
        raise ValueError("observation must contain a one-dimensional action_mask")
    mask = np.asarray(observation["action_mask"])
    if mask.ndim != 1:
        raise ValueError("action_mask must be one-dimensional")
    try:
        finite = np.isfinite(mask)
    except TypeError as exc:
        raise ValueError("action_mask must contain only binary numeric values") from exc
    if not bool(np.all(finite)) or not bool(np.all((mask == 0) | (mask == 1))):
        raise ValueError("action_mask must contain only 0 and 1")
    return np.flatnonzero(mask.astype(bool))


def _candidate_view(observation: Any) -> tuple[np.ndarray, np.ndarray, int]:
    valid = _valid_action_indices(observation)
    if "candidates" not in observation or "destination" not in observation:
        raise ValueError("observation must contain candidates and destination")
    try:
        candidates = np.asarray(observation["candidates"], dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("candidates must be a numeric matrix") from exc
    if candidates.ndim != 2 or candidates.shape[1] != len(CANDIDATE_FEATURES):
        raise ValueError(f"candidates must have shape (N, {len(CANDIDATE_FEATURES)})")
    if candidates.shape[0] != np.asarray(observation["action_mask"]).shape[0]:
        raise ValueError("candidates and action_mask must have the same row count")

    destination = observation["destination"]
    if isinstance(destination, bool) or not isinstance(destination, Integral):
        raise ValueError("destination must be an integer airport index")  # noqa: TRY004
    destination_index = int(destination)
    if valid.size and not bool(
        np.all(np.isfinite(candidates[valid][:, [_DESTINATION, _DEPARTURE, _ARRIVAL]]))
    ):
        raise ValueError("valid candidate destination and timing features must be finite")
    return candidates, valid, destination_index


def _best_index(candidates: np.ndarray, indices: np.ndarray, primary: int) -> int:
    secondary = _DEPARTURE if primary == _ARRIVAL else _ARRIVAL
    return min(
        (int(index) for index in indices),
        key=lambda index: (
            float(candidates[index, primary]),
            float(candidates[index, secondary]),
            index,
        ),
    )


class RandomPolicy:
    """Sample uniformly from the currently valid action indices."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, observation: Any) -> int:
        valid = _valid_action_indices(observation)
        if valid.size == 0:
            return 0
        return int(self._rng.choice(valid))


class NonstopFirstPolicy:
    """Prefer earliest-arriving nonstop service, then earliest departure."""

    def act(self, observation: Any) -> int:
        candidates, valid, destination = _candidate_view(observation)
        if valid.size == 0:
            return 0
        nonstop = valid[candidates[valid, _DESTINATION] == float(destination)]
        if nonstop.size:
            return _best_index(candidates, nonstop, _ARRIVAL)
        return _best_index(candidates, valid, _DEPARTURE)


class ShortestScheduledArrivalPolicy:
    """Choose the valid candidate with the earliest scheduled arrival.

    This is a schedule-only comparison baseline. It deliberately uses no
    sampled outcome, hidden environment RNG state, or fitted donor statistic.
    """

    def act(self, observation: Any) -> int:
        candidates, valid, _destination = _candidate_view(observation)
        if valid.size == 0:
            return 0
        return _best_index(candidates, valid, _ARRIVAL)
