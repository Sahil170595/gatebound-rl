"""Transparent finite-horizon deadline planner, with explicit approximation controls."""

from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache
from numbers import Integral, Real

import numpy as np

from flight_rl.candidates import canonical_candidates
from flight_rl.models import FlightCandidate, FlightDataSource, SampledOutcome, TripRequest


def _finite_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _arrival(flight: FlightCandidate, outcome: SampledOutcome, clock: int) -> int | None:
    """Resolve a historical donor for planning without consuming the environment RNG."""
    if outcome.cancelled or (outcome.diverted and not outcome.div_reached_dest):
        return None
    departure_delay = outcome.dep_delay_min
    if not _finite_number(departure_delay):
        return None
    departure = flight.scheduled_departure_utc + round(departure_delay)
    if outcome.diverted:
        delay = outcome.div_arr_delay_min
        if delay is not None:
            if not _finite_number(delay):
                return None
            arrival = flight.scheduled_arrival_utc + round(delay)
        elif _finite_number(outcome.div_actual_elapsed_min):
            arrival = departure + round(outcome.div_actual_elapsed_min)
        else:
            return None
    elif _finite_number(outcome.arr_delay_min):
        arrival = flight.scheduled_arrival_utc + round(outcome.arr_delay_min)
    else:
        return None
    return arrival if departure >= clock and arrival > departure else None


class DeadlinePlannerPolicy:
    """Maximize modeled deadline-arrival probability over the allowed attempt horizon.

    Root actions match the environment's candidate cap. Future branching may be pruned to earliest
    scheduled arrivals. Connecting arrival times round upward and optional quantile bins use their
    upper endpoints. Combined with candidate caps these are approximations, not proven bounds.
    For a tiny exact oracle use max_branches=None, time_bin_min=1, outcome_bins=0.
    This policy uses fitted donor records, never sampled future outcomes.
    """

    def __init__(
        self,
        request: TripRequest,
        data: FlightDataSource,
        *,
        max_candidates: int = 64,
        max_branches: int | None = 12,
        time_bin_min: int = 15,
        outcome_bins: int = 8,
    ) -> None:
        for name, value in (
            ("max_candidates", max_candidates),
            ("time_bin_min", time_bin_min),
            ("outcome_bins", outcome_bins),
            ("max_branches", max_branches),
        ):
            if name == "max_branches" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")  # noqa: TRY004 - config contract
        if max_candidates < 1 or time_bin_min < 1 or outcome_bins < 0:
            raise ValueError("Invalid planner capacity or discretization")
        if max_branches is not None and max_branches < 1:
            raise ValueError("max_branches must be positive or None")
        self.request = request
        self.data = data
        self.max_candidates = max_candidates
        self.max_branches = max_branches
        self.time_bin_min = time_bin_min
        self.outcome_bins = outcome_bins
        # Instance-owned caches do not retain every policy ever created via a class-level cache.
        self._pool = lru_cache(maxsize=4096)(self._pool_uncached)
        self._value = lru_cache(maxsize=50000)(self._value_uncached)
        self._action_value = lru_cache(maxsize=50000)(self._action_value_uncached)

    def _pool_uncached(self, flight: FlightCandidate) -> tuple[SampledOutcome, ...]:
        return self.data.historical_outcomes(flight)

    def _value_uncached(self, airport: str, clock: int, attempts_left: int) -> float:
        if clock > self.request.deadline_utc:
            return 0.0
        if airport == self.request.destination:
            return 1.0
        if attempts_left < 1:
            return 0.0
        candidates = canonical_candidates(
            self.data,
            airport,
            clock + self.request.min_connection_min,
            self.request.horizon_utc,
            self.max_candidates,
        )
        if self.max_branches is not None:
            candidates = tuple(
                sorted(
                    candidates,
                    key=lambda f: (
                        f.scheduled_arrival_utc,
                        f.scheduled_departure_utc,
                        f.flight_id,
                    ),
                )[: self.max_branches]
            )
        return max((self._action_value(f, clock, attempts_left) for f in candidates), default=0.0)

    def _action_value_uncached(
        self, flight: FlightCandidate, clock: int, attempts_left: int
    ) -> float:
        pool = self._pool(flight)
        if not pool:
            return 0.0
        arrivals = [
            a
            for o in pool
            if (a := _arrival(flight, o, clock)) is not None and a <= self.request.deadline_utc
        ]
        if flight.dest == self.request.destination:
            return len(arrivals) / len(pool)
        if attempts_left <= 1 or not arrivals:
            return 0.0
        arrivals.sort()
        if self.outcome_bins and len(set(arrivals)) > self.outcome_bins:
            groups = np.array_split(np.asarray(arrivals, dtype=np.int64), self.outcome_bins)
            masses = [(int(group[-1]), len(group)) for group in groups]
        else:
            masses = list(Counter(arrivals).items())
        result = 0.0
        for arrival, count in masses:
            rounded = ((arrival + self.time_bin_min - 1) // self.time_bin_min) * self.time_bin_min
            result += count / len(pool) * self._value(flight.dest, rounded, attempts_left - 1)
        return result

    def action_values(self, observation: dict) -> tuple[float, ...]:
        airport = self.data.airports[int(observation["current_airport"])]
        clock = self.request.ready_utc + round(float(observation["time"][0]))
        attempts = round(float(observation["time"][3]))
        candidates = canonical_candidates(
            self.data,
            airport,
            clock + self.request.min_connection_min,
            self.request.horizon_utc,
            self.max_candidates,
        )
        legal = np.flatnonzero(observation["action_mask"])
        if not np.array_equal(legal, np.arange(len(candidates))):
            raise ValueError("Planner candidates do not match the observation mask")
        for index, candidate in enumerate(candidates):
            expected = [
                self.data.airports.index(candidate.dest),
                self.data.carriers.index(candidate.carrier),
                candidate.scheduled_departure_utc - clock,
                candidate.scheduled_arrival_utc - clock,
                candidate.scheduled_elapsed_min,
            ]
            actual = np.asarray(observation["candidates"][index, :5])
            if actual.shape != (5,) or not np.allclose(actual, expected, rtol=0, atol=0.01):
                raise ValueError("Planner schedule order differs from observation candidates")
        return tuple(self._action_value(f, clock, attempts) for f in candidates)

    def act(self, observation: dict) -> int:
        legal = np.flatnonzero(observation["action_mask"])
        if not len(legal):
            return 0
        values = self.action_values(observation)
        if int(legal[-1]) >= len(values):
            raise ValueError("Planner candidates do not match the observation mask")
        return int(
            max(
                legal,
                key=lambda i: (
                    values[int(i)],
                    -float(observation["candidates"][int(i), 3]),
                    -int(i),
                ),
            )
        )

    def diagnostics(self) -> dict:
        return {
            "objective": "on_time_arrival",
            "max_candidates": self.max_candidates,
            "max_branches": self.max_branches,
            "time_bin_min": self.time_bin_min,
            "outcome_bins": self.outcome_bins,
            "assumption": "independent historical donors; future candidates capped",
            "value_cache": self._value.cache_info()._asdict(),
        }
