"""Gymnasium environment for sequential historical flight selection.

The observation contains schedules and fitted historical summaries only. A
flight's jointly sampled donor outcome remains private until its action is taken.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from flight_rl.candidates import canonical_candidates
from flight_rl.models import (
    CANDIDATE_FEATURES,
    EpisodeRecord,
    FlightCandidate,
    FlightDataSource,
    LegRecord,
    OutcomeSummary,
    SampledOutcome,
    TripRequest,
)

_REWARD_MODES = frozenset({"deadline_first", "on_time_arrival", "legacy_six_v1", "rubric"})
_REWARD_MODE_ALIASES = {"rubric": "legacy_six_v1"}
_IN_PROGRESS = "in_progress"
_FLOAT32_MAX = np.finfo(np.float32).max


def canonical_reward_mode(reward_mode: str) -> str:
    """Validate a reward mode and resolve compatibility aliases."""

    if not isinstance(reward_mode, str) or reward_mode not in _REWARD_MODES:
        raise ValueError(f"Unknown reward_mode {reward_mode!r}")
    return _REWARD_MODE_ALIASES.get(reward_mode, reward_mode)


def score_profile_for_reward_mode(reward_mode: str) -> str:
    """Return the stable report identifier for a selected reward mode."""

    mode = canonical_reward_mode(reward_mode)
    if mode == "deadline_first":
        return "deadline_first_v1"
    return mode


class FlightRouteEnv(gym.Env[dict[str, np.ndarray | int], int]):
    """Let one passenger select flights until arrival or terminal failure."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        request: TripRequest,
        data: FlightDataSource,
        *,
        max_candidates: int = 64,
        reward_mode: str = "deadline_first",
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(request, TripRequest):
            raise ValueError("request must be a TripRequest")  # noqa: TRY004 - contract
        self._validate_request(request)
        if isinstance(max_candidates, bool) or not isinstance(max_candidates, Integral):
            raise ValueError("max_candidates must be a positive integer")  # noqa: TRY004
        if max_candidates < 1:
            raise ValueError("max_candidates must be a positive integer")
        reward_mode = canonical_reward_mode(reward_mode)
        if render_mode not in {None, "human"}:
            raise ValueError(f"Unsupported render_mode {render_mode!r}")

        try:
            airports = tuple(data.airports)
            carriers = tuple(data.carriers)
        except (AttributeError, TypeError) as exc:
            raise ValueError("data must expose airport and carrier tuples") from exc
        if not airports or any(not isinstance(code, str) or not code for code in airports):
            raise ValueError("data.airports must contain nonempty airport codes")
        if len(set(airports)) != len(airports):
            raise ValueError("data.airports must not contain duplicates")
        if any(not isinstance(code, str) or not code for code in carriers):
            raise ValueError("data.carriers must contain nonempty carrier codes")
        if len(set(carriers)) != len(carriers):
            raise ValueError("data.carriers must not contain duplicates")
        if request.origin not in airports or request.destination not in airports:
            raise ValueError("request airports must be present in data.airports")

        self._request = request
        self._data = data
        self._max_candidates = int(max_candidates)
        self._reward_mode = reward_mode
        self.render_mode = render_mode
        self._airports = airports
        self._carriers = carriers
        self._airport_indexes = {code: index for index, code in enumerate(airports)}
        self._carrier_indexes = {code: index for index, code in enumerate(carriers)}

        episode_span = float(request.horizon_utc - request.ready_utc)
        deadline_span = float(request.deadline_utc - request.ready_utc)
        deadline_floor = float(request.deadline_utc - request.horizon_utc)
        self.observation_space = spaces.Dict(
            {
                "current_airport": spaces.Discrete(len(airports)),
                "destination": spaces.Discrete(len(airports)),
                "time": spaces.Box(
                    low=np.array([0.0, deadline_floor, 0.0, 0.0], dtype=np.float32),
                    high=np.array(
                        [episode_span, deadline_span, episode_span, request.max_attempts],
                        dtype=np.float32,
                    ),
                    dtype=np.float32,
                ),
                "disrupted": spaces.Discrete(2),
                "candidates": spaces.Box(
                    low=-_FLOAT32_MAX,
                    high=_FLOAT32_MAX,
                    shape=(self._max_candidates, len(CANDIDATE_FEATURES)),
                    dtype=np.float32,
                ),
                "action_mask": spaces.MultiBinary(self._max_candidates),
            }
        )
        self.action_space = spaces.Discrete(self._max_candidates)

        self._current_airport = request.origin
        self._clock_utc = request.ready_utc
        self._attempts = 0
        self._disrupted = False
        self._done = False
        self._legs: list[LegRecord] = []
        self._available_flights: tuple[FlightCandidate, ...] = ()
        self._candidate_rows = np.zeros(
            (self._max_candidates, len(CANDIDATE_FEATURES)), dtype=np.float32
        )
        self._record = self._make_record(_IN_PROGRESS)

    @property
    def request(self) -> TripRequest:
        return self._request

    @property
    def data(self) -> FlightDataSource:
        return self._data

    @property
    def reward_mode(self) -> str:
        return self._reward_mode

    @property
    def available_flights(self) -> tuple[FlightCandidate, ...]:
        return self._available_flights

    @property
    def record(self) -> EpisodeRecord:
        return self._record

    @property
    def clock_utc(self) -> int:
        return self._clock_utc

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray | int], dict[str, Any]]:
        """Start a fresh episode and enumerate flights after the boarding buffer."""
        super().reset(seed=seed)
        del options
        self._current_airport = self.request.origin
        self._clock_utc = self.request.ready_utc
        self._attempts = 0
        self._disrupted = False
        self._done = False
        self._legs = []
        self._refresh_candidates()
        self._record = self._make_record(_IN_PROGRESS)
        return self._observation(), {"episode": self.record}

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray | int], float, bool, bool, dict[str, Any]]:
        """Select one visible schedule and reveal one jointly sampled outcome."""
        if self._done:
            raise RuntimeError("step() called after episode termination")
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise ValueError("action must be an integer in the action space")  # noqa: TRY004
        action_index = int(action)
        if not self.action_space.contains(action_index):
            raise ValueError("action is outside the action space")

        if not self.available_flights:
            return self._finish("no_candidates")
        if action_index >= len(self.available_flights):
            return self._finish("invalid_action", forced_reward=0.0)

        flight = self.available_flights[action_index]
        prior_clock = self.clock_utc
        outcome = self.data.sample_outcome(flight, self.np_random)
        if not isinstance(outcome, SampledOutcome):
            raise ValueError("data.sample_outcome must return SampledOutcome")  # noqa: TRY004
        self._validate_outcome(outcome)
        self._attempts += 1
        self._disrupted = self._disrupted or outcome.cancelled or outcome.diverted

        if outcome.cancelled:
            self._legs.append(
                LegRecord(
                    flight=flight,
                    outcome=outcome,
                    departure_utc=None,
                    arrival_utc=None,
                    resolved_airport=self._current_airport,
                )
            )
            self._clock_utc = flight.scheduled_departure_utc
            return self._finish("cancelled")

        try:
            departure_utc = flight.scheduled_departure_utc + self._rounded(
                outcome.dep_delay_min, "dep_delay_min"
            )
            if outcome.diverted:
                arrival_utc = self._diversion_arrival(flight, outcome, departure_utc)
            else:
                arrival_utc = flight.scheduled_arrival_utc + self._rounded(
                    outcome.arr_delay_min, "arr_delay_min"
                )
        except ValueError:
            self._legs.append(
                LegRecord(
                    flight=flight,
                    outcome=outcome,
                    departure_utc=None,
                    arrival_utc=None,
                    resolved_airport=None,
                )
            )
            self._clock_utc = flight.scheduled_departure_utc
            return self._finish("invalid_outcome")

        resolved_airport = self._resolved_airport(flight, outcome)
        self._legs.append(
            LegRecord(
                flight=flight,
                outcome=outcome,
                departure_utc=departure_utc,
                arrival_utc=arrival_utc,
                resolved_airport=resolved_airport,
            )
        )

        if departure_utc < prior_clock:
            return self._finish("missed_departure")
        if arrival_utc is not None and arrival_utc <= departure_utc:
            self._clock_utc = min(departure_utc, self.request.horizon_utc)
            return self._finish("invalid_outcome")

        if outcome.diverted and not outcome.div_reached_dest:
            if arrival_utc is not None:
                self._clock_utc = min(arrival_utc, self.request.horizon_utc)
            else:
                self._clock_utc = min(departure_utc, self.request.horizon_utc)
            return self._finish("unresolved_diversion")

        if arrival_utc is None:
            self._clock_utc = min(departure_utc, self.request.horizon_utc)
            return self._finish("invalid_outcome")
        if arrival_utc > self.request.horizon_utc:
            self._clock_utc = self.request.horizon_utc
            return self._finish("horizon")

        self._clock_utc = arrival_utc
        self._current_airport = flight.dest
        if self._current_airport == self.request.destination:
            return self._finish("arrived")
        if self._clock_utc >= self.request.horizon_utc:
            return self._finish("horizon")
        if self._attempts >= self.request.max_attempts:
            return self._finish("max_attempts")

        self._refresh_candidates()
        if not self.available_flights:
            return self._finish("no_candidates")
        self._record = self._make_record(_IN_PROGRESS)
        return self._observation(), 0.0, False, False, {"episode": self.record}

    def render(self) -> str:
        """Return a compact human-readable state and print it in human mode."""
        text = (
            f"airport={self._current_airport} destination={self.request.destination} "
            f"clock_utc={self.clock_utc} attempts={self._attempts}/"
            f"{self.request.max_attempts} status={self.record.termination_reason}"
        )
        if self.render_mode == "human":
            print(text)
        return text

    @staticmethod
    def _rounded(value: float | None, field: str) -> int:
        if value is None or isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{field} must be a real number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{field} must be finite")
        return round(numeric)

    def _diversion_arrival(
        self, flight: FlightCandidate, outcome: SampledOutcome, departure_utc: int
    ) -> int | None:
        if outcome.div_arr_delay_min is not None:
            return flight.scheduled_arrival_utc + self._rounded(
                outcome.div_arr_delay_min, "div_arr_delay_min"
            )
        if outcome.div_actual_elapsed_min is not None:
            elapsed = self._rounded(outcome.div_actual_elapsed_min, "div_actual_elapsed_min")
            if elapsed <= 0:
                raise ValueError("div_actual_elapsed_min must be positive")
            return departure_utc + elapsed
        if outcome.div_reached_dest:
            raise ValueError("a destination-reaching diversion requires arrival timing")
        return None

    @staticmethod
    def _resolved_airport(flight: FlightCandidate, outcome: SampledOutcome) -> str | None:
        if not outcome.diverted:
            return flight.dest
        if outcome.div_reached_dest:
            return flight.dest
        return outcome.div_airport

    @staticmethod
    def _validate_request(request: TripRequest) -> None:
        if not isinstance(request.origin, str) or not isinstance(request.destination, str):
            raise ValueError("request airport codes must be strings")  # noqa: TRY004 - contract
        integer_fields = (
            request.ready_utc,
            request.deadline_utc,
            request.horizon_utc,
            request.max_attempts,
            request.min_connection_min,
            request.delay_budget_min,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) for value in integer_fields
        ):
            raise ValueError("request time and limit fields must be integers")

    @staticmethod
    def _validate_outcome(outcome: SampledOutcome) -> None:
        if not isinstance(outcome.donor_id, str) or not outcome.donor_id:
            raise ValueError("sampled outcome donor_id must be nonempty")
        if not all(
            isinstance(flag, bool)
            for flag in (outcome.cancelled, outcome.diverted, outcome.div_reached_dest)
        ):
            raise ValueError("sampled outcome status flags must be bool")
        if (
            isinstance(outcome.support, bool)
            or not isinstance(outcome.support, Integral)
            or outcome.support < 1
        ):
            raise ValueError("sampled outcome support must be a positive integer")
        if not isinstance(outcome.fallback_level, str) or not outcome.fallback_level:
            raise ValueError("sampled outcome fallback_level must be nonempty")
        if outcome.div_airport is not None and (
            not isinstance(outcome.div_airport, str) or not outcome.div_airport
        ):
            raise ValueError("sampled diversion airport must be a nonempty code or None")

    def _refresh_candidates(self) -> None:
        self._available_flights = canonical_candidates(
            self.data,
            self._current_airport,
            self.clock_utc + self.request.min_connection_min,
            self.request.horizon_utc,
            limit=self._max_candidates,
        )
        self._candidate_rows = np.zeros(
            (self._max_candidates, len(CANDIDATE_FEATURES)), dtype=np.float32
        )
        for index, flight in enumerate(self.available_flights):
            summary = self.data.outcome_summary(flight)
            self._candidate_rows[index] = self._candidate_features(flight, summary)

    def _candidate_features(self, flight: FlightCandidate, summary: OutcomeSummary) -> np.ndarray:
        if not isinstance(summary, OutcomeSummary):
            raise ValueError(  # noqa: TRY004 - invalid data is a contract ValueError
                "data.outcome_summary must return OutcomeSummary"
            )
        values = (summary.p_cancelled, summary.p_diverted, summary.mean_arrival_delay_min)
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ValueError("candidate summary values must be finite numbers")
        p_cancelled = float(summary.p_cancelled)
        p_diverted = float(summary.p_diverted)
        if not 0.0 <= p_cancelled <= 1.0 or not 0.0 <= p_diverted <= 1.0:
            raise ValueError("candidate probabilities must be in [0, 1]")
        if (
            isinstance(summary.support, bool)
            or not isinstance(summary.support, Integral)
            or summary.support < 1
        ):
            raise ValueError("candidate support must be a positive integer")
        row = np.asarray(
            [
                self._airport_indexes[flight.dest],
                self._carrier_indexes[flight.carrier],
                flight.scheduled_departure_utc - self.clock_utc,
                flight.scheduled_arrival_utc - self.clock_utc,
                flight.scheduled_elapsed_min,
                summary.p_cancelled,
                summary.p_diverted,
                summary.mean_arrival_delay_min,
                summary.support,
            ],
            dtype=np.float32,
        )
        if not np.isfinite(row).all():
            raise ValueError("candidate features must be finite float32 values")
        return row

    def _make_record(self, reason: str) -> EpisodeRecord:
        return EpisodeRecord(
            request=self.request,
            legs=tuple(self._legs),
            final_airport=self._current_airport,
            clock_utc=self.clock_utc,
            termination_reason=reason,
        )

    def _observation(self) -> dict[str, np.ndarray | int]:
        elapsed = self.clock_utc - self.request.ready_utc
        time = np.asarray(
            [
                elapsed,
                self.request.deadline_utc - self.clock_utc,
                self.request.horizon_utc - self.clock_utc,
                self.request.max_attempts - self._attempts,
            ],
            dtype=np.float32,
        )
        mask = np.zeros(self._max_candidates, dtype=np.int8)
        mask[: len(self.available_flights)] = 1
        return {
            "current_airport": self._airport_indexes[self._current_airport],
            "destination": self._airport_indexes[self.request.destination],
            "time": time,
            "disrupted": int(self._disrupted),
            "candidates": self._candidate_rows.copy(),
            "action_mask": mask,
        }

    def _finish(
        self, reason: str, *, forced_reward: float | None = None
    ) -> tuple[dict[str, np.ndarray | int], float, bool, bool, dict[str, Any]]:
        from flight_rl.verifier import episode_metrics, verify_episode, verify_legacy_six_v1

        self._done = True
        self._available_flights = ()
        self._candidate_rows.fill(0.0)
        self._record = self._make_record(reason)
        verification = verify_episode(self.record)
        metrics = episode_metrics(self.record)
        legacy_verification = (
            verify_legacy_six_v1(self.record) if self.reward_mode == "legacy_six_v1" else None
        )
        if forced_reward is not None:
            reward = forced_reward
        elif legacy_verification is not None:
            reward = float(legacy_verification.aggregate_score)
        elif self.reward_mode == "deadline_first":
            reward = float(verification.aggregate_score)
        else:
            reward = float(bool(metrics["on_time_arrival"]))
        if not math.isfinite(reward):
            raise ValueError("terminal reward must be finite")
        info = {
            "reward_mode": self.reward_mode,
            "score_profile": score_profile_for_reward_mode(self.reward_mode),
            "episode": self.record,
            "verification": verification,
            "metrics": metrics,
        }
        if legacy_verification is not None:
            info["legacy_six_v1_verification"] = legacy_verification
        return self._observation(), reward, True, False, info
