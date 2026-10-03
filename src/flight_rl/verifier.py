"""Deterministic verification and rubric scoring for flight episodes.

``Verifier`` remains generic: callers may supply arbitrary criteria over any
episode representation. The module-level helpers apply the project's frozen
rubric to :class:`~flight_rl.models.EpisodeRecord` values after independently
checking the claimed itinerary against its schedules and sampled donor rows.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from itertools import pairwise
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any

from flight_rl.models import EpisodeRecord, FlightCandidate, LegRecord, SampledOutcome, TripRequest

ScoreFn = Callable[[Any], float]

_TERMINATION_REASONS = frozenset(
    {
        "arrived",
        "cancelled",
        "unresolved_diversion",
        "horizon",
        "max_attempts",
        "no_candidates",
        "invalid_action",
        "invalid_outcome",
        "missed_departure",
    }
)


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{label} must be a finite number")  # noqa: TRY004
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be a finite number")
    return converted


def _utc_int(value: object) -> int | None:
    """Normalize UTC epoch minutes without silently accepting bool/float."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    return int(value)


def _rounded_minutes(value: object) -> int | None:
    if isinstance(value, (bool, str, bytes)) or not isinstance(value, Real):
        return None
    converted = float(value)
    if not math.isfinite(converted):
        return None
    return round(converted)


@dataclass(frozen=True)
class Criterion:
    """One immutable named, weighted scoring rule."""

    name: str
    weight: float
    description: str
    score_fn: ScoreFn

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Criterion name must be a nonempty string")
        weight = _finite_float(self.weight, label=f"Criterion '{self.name}' weight")
        if weight < 0.0:
            raise ValueError(f"Criterion '{self.name}' weight must be nonnegative")
        if not callable(self.score_fn):
            raise ValueError(  # noqa: TRY004
                f"Criterion '{self.name}' score_fn must be callable"
            )
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True)
class CriterionResult:
    """Immutable per-criterion detail for one verifier call."""

    name: str
    raw_score: float
    weight: float
    weighted_contribution: float


@dataclass(frozen=True)
class VerificationResult:
    """Aggregate verifier score and the complete immutable breakdown."""

    aggregate_score: float
    breakdown: tuple[CriterionResult, ...] = field(default_factory=tuple)


class Verifier:
    """Aggregate an immutable sequence of named criteria by relative weight."""

    def __init__(self, criteria: Iterable[Criterion]) -> None:
        try:
            criterion_tuple = tuple(criteria)
        except TypeError as exc:
            raise ValueError("criteria must be an iterable of Criterion values") from exc
        if not criterion_tuple:
            raise ValueError("Verifier requires at least one Criterion")
        if any(not isinstance(criterion, Criterion) for criterion in criterion_tuple):
            raise ValueError("Verifier criteria must all be Criterion values")

        names = tuple(criterion.name for criterion in criterion_tuple)
        if len(set(names)) != len(names):
            raise ValueError(f"Criterion names must be unique, got: {list(names)}")

        weights = tuple(
            _finite_float(criterion.weight, label=f"Criterion '{criterion.name}' weight")
            for criterion in criterion_tuple
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("Criterion weights must be nonnegative")
        try:
            total_weight = math.fsum(weights)
        except OverflowError as exc:
            raise ValueError("Sum of criterion weights must be finite and positive") from exc
        if not math.isfinite(total_weight) or total_weight <= 0.0:
            raise ValueError("Sum of criterion weights must be finite and positive")

        self._criteria = criterion_tuple
        self._total_weight = total_weight

    @property
    def criteria(self) -> tuple[Criterion, ...]:
        """The fixed criterion sequence used for every call to :meth:`score`."""

        return self._criteria

    def score(self, episode: Any) -> VerificationResult:
        """Score ``episode`` and return a finite breakdown in ``[0, 1]``."""

        return self._result_from_scores(criterion.score_fn(episode) for criterion in self.criteria)

    def _result_from_scores(self, raw_scores: Iterable[object]) -> VerificationResult:
        """Build a checked result from scores ordered like :attr:`criteria`."""

        breakdown: list[CriterionResult] = []
        contributions: list[float] = []
        for criterion, score in zip(self.criteria, raw_scores, strict=True):
            raw_score = _finite_float(
                score,
                label=f"Criterion '{criterion.name}' score",
            )
            if not 0.0 <= raw_score <= 1.0:
                raise ValueError(
                    f"Criterion '{criterion.name}' returned {raw_score!r}, "
                    "expected a value in [0, 1]"
                )
            contribution = raw_score * criterion.weight
            if not math.isfinite(contribution):
                raise ValueError(f"Criterion '{criterion.name}' contribution must be finite")
            contributions.append(contribution)
            breakdown.append(
                CriterionResult(
                    name=criterion.name,
                    raw_score=raw_score,
                    weight=criterion.weight,
                    weighted_contribution=contribution,
                )
            )

        aggregate = math.fsum(contributions) / self._total_weight
        if not math.isfinite(aggregate):
            raise ValueError("Aggregate verifier score must be finite")
        return VerificationResult(aggregate_score=aggregate, breakdown=tuple(breakdown))


@dataclass(frozen=True)
class _EpisodeAnalysis:
    validity: bool
    arrived: bool
    on_time_arrival: bool
    elapsed_min: int | None
    disrupted: bool
    attempts: int
    termination_reason: str


@dataclass(frozen=True)
class _ResolvedLeg:
    kind: str
    departure_utc: int | None
    arrival_utc: int | None
    detail_airport: str | None


def _valid_request(request: object) -> bool:
    if not isinstance(request, TripRequest):
        return False
    ready = _utc_int(request.ready_utc)
    deadline = _utc_int(request.deadline_utc)
    horizon = _utc_int(request.horizon_utc)
    max_attempts = _utc_int(request.max_attempts)
    connection = _utc_int(request.min_connection_min)
    budget = _utc_int(request.delay_budget_min)
    return bool(
        isinstance(request.origin, str)
        and request.origin
        and isinstance(request.destination, str)
        and request.destination
        and request.origin != request.destination
        and ready is not None
        and deadline is not None
        and horizon is not None
        and ready < horizon
        and deadline <= horizon
        and max_attempts is not None
        and max_attempts >= 1
        and connection is not None
        and connection >= 0
        and budget is not None
        and budget > 0
    )


def _valid_flight(flight: object) -> bool:
    if not isinstance(flight, FlightCandidate):
        return False
    departure = _utc_int(flight.scheduled_departure_utc)
    arrival = _utc_int(flight.scheduled_arrival_utc)
    return bool(
        isinstance(flight.flight_id, str)
        and flight.flight_id
        and isinstance(flight.origin, str)
        and flight.origin
        and isinstance(flight.dest, str)
        and flight.dest
        and flight.origin != flight.dest
        and departure is not None
        and arrival is not None
        and departure < arrival
    )


def _valid_outcome_identity(outcome: object) -> bool:
    if not isinstance(outcome, SampledOutcome):
        return False
    support = _utc_int(outcome.support)
    return bool(
        isinstance(outcome.donor_id, str)
        and outcome.donor_id
        and isinstance(outcome.cancelled, bool)
        and isinstance(outcome.diverted, bool)
        and isinstance(outcome.div_reached_dest, bool)
        and support is not None
        and support >= 1
    )


def _derived_arrival(
    flight: FlightCandidate, outcome: SampledOutcome, departure: int
) -> int | None:
    if outcome.diverted:
        div_arrival_delay = _rounded_minutes(outcome.div_arr_delay_min)
        if div_arrival_delay is not None:
            return flight.scheduled_arrival_utc + div_arrival_delay
        div_elapsed = _rounded_minutes(outcome.div_actual_elapsed_min)
        if div_elapsed is not None and div_elapsed > 0:
            return departure + div_elapsed
        return None
    arrival_delay = _rounded_minutes(outcome.arr_delay_min)
    if arrival_delay is None:
        return None
    return flight.scheduled_arrival_utc + arrival_delay


def _resolve_leg(leg: LegRecord, prior_airport: str, prior_clock: int) -> _ResolvedLeg | None:
    """Derive one selected leg solely from schedule and donor fields."""

    flight = leg.flight
    outcome = leg.outcome
    if not _valid_flight(flight) or not _valid_outcome_identity(outcome):
        return None
    if flight.origin != prior_airport:
        return None

    recorded_departure = None if leg.departure_utc is None else _utc_int(leg.departure_utc)
    recorded_arrival = None if leg.arrival_utc is None else _utc_int(leg.arrival_utc)
    if leg.departure_utc is not None and recorded_departure is None:
        return None
    if leg.arrival_utc is not None and recorded_arrival is None:
        return None
    if leg.resolved_airport is not None and not isinstance(leg.resolved_airport, str):
        return None

    # Cancellation is authoritative, including source rows with both flags set.
    if outcome.cancelled:
        if recorded_departure is not None or recorded_arrival is not None:
            return None
        if leg.resolved_airport != prior_airport:
            return None
        return _ResolvedLeg("cancelled", None, None, prior_airport)

    dep_delay = _rounded_minutes(outcome.dep_delay_min)
    if dep_delay is None:
        if recorded_departure is None and recorded_arrival is None and leg.resolved_airport is None:
            return _ResolvedLeg("invalid_outcome", None, None, None)
        return None
    expected_departure = flight.scheduled_departure_utc + dep_delay
    expected_arrival = _derived_arrival(flight, outcome, expected_departure)
    if expected_arrival is None and outcome.diverted and not outcome.div_reached_dest:
        if (
            recorded_departure == expected_departure
            and recorded_arrival is None
            and leg.resolved_airport == outcome.div_airport
        ):
            if expected_departure < prior_clock:
                return _ResolvedLeg(
                    "missed_departure", expected_departure, None, outcome.div_airport
                )
            return _ResolvedLeg(
                "unresolved_diversion", expected_departure, None, outcome.div_airport
            )
        return None
    if expected_arrival is None or expected_arrival <= expected_departure:
        if recorded_departure is None and recorded_arrival is None and leg.resolved_airport is None:
            return _ResolvedLeg("invalid_outcome", None, None, None)
        return None

    detail_airport = (
        outcome.div_airport if outcome.diverted and not outcome.div_reached_dest else flight.dest
    )
    if recorded_departure != expected_departure:
        return None
    if recorded_arrival != expected_arrival or leg.resolved_airport != detail_airport:
        return None
    if expected_departure < prior_clock:
        return _ResolvedLeg(
            "missed_departure", expected_departure, expected_arrival, detail_airport
        )
    if outcome.diverted and not outcome.div_reached_dest:
        return _ResolvedLeg(
            "unresolved_diversion", expected_departure, expected_arrival, detail_airport
        )
    return _ResolvedLeg("completed", expected_departure, expected_arrival, flight.dest)


def _record_is_valid(record: object) -> bool:
    if not isinstance(record, EpisodeRecord) or not _valid_request(record.request):
        return False
    if not isinstance(record.legs, tuple) or any(
        not isinstance(leg, LegRecord) for leg in record.legs
    ):
        return False
    request = record.request
    if len(record.legs) > request.max_attempts:
        return False
    clock = _utc_int(record.clock_utc)
    if clock is None or not request.ready_utc <= clock <= request.horizon_utc:
        return False
    if not isinstance(record.final_airport, str) or not record.final_airport:
        return False
    if record.termination_reason not in _TERMINATION_REASONS:
        return False

    prior_airport = request.origin
    prior_clock = request.ready_utc
    terminal_kind: str | None = None
    terminal_clock: int | None = None
    terminal_final_airport: str | None = None

    for index, leg in enumerate(record.legs):
        flight = leg.flight
        scheduled_departure = (
            _utc_int(flight.scheduled_departure_utc)
            if isinstance(flight, FlightCandidate)
            else None
        )
        if scheduled_departure is None:
            return False
        if scheduled_departure < prior_clock + request.min_connection_min:
            return False
        if scheduled_departure > request.horizon_utc:
            return False

        resolved = _resolve_leg(leg, prior_airport, prior_clock)
        if resolved is None or terminal_kind is not None:
            return False
        is_last = index == len(record.legs) - 1

        if resolved.kind == "cancelled":
            terminal_kind = "cancelled"
            terminal_clock = scheduled_departure
            terminal_final_airport = prior_airport
        elif resolved.kind == "invalid_outcome":
            # The record truthfully exposes bad donor timing but is itself invalid
            # for scoring, as signaled by validity=False in episode_metrics.
            return False
        elif resolved.kind == "missed_departure":
            terminal_kind = "missed_departure"
            terminal_clock = prior_clock
            terminal_final_airport = prior_airport
        elif resolved.kind == "unresolved_diversion":
            terminal_kind = "unresolved_diversion"
            elapsed_to = (
                resolved.arrival_utc if resolved.arrival_utc is not None else resolved.departure_utc
            )
            assert elapsed_to is not None
            terminal_clock = min(request.horizon_utc, max(prior_clock, elapsed_to))
            terminal_final_airport = prior_airport
        else:
            assert resolved.arrival_utc is not None
            if resolved.arrival_utc > request.horizon_utc or (
                resolved.arrival_utc == request.horizon_utc
                and resolved.detail_airport != request.destination
            ):
                terminal_kind = "horizon"
                terminal_clock = request.horizon_utc
                terminal_final_airport = (
                    resolved.detail_airport
                    if resolved.arrival_utc == request.horizon_utc
                    else prior_airport
                )
            else:
                prior_airport = resolved.detail_airport or prior_airport
                prior_clock = resolved.arrival_utc
                if prior_airport == request.destination:
                    terminal_kind = "arrived"
                    terminal_clock = prior_clock
                    terminal_final_airport = prior_airport
                elif index + 1 == request.max_attempts:
                    terminal_kind = "max_attempts"
                    terminal_clock = prior_clock
                    terminal_final_airport = prior_airport

        if terminal_kind is not None and not is_last:
            return False

    reason = record.termination_reason
    if terminal_kind is not None:
        if (
            reason != terminal_kind
            or record.clock_utc != terminal_clock
            or record.final_airport != terminal_final_airport
        ):
            return False
    else:
        # no_candidates and invalid_action happen between selections and add no leg.
        if reason not in {"no_candidates", "invalid_action"}:
            return False
        if record.clock_utc != prior_clock or record.final_airport != prior_airport:
            return False
    return True


def _analyze_episode(record: object) -> _EpisodeAnalysis:
    validity = _record_is_valid(record)
    if isinstance(record, EpisodeRecord) and isinstance(record.legs, tuple):
        attempts = len(record.legs)
        disrupted = any(
            isinstance(leg, LegRecord)
            and isinstance(leg.outcome, SampledOutcome)
            and (leg.outcome.cancelled or leg.outcome.diverted)
            for leg in record.legs
        )
    else:
        attempts = 0
        disrupted = False

    reason = (
        record.termination_reason
        if isinstance(record, EpisodeRecord) and isinstance(record.termination_reason, str)
        else "invalid_record"
    )
    elapsed: int | None = None
    if isinstance(record, EpisodeRecord) and isinstance(record.request, TripRequest):
        ready = _utc_int(record.request.ready_utc)
        clock = _utc_int(record.clock_utc)
        if ready is not None and clock is not None and clock >= ready:
            elapsed = clock - ready

    arrived = bool(validity and reason == "arrived")
    on_time = bool(arrived and record.clock_utc <= record.request.deadline_utc)
    return _EpisodeAnalysis(
        validity=validity,
        arrived=arrived,
        on_time_arrival=on_time,
        elapsed_min=elapsed,
        disrupted=bool(disrupted),
        attempts=attempts,
        termination_reason=reason,
    )


def episode_metrics(record: EpisodeRecord) -> dict[str, bool | int | str | None]:
    """Return a JSON-safe, deterministic episode summary."""

    analysis = _analyze_episode(record)
    return {
        "validity": analysis.validity,
        "arrived": analysis.arrived,
        "on_time_arrival": analysis.on_time_arrival,
        "elapsed_min": analysis.elapsed_min,
        "disrupted": analysis.disrupted,
        "attempts": analysis.attempts,
        "termination_reason": analysis.termination_reason,
    }


PRIMARY_SCORE_PROFILE = "deadline_first_v1"
LEGACY_SIX_V1_PROFILE = "legacy_six_v1"

LEGACY_SIX_V1_WEIGHTS = MappingProxyType(
    {
        "arrived": 0.40,
        "on_time_arrival": 0.25,
        "total_delay": 0.10,
        "connections_count": 0.10,
        "cancellation_exposure": 0.05,
        "connection_buffer": 0.10,
    }
)
DEADLINE_FIRST_V1_WEIGHTS = MappingProxyType(
    {
        "on_time_arrival": 0.80,
        "arrived": 0.10,
        "earliness": 0.10,
    }
)
if math.fsum(LEGACY_SIX_V1_WEIGHTS.values()) != 1.0:
    raise RuntimeError("legacy six-v1 weights must sum to one")
if math.fsum(DEADLINE_FIRST_V1_WEIGHTS.values()) != 1.0:
    raise RuntimeError("deadline-first reward weights must sum to one")
CONNECTION_BUFFER_TARGET_MIN = 90


def validated_trip_scores(
    request: TripRequest, legs: tuple[LegRecord, ...], *, arrived: bool, clock_utc: int
) -> dict[str, float]:
    """Return raw rubric scores for an already validated core or recovery trip.

    The caller must first establish that ``request``, ``legs``, ``arrived``, and
    ``clock_utc`` form one consistent trip. This helper deliberately performs no
    record or recovery-chain validation.
    """

    if not arrived:
        return dict.fromkeys(LEGACY_SIX_V1_WEIGHTS, 0.0)
    completed = tuple(
        leg for leg in legs if not leg.outcome.cancelled and leg.arrival_utc is not None
    )
    # Sum positive arrival lateness once per flown leg; early legs cannot erase a later delay.
    delay = sum(max(0, leg.arrival_utc - leg.flight.scheduled_arrival_utc) for leg in completed)
    gaps = [
        following.flight.scheduled_departure_utc - previous.flight.scheduled_arrival_utc
        for previous, following in pairwise(completed)
    ]
    return {
        "arrived": 1.0,
        "on_time_arrival": float(clock_utc <= request.deadline_utc),
        "total_delay": max(0.0, 1.0 - delay / request.delay_budget_min),
        "connections_count": 1.0 / max(1, len(completed)),
        "cancellation_exposure": float(not any(leg.outcome.cancelled for leg in legs)),
        "connection_buffer": (
            max(0.0, min(1.0, min(gaps) / CONNECTION_BUFFER_TARGET_MIN)) if gaps else 1.0
        ),
    }


def validated_deadline_first_scores(
    request: TripRequest, *, arrived: bool, clock_utc: int
) -> dict[str, float]:
    """Return primary raw scores after the caller validates the complete trip."""

    if not arrived:
        return dict.fromkeys(DEADLINE_FIRST_V1_WEIGHTS, 0.0)
    span = request.horizon_utc - request.ready_utc
    earliness = max(0.0, min(1.0, (request.horizon_utc - clock_utc) / span))
    return {
        "on_time_arrival": float(clock_utc <= request.deadline_utc),
        "arrived": 1.0,
        "earliness": earliness,
    }


def _legacy_criterion_score(record: object, name: str) -> float:
    analysis = _analyze_episode(record)
    if not analysis.arrived:
        return 0.0
    assert isinstance(record, EpisodeRecord)
    return validated_trip_scores(
        record.request, record.legs, arrived=True, clock_utc=record.clock_utc
    )[name]


def _deadline_first_criterion_score(record: object, name: str) -> float:
    analysis = _analyze_episode(record)
    if not analysis.arrived:
        return 0.0
    assert isinstance(record, EpisodeRecord)
    return validated_deadline_first_scores(
        record.request, arrived=True, clock_utc=record.clock_utc
    )[name]


class _EpisodeVerifier(Verifier):
    """Validate once before evaluating one immutable episode score profile."""

    def __init__(
        self,
        criteria: Iterable[Criterion],
        score_builder: Callable[[EpisodeRecord, _EpisodeAnalysis], dict[str, float]],
    ) -> None:
        super().__init__(criteria)
        self._score_builder = score_builder

    def score(self, episode: Any) -> VerificationResult:
        analysis = _analyze_episode(episode)
        if analysis.validity:
            assert isinstance(episode, EpisodeRecord)
            raw_scores = self._score_builder(episode, analysis)
        else:
            raw_scores = dict.fromkeys((item.name for item in self.criteria), 0.0)
        return self._result_from_scores(raw_scores[item.name] for item in self.criteria)


def make_default_verifier() -> Verifier:
    """Construct the primary deadline-first-v1 episode verifier."""

    descriptions = {
        "on_time_arrival": "Reached the requested destination by its deadline.",
        "arrived": "Reached the requested destination by the episode horizon.",
        "earliness": "Fraction of the fixed request horizon remaining at destination arrival.",
    }
    return _EpisodeVerifier(
        (
            Criterion(
                name=name,
                weight=weight,
                description=descriptions[name],
                score_fn=lambda record, name=name: _deadline_first_criterion_score(record, name),
            )
            for name, weight in DEADLINE_FIRST_V1_WEIGHTS.items()
        ),
        lambda episode, analysis: validated_deadline_first_scores(
            episode.request,
            arrived=analysis.arrived,
            clock_utc=episode.clock_utc,
        ),
    )


def make_legacy_six_v1_verifier() -> Verifier:
    """Construct the frozen six-signal compatibility verifier."""

    descriptions = {
        "arrived": "Reached the requested destination by the episode horizon.",
        "on_time_arrival": "Reached the requested destination by its deadline.",
        "total_delay": "Positive arrival delay summed across flown legs, scaled by the request delay budget (default 180 minutes).",
        "connections_count": "Reciprocal of flown legs: one for nonstop, one half for one connection.",
        "cancellation_exposure": "Completed the trip without selecting a cancelled flight.",
        "connection_buffer": "Worst scheduled connection slack, capped at a 90-minute target.",
    }
    return _EpisodeVerifier(
        (
            Criterion(
                name=name,
                weight=weight,
                description=descriptions[name],
                score_fn=lambda record, name=name: _legacy_criterion_score(record, name),
            )
            for name, weight in LEGACY_SIX_V1_WEIGHTS.items()
        ),
        lambda episode, analysis: validated_trip_scores(
            episode.request,
            episode.legs,
            arrived=analysis.arrived,
            clock_utc=episode.clock_utc,
        ),
    )


_DEFAULT_VERIFIER = make_default_verifier()
_LEGACY_SIX_V1_VERIFIER = make_legacy_six_v1_verifier()


def verify_episode(record: EpisodeRecord) -> VerificationResult:
    """Validate and score one episode with the primary deadline-first-v1 profile."""

    return _DEFAULT_VERIFIER.score(record)


def verify_deadline_first(record: EpisodeRecord) -> VerificationResult:
    """Compatible entry point for the primary deadline-first-v1 verifier.

    This is a scalar episode preference: all on-time arrivals outrank all late
    arrivals, and earlier arrivals are strictly better within either group for
    a fixed request. It does not imply strict lexicographic optimization over
    policy-level expected success probabilities.
    """

    return verify_episode(record)


def verify_legacy_six_v1(record: EpisodeRecord) -> VerificationResult:
    """Validate and score one episode with the frozen legacy six-signal profile."""

    return _LEGACY_SIX_V1_VERIFIER.score(record)
