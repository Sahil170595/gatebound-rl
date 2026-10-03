"""Committed-itinerary flight routing with fixed published boarding cutoffs.

The complete itinerary is selected from schedules before ``reset`` reveals any
outcome.  A published onward departure is treated as the boarding cutoff: an
inbound arrival is catchable at equality, but no future outbound delay is
sampled to rescue a missed connection.  This models neither passenger boarding
events nor seat inventory.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, replace
from itertools import count
from numbers import Integral
from typing import Any, ClassVar, Self

import gymnasium as gym

from flight_rl.candidates import canonical_candidates
from flight_rl.env import FlightRouteEnv
from flight_rl.models import (
    EpisodeRecord,
    FlightCandidate,
    FlightDataSource,
    OutcomeSummary,
    SampledOutcome,
    TripRequest,
)
from flight_rl.verifier import VerificationResult, episode_metrics, verify_deadline_first

_IN_PROGRESS = "in_progress"
_MISSED_CONNECTION = "missed_connection"


@dataclass(frozen=True)
class PrebookedEpisodeRecord:
    """One immutable booking and the exact core episode executed against it."""

    request: TripRequest
    itinerary: tuple[FlightCandidate, ...]
    episode: EpisodeRecord
    termination_reason: str
    missed_flight_id: str | None


@dataclass(frozen=True)
class PrebookedVerificationResult:
    """Structural validity, source authentication and primary episode score."""

    validity: bool
    source_authenticated: bool
    reason: str
    score: VerificationResult


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _request_problem(request: object) -> str | None:
    if type(request) is not TripRequest:
        return "request must be a TripRequest"
    if (
        not all(
            type(airport) is str and bool(airport)
            for airport in (request.origin, request.destination)
        )
        or request.origin == request.destination
    ):
        return "request airports must be distinct nonempty strings"
    values = (
        request.ready_utc,
        request.deadline_utc,
        request.horizon_utc,
        request.max_attempts,
        request.min_connection_min,
        request.delay_budget_min,
    )
    if any(type(value) is not int for value in values):
        return "request time and budget fields must be integers"
    if request.ready_utc >= request.horizon_utc or request.deadline_utc > request.horizon_utc:
        return "request has invalid ready, deadline, or horizon bounds"
    if request.max_attempts < 1 or request.min_connection_min < 0:
        return "request has an invalid attempt or connection budget"
    if request.delay_budget_min <= 0:
        return "request delay scoring budget must be positive"
    return None


def _itinerary_problem(request: object, itinerary: object) -> str | None:
    request_problem = _request_problem(request)
    if request_problem is not None:
        return request_problem
    assert isinstance(request, TripRequest)
    if not isinstance(itinerary, tuple) or not itinerary:
        return "itinerary must be a nonempty tuple"
    if len(itinerary) > request.max_attempts:
        return "itinerary exceeds the request attempt budget"

    prior_airport = request.origin
    cutoff = request.ready_utc + request.min_connection_min
    for index, flight in enumerate(itinerary):
        if not isinstance(flight, FlightCandidate):
            return f"itinerary leg {index} is not a FlightCandidate"
        if not all(
            isinstance(value, str) and bool(value)
            for value in (flight.flight_id, flight.origin, flight.dest, flight.carrier)
        ):
            return f"itinerary leg {index} has an invalid route or identity"
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in (flight.scheduled_departure_utc, flight.scheduled_arrival_utc)
        ):
            return f"itinerary leg {index} has non-integer schedule times"
        if flight.origin == flight.dest:
            return f"itinerary leg {index} has identical endpoints"
        if flight.scheduled_arrival_utc <= flight.scheduled_departure_utc:
            return f"itinerary leg {index} has a nonpositive scheduled duration"
        if flight.origin != prior_airport:
            return f"itinerary leg {index} breaks airport continuity"
        if flight.scheduled_departure_utc < cutoff:
            return f"itinerary leg {index} violates the scheduled connection minimum"
        if flight.scheduled_arrival_utc > request.horizon_utc:
            return f"itinerary leg {index} arrives after the request horizon"
        if index < len(itinerary) - 1 and flight.dest == request.destination:
            return "itinerary reaches the destination before its final leg"
        prior_airport = flight.dest
        cutoff = flight.scheduled_arrival_utc + request.min_connection_min

    if itinerary[-1].dest != request.destination:
        return "itinerary does not end at the requested destination"
    return None


def _validate_itinerary(
    request: TripRequest, itinerary: tuple[FlightCandidate, ...]
) -> tuple[FlightCandidate, ...]:
    problem = _itinerary_problem(request, itinerary)
    if problem is not None:
        raise ValueError(problem)
    return itinerary


class _ItineraryData:
    """Expose only the next booked flight while delegating outcome mechanics."""

    def __init__(self, data: FlightDataSource, itinerary: tuple[FlightCandidate, ...]) -> None:
        self._data = data
        self._itinerary = itinerary
        self._index = 0
        self.airports = tuple(data.airports)
        self.carriers = tuple(data.carriers)

    def reset(self) -> None:
        self._index = 0

    def candidates(
        self, origin: str, earliest_utc: int, horizon_utc: int, limit: int = 64
    ) -> tuple[FlightCandidate, ...]:
        if isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if self._index >= len(self._itinerary):
            return ()
        flight = self._itinerary[self._index]
        if (
            flight.origin != origin
            or flight.scheduled_departure_utc < earliest_utc
            or flight.scheduled_departure_utc > horizon_utc
        ):
            return ()
        return (flight,)

    def sample_outcome(self, flight: FlightCandidate, rng: Any) -> SampledOutcome:
        if self._index >= len(self._itinerary) or flight != self._itinerary[self._index]:
            raise ValueError("only the next booked flight may be sampled")
        outcome = self._data.sample_outcome(flight, rng)
        self._index += 1
        return outcome

    def outcome_summary(self, flight: FlightCandidate) -> OutcomeSummary:
        if self._index >= len(self._itinerary) or flight != self._itinerary[self._index]:
            raise ValueError("only the next booked flight has a visible summary")
        return self._data.outcome_summary(flight)

    def historical_outcomes(self, flight: FlightCandidate) -> tuple[SampledOutcome, ...]:
        return self._data.historical_outcomes(flight)


def _missed_flight(
    request: TripRequest,
    itinerary: tuple[FlightCandidate, ...],
    episode: EpisodeRecord,
) -> FlightCandidate | None:
    index = len(episode.legs)
    if episode.termination_reason != "no_candidates" or index == 0 or index >= len(itinerary):
        return None
    inbound = episode.legs[-1]
    onward = itinerary[index]
    if (
        inbound.arrival_utc is None
        or inbound.resolved_airport != onward.origin
        or inbound.arrival_utc + request.min_connection_min <= onward.scheduled_departure_utc
    ):
        return None
    return onward


class PrebookedRouteEnv(gym.Env[dict[str, Any], int]):
    """Execute a complete, immutable booking through ``FlightRouteEnv``."""

    metadata: ClassVar[dict[str, Any]] = FlightRouteEnv.metadata

    def __init__(
        self,
        request: TripRequest,
        data: FlightDataSource,
        itinerary: tuple[FlightCandidate, ...],
        *,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        committed = _validate_itinerary(request, itinerary)
        restricted = _ItineraryData(data, committed)
        airports = set(restricted.airports)
        carriers = set(restricted.carriers)
        if any(
            flight.origin not in airports
            or flight.dest not in airports
            or flight.carrier not in carriers
            for flight in committed
        ):
            raise ValueError("itinerary airports and carriers must be present in the data source")
        core = FlightRouteEnv(
            request,
            restricted,
            max_candidates=1,
            reward_mode="deadline_first",
            render_mode=render_mode,
        )
        self._request = request
        self._data = data
        self._itinerary = committed
        self._restricted = restricted
        self._core = core
        self.action_space = core.action_space
        self.observation_space = core.observation_space
        self.render_mode = render_mode
        self._record = self._make_record()

    @property
    def request(self) -> TripRequest:
        return self._request

    @property
    def data(self) -> FlightDataSource:
        return self._data

    @property
    def itinerary(self) -> tuple[FlightCandidate, ...]:
        return self._itinerary

    @property
    def available_flights(self) -> tuple[FlightCandidate, ...]:
        return self._core.available_flights

    @property
    def clock_utc(self) -> int:
        return self._core.clock_utc

    @property
    def reward_mode(self) -> str:
        return "deadline_first"

    @property
    def record(self) -> PrebookedEpisodeRecord:
        return self._record

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        self._restricted.reset()
        observation, info = self._core.reset(seed=seed, options=options)
        self._record = self._make_record()
        result = dict(info)
        result["core_episode"] = info["episode"]
        result["episode"] = self.record
        return observation, result

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self._core.step(action)
        self._record = self._make_record()
        result = dict(info)
        result["core_episode"] = info["episode"]
        result["episode"] = self.record
        result["core_verification"] = info.get("verification")
        result["missed_connection"] = self.record.termination_reason == _MISSED_CONNECTION
        result["missed_flight_id"] = self.record.missed_flight_id
        return observation, reward, terminated, truncated, result

    def render(self) -> str:
        return self._core.render()

    def close(self) -> None:
        self._core.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _make_record(self) -> PrebookedEpisodeRecord:
        episode = self._core.record
        missed = _missed_flight(self.request, self.itinerary, episode)
        reason = _MISSED_CONNECTION if missed is not None else episode.termination_reason
        return PrebookedEpisodeRecord(
            request=self.request,
            itinerary=self.itinerary,
            episode=episode,
            termination_reason=reason,
            missed_flight_id=missed.flight_id if missed is not None else None,
        )


def plan_scheduled_itinerary(
    request: TripRequest,
    data: FlightDataSource,
    *,
    route: tuple[str, ...] | None = None,
    max_candidates: int = 64,
    max_search_nodes: int = 10_000,
) -> tuple[FlightCandidate, ...]:
    """Commit the earliest-arriving bounded route using schedules only."""

    max_candidates = _positive_int(max_candidates, "max_candidates")
    max_search_nodes = _positive_int(max_search_nodes, "max_search_nodes")
    request_problem = _request_problem(request)
    if request_problem is not None:
        raise ValueError(request_problem)
    assert isinstance(request, TripRequest)
    if route is not None:
        if not isinstance(route, tuple) or any(
            not isinstance(airport, str) or not airport for airport in route
        ):
            raise ValueError("route must be a tuple of nonempty airport codes")
        if len(route) < 2 or route[0] != request.origin or route[-1] != request.destination:
            raise ValueError("route must run from the request origin to its destination")
        if request.destination in route[1:-1]:
            raise ValueError("route reaches the destination before its final airport")
        if len(route) - 1 > request.max_attempts:
            raise ValueError("route exceeds the request attempt budget")

    serial = count()
    frontier: list[tuple[int, int, int, str, int, tuple[FlightCandidate, ...]]] = [
        (request.ready_utc, 0, next(serial), request.origin, 0, ())
    ]
    # With a per-state candidate cap, a later arrival can expose flights omitted
    # from an earlier state's window. Only identical states may be collapsed.
    seen = {(request.origin, 0, 0, request.ready_utc)}
    expanded = 0
    while frontier:
        clock, attempts, _, airport, route_index, itinerary = heapq.heappop(frontier)
        expanded += 1
        if expanded > max_search_nodes:
            raise RuntimeError("schedule search exceeded max_search_nodes")
        if airport == request.destination:
            return _validate_itinerary(request, itinerary)
        if attempts >= request.max_attempts:
            continue

        flights = canonical_candidates(
            data,
            airport,
            clock + request.min_connection_min,
            request.horizon_utc,
            limit=max_candidates,
        )
        for flight in flights:
            if flight.scheduled_arrival_utc > request.horizon_utc:
                continue
            next_route_index = route_index
            if route is not None:
                if route_index + 1 >= len(route) or flight.dest != route[route_index + 1]:
                    continue
                next_route_index += 1
            elif flight.dest == request.destination:
                next_route_index = 1

            candidate_itinerary = (*itinerary, flight)
            arrival = flight.scheduled_arrival_utc
            key = (flight.dest, attempts + 1, next_route_index, arrival)
            if key in seen:
                continue
            seen.add(key)
            heapq.heappush(
                frontier,
                (
                    arrival,
                    attempts + 1,
                    next(serial),
                    flight.dest,
                    next_route_index,
                    candidate_itinerary,
                ),
            )
    raise ValueError("no schedule-feasible itinerary found within the search bounds")


def _zero_score(score: VerificationResult) -> VerificationResult:
    return VerificationResult(
        aggregate_score=0.0,
        breakdown=tuple(
            replace(item, raw_score=0.0, weighted_contribution=0.0) for item in score.breakdown
        ),
    )


def _structural_problem(record: PrebookedEpisodeRecord) -> str | None:
    problem = _itinerary_problem(record.request, record.itinerary)
    if problem is not None:
        return problem
    episode = record.episode
    if not isinstance(episode, EpisodeRecord):
        return "episode must be an EpisodeRecord"
    if _request_problem(episode.request) is not None:
        return "core episode request is invalid"
    if episode.request != record.request:
        return "core episode request differs from the booking request"
    if episode.termination_reason == _IN_PROGRESS:
        return "prebooked episode is not terminal"
    if not bool(episode_metrics(episode)["validity"]):
        return "core episode is invalid"
    if len(episode.legs) > len(record.itinerary):
        return "core episode contains more legs than the booking"
    for index, leg in enumerate(episode.legs):
        if leg.flight != record.itinerary[index]:
            return f"core episode leg {index} differs from the committed itinerary"

    missed = _missed_flight(record.request, record.itinerary, episode)
    if missed is not None:
        if record.termination_reason != _MISSED_CONNECTION:
            return "outer termination reason does not match the missed cutoff"
        if record.missed_flight_id != missed.flight_id:
            return "outer missed flight does not match the reconstructed cutoff failure"
        return None
    if episode.termination_reason == "no_candidates":
        return "core no_candidates is not explained by a missed booked connection"
    if record.termination_reason != episode.termination_reason:
        return "outer termination reason differs from the core episode"
    if record.missed_flight_id is not None:
        return "non-missed episode must not name a missed flight"
    if episode.termination_reason == "arrived" and len(episode.legs) != len(record.itinerary):
        return "arrival does not execute the complete committed itinerary"
    return None


def verify_prebooked(
    record: PrebookedEpisodeRecord, data: FlightDataSource
) -> PrebookedVerificationResult:
    """Reconstruct a booking and authenticate flown outcomes plus unflown schedules."""

    if (
        type(record) is not PrebookedEpisodeRecord
        or type(record.episode) is not EpisodeRecord
        or type(record.itinerary) is not tuple
        or not record.itinerary
        or type(record.termination_reason) is not str
        or (record.missed_flight_id is not None and type(record.missed_flight_id) is not str)
        or _request_problem(record.request) is not None
    ):
        return PrebookedVerificationResult(
            validity=False,
            source_authenticated=False,
            reason="invalid prebooked record",
            score=verify_deadline_first(object()),
        )

    from flight_rl.source_auth import SourceBackedVerifier

    source = SourceBackedVerifier(data)
    core = source.verify(record.episode, profile="deadline_first")
    source_authenticated = core.authentication.authenticated
    authentication_reason = core.authentication.reason
    authentication_leg = core.authentication.leg_index

    # Authenticate flown booking entries as well as the unflown suffix before
    # comparing dataclasses. Equality alone does not enforce primitive field types.
    safe_to_compare = source_authenticated
    if source_authenticated:
        for index, flight in enumerate(record.itinerary):
            authentication = source.authenticate_flight(flight)
            if not authentication.authenticated:
                source_authenticated = False
                authentication_reason = authentication.reason
                if authentication.reason == "invalid_flight":
                    safe_to_compare = False
                authentication_leg = (
                    index if authentication.leg_index is None else authentication.leg_index
                )

    structure = _structural_problem(record) if safe_to_compare else "invalid source payload shape"

    if structure is not None:
        reason = structure
    elif not source_authenticated:
        location = "" if authentication_leg is None else f" at leg {authentication_leg}"
        reason = f"source authentication failed{location}: {authentication_reason}"
    else:
        reason = "verified"

    complete_arrival = (
        structure is None
        and source_authenticated
        and record.termination_reason == "arrived"
        and len(record.episode.legs) == len(record.itinerary)
    )
    score = core.score if complete_arrival else _zero_score(core.score)
    return PrebookedVerificationResult(
        validity=structure is None,
        source_authenticated=source_authenticated,
        reason=reason,
        score=score,
    )
