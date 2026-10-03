"""Separate schedule, fitted-information and transition distributions for evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from numbers import Integral
from types import MappingProxyType

from flight_rl.models import FlightCandidate, FlightDataSource, OutcomeSummary, SampledOutcome


class SplitFlightData:
    """Expose fit-only information while sampling a distinct transition window.

    The transition pool is an environment interface. Policies use only candidates,
    outcome_summary and historical_outcomes. Python object access is not a security boundary.
    """

    def __init__(
        self,
        schedule_data: FlightDataSource,
        fit_data: FlightDataSource,
        outcome_data: FlightDataSource,
        *,
        candidate_routes: Iterable[tuple[str, str]] | None = None,
    ) -> None:
        self._schedule = schedule_data
        self._fit = fit_data
        self._outcomes = outcome_data
        self._candidate_routes = self._validate_candidate_routes(candidate_routes)
        self.airports = tuple(sorted(set(schedule_data.airports) | set(fit_data.airports)))
        self.carriers = tuple(sorted(set(schedule_data.carriers) | set(fit_data.carriers)))
        self._excluded_no_fit: set[str] = set()
        self._excluded_outside_catalog: set[str] = set()
        self._transition_pool = lru_cache(maxsize=512)(outcome_data.historical_outcomes)

    @staticmethod
    def _validate_candidate_routes(
        routes: Iterable[tuple[str, str]] | None,
    ) -> frozenset[tuple[str, str]] | None:
        if routes is None:
            return None
        if isinstance(routes, (str, bytes)):
            raise TypeError("candidate_routes must be an iterable of airport-code pairs")
        result: set[tuple[str, str]] = set()
        try:
            values = tuple(routes)
        except TypeError as exc:
            raise TypeError("candidate_routes must be an iterable of airport-code pairs") from exc
        for route in values:
            if (
                not isinstance(route, tuple)
                or len(route) != 2
                or any(not isinstance(code, str) or not code.strip() for code in route)
            ):
                raise ValueError("candidate_routes must contain pairs of nonempty airport codes")
            origin, destination = (code.strip().upper() for code in route)
            if origin == destination:
                raise ValueError("candidate_routes cannot contain same-airport routes")
            result.add((origin, destination))
        return frozenset(result)

    @property
    def metadata(self):
        return MappingProxyType(
            {
                "information_window": dict(getattr(self._fit, "metadata", {})),
                "transition_window": dict(getattr(self._outcomes, "metadata", {})),
                "schedule_window": dict(getattr(self._schedule, "metadata", {})),
                "unique_queried_schedules_without_fit_coverage": len(self._excluded_no_fit),
                "candidate_route_filter": (
                    [list(route) for route in sorted(self._candidate_routes)]
                    if self._candidate_routes is not None
                    else None
                ),
                "unique_queried_schedules_outside_candidate_routes": len(
                    self._excluded_outside_catalog
                ),
                "scope": "Historical schedule catalog; fit-only priors/history; separate empirical "
                "transition pools. No seat-inventory or causal passenger claim.",
            }
        )

    def candidates(self, origin: str, earliest_utc: int, horizon_utc: int, limit: int = 64):
        if isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 1:
            raise ValueError("limit must be a positive integer")
        requested = int(limit)
        while True:
            catalog = self._schedule.candidates(origin, earliest_utc, horizon_utc, requested)
            retained = []
            for flight in catalog:
                route = (flight.origin, flight.dest)
                if self._candidate_routes is not None and route not in self._candidate_routes:
                    self._excluded_outside_catalog.add(flight.flight_id)
                    continue
                if self._fit.outcome_summary(flight).support <= 0:
                    self._excluded_no_fit.add(flight.flight_id)
                    if self._candidate_routes is not None:
                        raise ValueError(
                            "candidate_routes includes a route without coverage in the fit source: "
                            f"{flight.origin}-{flight.dest}"
                        )
                    continue
                retained.append(flight)
                if len(retained) == limit:
                    return tuple(retained)
            if len(catalog) < requested:
                return tuple(retained)
            requested *= 2

    def outcome_summary(self, flight: FlightCandidate) -> OutcomeSummary:
        return self._fit.outcome_summary(flight)

    @staticmethod
    def _source_method(data: object, name: str):
        method = getattr(data, name, None)
        if not callable(method):
            raise TypeError(f"wrapped source must provide callable {name}()")
        return method

    def source_flight(self, flight_id: str) -> FlightCandidate | None:
        """Resolve schedules through the configured catalog and fit-coverage gate."""

        canonical = self._source_method(self._schedule, "source_flight")(flight_id)
        if canonical is None:
            return None
        if not isinstance(canonical, FlightCandidate):
            raise TypeError("schedule source_flight() must return FlightCandidate or None")
        route = (canonical.origin, canonical.dest)
        if self._candidate_routes is not None and route not in self._candidate_routes:
            return None
        if self._fit.outcome_summary(canonical).support <= 0:
            return None
        return canonical

    def source_transition_outcome(self, flight_id: str, donor_id: str) -> SampledOutcome | None:
        """Resolve provenance from held-out transitions, never fit-only history."""

        canonical = self.source_flight(flight_id)
        if canonical is None:
            return None
        outcome_flight = self._source_method(self._outcomes, "source_flight")(flight_id)
        if outcome_flight != canonical:
            return None
        return self._source_method(self._outcomes, "source_transition_outcome")(flight_id, donor_id)

    def historical_outcomes(self, flight: FlightCandidate) -> tuple[SampledOutcome, ...]:
        return self._fit.historical_outcomes(flight)

    def transition_outcomes(self, flight: FlightCandidate) -> tuple[SampledOutcome, ...]:
        return self._transition_pool(flight)

    def sample_outcome(self, flight: FlightCandidate, rng) -> SampledOutcome:
        return self._outcomes.sample_outcome(flight, rng)
