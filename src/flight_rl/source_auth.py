"""Authenticate episode source claims before applying record-only scoring.

The trusted data source resolves canonical schedules and transition outcomes.
Record fields are never used to select a comparable-flight pool.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Literal, Protocol

from flight_rl.models import EpisodeRecord, FlightCandidate, LegRecord, SampledOutcome, TripRequest
from flight_rl.verifier import VerificationResult

ScoreProfile = Literal["deadline_first", "legacy_six_v1"]


class SourceEvidenceProvider(Protocol):
    """Narrow trusted-source contract used by :class:`SourceBackedVerifier`."""

    def source_flight(self, flight_id: str) -> FlightCandidate | None: ...

    def source_transition_outcome(self, flight_id: str, donor_id: str) -> SampledOutcome | None: ...


@dataclass(frozen=True)
class SourceAuthentication:
    authenticated: bool
    reason: str
    leg_index: int | None = None


@dataclass(frozen=True)
class SourceVerification:
    authentication: SourceAuthentication
    score: VerificationResult


def _primitive_payload(value: object, model: type) -> tuple | None:
    """Compare only schema primitives, never submitted objects' overloaded equality."""
    if type(value) is not model:
        return None
    integers = {
        "scheduled_departure_utc",
        "scheduled_arrival_utc",
        "dep_hour_bucket",
        "support",
        "ready_utc",
        "deadline_utc",
        "horizon_utc",
        "max_attempts",
        "min_connection_min",
        "delay_budget_min",
        "clock_utc",
    }
    booleans = {"cancelled", "diverted", "div_reached_dest"}
    numbers = {
        "dep_delay_min",
        "arr_delay_min",
        "actual_elapsed_min",
        "div_arr_delay_min",
        "div_actual_elapsed_min",
    }
    payload = []
    for field in fields(model):
        item = getattr(value, field.name)
        nested_models = {
            "request": TripRequest,
            "flight": FlightCandidate,
            "outcome": SampledOutcome,
        }
        if field.name in nested_models:
            item = _primitive_payload(item, nested_models[field.name])
            if item is None:
                return None
        elif field.name == "legs":
            if type(item) is not tuple:
                return None
            item = tuple(_primitive_payload(leg, LegRecord) for leg in item)
            if any(leg is None for leg in item):
                return None
        elif field.name in {"departure_utc", "arrival_utc"}:
            if item is not None and type(item) is not int:
                return None
        elif field.name in integers:
            if type(item) is not int:
                return None
        elif field.name in booleans:
            if type(item) is not bool:
                return None
        elif field.name in numbers:
            if item is not None:
                if type(item) not in (int, float):
                    return None
                try:
                    item = float(item)
                except OverflowError:
                    return None
                if not math.isfinite(item):
                    return None
        elif field.name in {"div_airport", "resolved_airport"} and item is None:
            pass
        elif type(item) is not str:
            return None
        payload.append(item)
    return tuple(payload)


def verify_record_only(
    record: object, profile: ScoreProfile = "deadline_first"
) -> VerificationResult:
    """Score internal/trusted records without authenticating dataset membership.

    Published or externally supplied records should use :class:`SourceBackedVerifier`.
    """

    scorer: Callable[[object], VerificationResult]
    if profile == "deadline_first":
        from flight_rl.verifier import verify_deadline_first

        scorer = verify_deadline_first
    elif profile == "legacy_six_v1":
        from flight_rl.verifier import verify_legacy_six_v1

        scorer = verify_legacy_six_v1
    else:
        raise ValueError(f"Unknown score profile {profile!r}")
    return scorer(record)


class SourceBackedVerifier:
    """Hard-gate record-only scores on canonical source membership and payloads."""

    def __init__(self, data: SourceEvidenceProvider) -> None:
        for method_name in ("source_flight", "source_transition_outcome"):
            if not callable(getattr(data, method_name, None)):
                raise TypeError(f"data must provide callable {method_name}()")
        self._data = data

    def authenticate_flight(self, flight: object) -> SourceAuthentication:
        submitted = _primitive_payload(flight, FlightCandidate)
        if submitted is None or not flight.flight_id:
            return SourceAuthentication(False, "invalid_flight")
        canonical = self._data.source_flight(flight.flight_id)
        if canonical is None:
            return SourceAuthentication(False, "unknown_or_ineligible_flight")
        expected = _primitive_payload(canonical, FlightCandidate)
        if expected is None:
            return SourceAuthentication(False, "invalid_source_flight")
        if submitted != expected:
            return SourceAuthentication(False, "flight_payload_mismatch")
        return SourceAuthentication(True, "authenticated")

    def authenticate_outcome(self, flight: object, outcome: object) -> SourceAuthentication:
        flight_auth = self.authenticate_flight(flight)
        if not flight_auth.authenticated:
            return flight_auth
        assert isinstance(flight, FlightCandidate)
        submitted = _primitive_payload(outcome, SampledOutcome)
        if submitted is None or not outcome.donor_id:
            return SourceAuthentication(False, "invalid_outcome")
        canonical = self._data.source_transition_outcome(flight.flight_id, outcome.donor_id)
        if canonical is None:
            return SourceAuthentication(False, "unknown_or_ineligible_donor")
        expected = _primitive_payload(canonical, SampledOutcome)
        if expected is None:
            return SourceAuthentication(False, "invalid_source_outcome")
        if submitted != expected:
            return SourceAuthentication(False, "outcome_payload_mismatch")
        return SourceAuthentication(True, "authenticated")

    def authenticate_episode(self, record: object) -> SourceAuthentication:
        if _primitive_payload(record, EpisodeRecord) is None:
            return SourceAuthentication(False, "invalid_episode")
        for index, leg in enumerate(record.legs):
            if not isinstance(leg, LegRecord):
                return SourceAuthentication(False, "invalid_leg", index)
            authentication = self.authenticate_outcome(leg.flight, leg.outcome)
            if not authentication.authenticated:
                return SourceAuthentication(False, authentication.reason, index)
        return SourceAuthentication(True, "authenticated")

    def verify(
        self, record: object, profile: ScoreProfile = "deadline_first"
    ) -> SourceVerification:
        authentication = self.authenticate_episode(record)
        # Never invoke record arithmetic or overloaded equality on rejected payloads.
        score = verify_record_only(record if authentication.authenticated else object(), profile)
        return SourceVerification(authentication=authentication, score=score)
