"""Shared contracts. Absolute times are integer UTC epoch minutes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import numpy as np

CANDIDATE_FEATURES = (
    "destination_index",
    "carrier_index",
    "departure_in_min",
    "arrival_in_min",
    "scheduled_elapsed_min",
    "p_cancelled",
    "p_diverted",
    "mean_arrival_delay_min",
    "support",
)
DEFAULT_AIRPORTS = (
    "ATL",
    "BOS",
    "DEN",
    "DFW",
    "EWR",
    "JFK",
    "LAX",
    "MIA",
    "ORD",
    "SEA",
    "SFO",
    "SLC",
)


def utc_minutes(value: str | datetime) -> int:
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Timestamp must include a UTC offset")
    return int(parsed.timestamp() // 60)


def iso_utc(value: int) -> str:
    return datetime.fromtimestamp(value * 60, UTC).isoformat()


@dataclass(frozen=True)
class TripRequest:
    """A deadline may already be expired; eventual arrival remains possible until horizon."""

    origin: str
    destination: str
    ready_utc: int
    deadline_utc: int
    horizon_utc: int
    max_attempts: int = 3
    min_connection_min: int = 45
    delay_budget_min: int = 180

    def __post_init__(self) -> None:
        if not self.origin or not self.destination or self.origin == self.destination:
            raise ValueError("Origin and destination must be distinct nonempty airports")
        if not (self.ready_utc < self.horizon_utc and self.deadline_utc <= self.horizon_utc):
            raise ValueError("Require ready < horizon and deadline <= horizon")
        if self.max_attempts < 1 or self.min_connection_min < 0 or self.delay_budget_min <= 0:
            raise ValueError("Invalid attempt, connection or delay budget")


@dataclass(frozen=True)
class FlightCandidate:
    flight_id: str
    origin: str
    dest: str
    carrier: str
    scheduled_departure_utc: int
    scheduled_arrival_utc: int
    flight_date: str
    dep_hour_bucket: int
    season: str
    operating_carrier: str = ""
    operating_flight_number: str = ""

    @property
    def scheduled_elapsed_min(self) -> int:
        return self.scheduled_arrival_utc - self.scheduled_departure_utc


@dataclass(frozen=True)
class SampledOutcome:
    donor_id: str
    cancelled: bool = False
    diverted: bool = False
    dep_delay_min: float | None = 0.0
    arr_delay_min: float | None = 0.0
    actual_elapsed_min: float | None = None
    div_reached_dest: bool = False
    div_arr_delay_min: float | None = None
    div_actual_elapsed_min: float | None = None
    div_airport: str | None = None
    support: int = 1
    fallback_level: str = "fixture"


@dataclass(frozen=True)
class OutcomeSummary:
    p_cancelled: float
    p_diverted: float
    mean_arrival_delay_min: float
    support: int
    fallback_level: str


@dataclass(frozen=True)
class LegRecord:
    flight: FlightCandidate
    outcome: SampledOutcome
    departure_utc: int | None
    arrival_utc: int | None
    resolved_airport: str | None


@dataclass(frozen=True)
class EpisodeRecord:
    request: TripRequest
    legs: tuple[LegRecord, ...]
    final_airport: str
    clock_utc: int
    termination_reason: str


class FlightDataSource(Protocol):
    airports: tuple[str, ...]
    carriers: tuple[str, ...]

    def candidates(
        self, origin: str, earliest_utc: int, horizon_utc: int, limit: int = 64
    ) -> tuple[FlightCandidate, ...]: ...

    def sample_outcome(
        self, flight: FlightCandidate, rng: np.random.Generator
    ) -> SampledOutcome: ...

    def outcome_summary(self, flight: FlightCandidate) -> OutcomeSummary: ...

    def historical_outcomes(self, flight: FlightCandidate) -> tuple[SampledOutcome, ...]: ...
