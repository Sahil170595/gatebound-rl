"""Small synthetic scenario for executable examples, never historical-data evidence."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from flight_rl.models import (
    FlightCandidate,
    OutcomeSummary,
    SampledOutcome,
    TripRequest,
    utc_minutes,
)


class DemoFlightData:
    """In-memory FlightDataSource with explicit empirical fixture pools."""

    def __init__(
        self, flights: tuple[FlightCandidate, ...], pools: dict[str, tuple[SampledOutcome, ...]]
    ) -> None:
        self.flights = tuple(
            sorted(
                flights,
                key=lambda f: (
                    f.scheduled_departure_utc,
                    f.scheduled_arrival_utc,
                    f.carrier,
                    f.flight_id,
                ),
            )
        )
        self.airports = tuple(sorted({x for f in flights for x in (f.origin, f.dest)}))
        self.carriers = tuple(sorted({f.carrier for f in flights}))
        self._pools = {key: tuple(value) for key, value in pools.items()}
        self._source_flights = {flight.flight_id: flight for flight in self.flights}
        if len(self._source_flights) != len(self.flights):
            raise ValueError("fixture flight_id values must be unique")

    def candidates(
        self, origin: str, earliest_utc: int, horizon_utc: int, limit: int = 64
    ) -> tuple[FlightCandidate, ...]:
        return tuple(
            f
            for f in self.flights
            if f.origin == origin and earliest_utc <= f.scheduled_departure_utc <= horizon_utc
        )[:limit]

    def historical_outcomes(self, flight: FlightCandidate) -> tuple[SampledOutcome, ...]:
        pool = self._pools[flight.flight_id]
        return tuple(replace(x, support=len(pool)) for x in pool)

    def source_flight(self, flight_id: str) -> FlightCandidate | None:
        if not isinstance(flight_id, str) or not flight_id:
            return None
        return self._source_flights.get(flight_id)

    def source_transition_outcome(self, flight_id: str, donor_id: str) -> SampledOutcome | None:
        flight = self.source_flight(flight_id)
        if flight is None or not isinstance(donor_id, str) or not donor_id:
            return None
        pool = self._pools.get(flight.flight_id, ())
        matching = tuple(outcome for outcome in pool if outcome.donor_id == donor_id)
        if len(matching) != 1:
            return None
        return replace(matching[0], support=len(pool))

    def sample_outcome(self, flight: FlightCandidate, rng: np.random.Generator) -> SampledOutcome:
        pool = self.historical_outcomes(flight)
        return pool[int(rng.integers(len(pool)))]

    def outcome_summary(self, flight: FlightCandidate) -> OutcomeSummary:
        pool = self.historical_outcomes(flight)
        delays = [
            x.arr_delay_min
            for x in pool
            if not x.cancelled and not x.diverted and x.arr_delay_min is not None
        ]
        return OutcomeSummary(
            sum(x.cancelled for x in pool) / len(pool),
            sum(x.diverted and not x.cancelled for x in pool) / len(pool),
            float(np.mean(delays)) if delays else 0.0,
            len(pool),
            "fixture",
        )


def make_demo_scenario() -> tuple[TripRequest, DemoFlightData]:
    """Return a direct-versus-connection scenario with rare disrupted outcomes."""
    ready = utc_minutes("2024-01-15T12:00:00Z")
    request = TripRequest("SFO", "JFK", ready, ready + 660, ready + 1440)

    def flight(name: str, origin: str, dest: str, depart: int, arrive: int) -> FlightCandidate:
        return FlightCandidate(
            name, origin, dest, "DEMO", ready + depart, ready + arrive, "2024-01-15", 2, "DJF"
        )

    flights = (
        flight("demo-SFO-DEN", "SFO", "DEN", 60, 210),
        flight("demo-SFO-ORD", "SFO", "ORD", 120, 360),
        flight("demo-SFO-JFK", "SFO", "JFK", 180, 510),
        flight("demo-DEN-JFK", "DEN", "JFK", 300, 510),
        flight("demo-ORD-JFK", "ORD", "JFK", 420, 570),
    )
    pools: dict[str, tuple[SampledOutcome, ...]] = {}
    for f in flights:
        ordinary = [
            SampledOutcome(
                f"{f.flight_id}-ok-{i}", actual_elapsed_min=float(f.scheduled_elapsed_min)
            )
            for i in range(8)
        ]
        late = SampledOutcome(
            f"{f.flight_id}-late",
            dep_delay_min=180.0,
            arr_delay_min=180.0,
            actual_elapsed_min=float(f.scheduled_elapsed_min),
        )
        cancelled = SampledOutcome(
            f"{f.flight_id}-cancel", cancelled=True, dep_delay_min=None, arr_delay_min=None
        )
        pools[f.flight_id] = (*ordinary, late, cancelled)
    direct = flights[2]
    pools[direct.flight_id] = (
        *pools[direct.flight_id][:-2],
        SampledOutcome(
            "demo-direct-diversion",
            diverted=True,
            arr_delay_min=None,
            div_reached_dest=True,
            div_arr_delay_min=180.0,
            div_actual_elapsed_min=510.0,
            div_airport="ORD",
        ),
        SampledOutcome(
            "demo-direct-cancel", cancelled=True, dep_delay_min=None, arr_delay_min=None
        ),
    )
    return request, DemoFlightData(flights, pools)
