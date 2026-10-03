"""Canonical flight selection shared by environment actions and planner lookahead."""

from numbers import Integral

from flight_rl.models import FlightCandidate, FlightDataSource


def canonical_candidates(
    data: FlightDataSource,
    origin: str,
    earliest_utc: int,
    horizon_utc: int,
    limit: int = 64,
) -> tuple[FlightCandidate, ...]:
    """Validate supplied rows, then apply the same boarding window, order and cap.

    The source selects the supplied set; this helper cannot recover rows that a source
    omitted before returning it. It never reads outcome donors or fitted summaries.
    """
    try:
        supplied = tuple(
            data.candidates(
                origin,
                earliest_utc,
                horizon_utc,
                limit=limit,
            )
        )
    except TypeError as exc:
        raise ValueError("data.candidates must return an iterable of flights") from exc

    eligible: list[FlightCandidate] = []
    for flight in supplied:
        if not isinstance(flight, FlightCandidate):
            raise ValueError(  # noqa: TRY004 - invalid data is a contract ValueError
                "data.candidates must return FlightCandidate values"
            )
        if (
            not isinstance(flight.origin, str)
            or not flight.origin
            or not isinstance(flight.dest, str)
            or not flight.dest
            or not isinstance(flight.carrier, str)
            or not flight.carrier
        ):
            raise ValueError("candidate route and carrier codes must be nonempty strings")
        if flight.origin != origin:
            raise ValueError("candidate origin does not match the requested airport")
        if not isinstance(flight.flight_id, str) or not flight.flight_id:
            raise ValueError("candidate flight_id must be nonempty")
        if flight.dest == flight.origin:
            raise ValueError("candidate origin and destination must differ")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in (
                flight.scheduled_departure_utc,
                flight.scheduled_arrival_utc,
            )
        ):
            raise ValueError("candidate schedule times must be integer UTC minutes")
        if flight.scheduled_arrival_utc <= flight.scheduled_departure_utc:
            raise ValueError("candidate scheduled arrival must follow departure")
        if flight.dest not in data.airports:
            raise ValueError(f"candidate destination {flight.dest!r} is absent from data.airports")
        if flight.carrier not in data.carriers:
            raise ValueError(f"candidate carrier {flight.carrier!r} is absent from data.carriers")
        if earliest_utc <= flight.scheduled_departure_utc <= horizon_utc:
            eligible.append(flight)

    eligible.sort(
        key=lambda flight: (
            flight.scheduled_departure_utc,
            flight.scheduled_arrival_utc,
            flight.carrier,
            flight.flight_id,
        )
    )
    return tuple(eligible[:limit])
