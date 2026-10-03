"""Deterministic, policy-independent outcome scenarios for paired evaluation."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from hashlib import blake2b
from math import isfinite
from numbers import Integral, Real
from typing import Any

import numpy as np

from .models import FlightCandidate, OutcomeSummary, SampledOutcome

SEVERITY_ORDER = ("cancelled", "diverted", "missing_timing", "ordinary")
_HASH_PERSON = b"fltrl-scen-v1"
_FLOAT_UNIFORM_RANGE = 1 << 53


def _is_bool(value: object) -> bool:
    return isinstance(value, (bool, np.bool_))


def _is_finite_real(value: object) -> bool:
    return isinstance(value, Real) and not _is_bool(value) and isfinite(float(value))


def _source_method(data: object, name: str):
    method = getattr(data, name, None)
    if not callable(method):
        raise TypeError(f"wrapped source must provide callable {name}()")
    return method


def _hashed_uniform(scenario_seed: int, *parts: str) -> float:
    """Return a stable value in [0, 1) without Python's salted hash."""

    digest = blake2b(digest_size=8, person=_HASH_PERSON)
    components = (str(scenario_seed), *parts)
    for component in components:
        encoded = component.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return _uint64_to_uniform(int.from_bytes(digest.digest(), "big"))


def _uint64_to_uniform(value: int) -> float:
    """Use the high 53 bits so binary64 conversion is exact and remains below one."""

    if not isinstance(value, Integral) or _is_bool(value) or not 0 <= int(value) < 1 << 64:
        raise ValueError("value must be an unsigned 64-bit integer")
    return (int(value) >> 11) / _FLOAT_UNIFORM_RANGE


def _canonical_value(value: object) -> tuple[str, str]:
    """Build a comparable representation for deterministic outcome tie-breaking."""

    if value is None:
        return ("none", "")
    if _is_bool(value):
        return ("bool", "1" if bool(value) else "0")
    if isinstance(value, Integral):
        return ("int", str(int(value)))
    if isinstance(value, Real):
        numeric = float(value)
        if np.isnan(numeric):
            return ("real", "nan")
        if np.isposinf(numeric):
            return ("real", "+inf")
        if np.isneginf(numeric):
            return ("real", "-inf")
        return ("real", numeric.hex())
    if isinstance(value, str):
        return ("str", value)
    return (f"other:{type(value).__module__}.{type(value).__qualname__}", repr(value))


def _descending_number(value: object) -> tuple[int, float]:
    if _is_finite_real(value):
        return (0, -float(value))
    return (1, 0.0)


def _severity_index(outcome: SampledOutcome) -> int:
    # Cancellation is authoritative when a malformed source row sets both flags.
    if outcome.cancelled is True:
        return 0
    if outcome.diverted is True:
        return 1
    if not _is_finite_real(outcome.dep_delay_min) or not _is_finite_real(outcome.arr_delay_min):
        return 2
    return 3


def _outcome_sort_key(outcome: SampledOutcome) -> tuple[Any, ...]:
    severity = _severity_index(outcome)
    if severity == 1:
        within_severity: tuple[Any, ...] = (
            0 if outcome.div_reached_dest is not True else 1,
            _descending_number(outcome.div_arr_delay_min),
            _descending_number(outcome.div_actual_elapsed_min),
            _descending_number(outcome.dep_delay_min),
        )
    elif severity == 3:
        within_severity = (
            _descending_number(outcome.arr_delay_min),
            _descending_number(outcome.dep_delay_min),
        )
    else:
        within_severity = ()

    all_fields = (
        outcome.donor_id,
        outcome.cancelled,
        outcome.diverted,
        outcome.dep_delay_min,
        outcome.arr_delay_min,
        outcome.actual_elapsed_min,
        outcome.div_reached_dest,
        outcome.div_arr_delay_min,
        outcome.div_actual_elapsed_min,
        outcome.div_airport,
        outcome.support,
        outcome.fallback_level,
    )
    return (severity, *within_severity, *(_canonical_value(value) for value in all_fields))


def _ordered_outcomes(outcomes: Iterable[SampledOutcome]) -> tuple[SampledOutcome, ...]:
    pool = tuple(outcomes)
    if not pool:
        raise ValueError("Scenario transition pool must be nonempty")
    if any(not isinstance(outcome, SampledOutcome) for outcome in pool):
        raise TypeError("Scenario transition pools must contain SampledOutcome values")
    donor_ids = [outcome.donor_id for outcome in pool]
    if any(not isinstance(donor_id, str) or not donor_id for donor_id in donor_ids):
        raise ValueError("Scenario donor IDs must be nonempty strings")
    if len(set(donor_ids)) != len(donor_ids):
        raise ValueError("Scenario donor IDs must be unique within each flight pool")
    return tuple(sorted(pool, key=_outcome_sort_key))


def _quantile_index(quantile: float, pool_size: int) -> int:
    """Map a uniform quantile to an empirical donor with equal-width mass bins."""

    if not _is_finite_real(quantile) or not 0.0 <= float(quantile) < 1.0:
        raise ValueError("quantile must be a finite number in [0, 1)")
    if not isinstance(pool_size, Integral) or _is_bool(pool_size) or int(pool_size) < 1:
        raise ValueError("pool_size must be a positive integer")
    return min(int(float(quantile) * int(pool_size)), int(pool_size) - 1)


def _select_quantile(outcomes: Iterable[SampledOutcome], quantile: float) -> SampledOutcome:
    """Select from a canonically ordered empirical pool; useful for invariant tests."""

    ordered = _ordered_outcomes(outcomes)
    return ordered[_quantile_index(quantile, len(ordered))]


class ScenarioPoolCache:
    """Bounded, explicit snapshot cache for ordered simulator transition pools.

    A cache instance belongs to exactly one wrapped source. Call :meth:`clear`
    after deliberately mutating that source; ordinary loaded flight data are
    immutable for the lifetime of an experiment.
    """

    def __init__(self, data: object, maxsize: int = 512) -> None:
        if not isinstance(maxsize, Integral) or _is_bool(maxsize) or int(maxsize) < 1:
            raise ValueError("maxsize must be a positive integer")
        for method_name in ("candidates", "outcome_summary", "historical_outcomes"):
            if not callable(getattr(data, method_name, None)):
                raise TypeError(f"Wrapped data must provide callable {method_name}()")
        transition_provider = getattr(data, "transition_outcomes", None)
        if transition_provider is not None and not callable(transition_provider):
            raise TypeError("transition_outcomes must be callable when provided")

        self.airports = tuple(getattr(data, "airports", ()))
        self.carriers = tuple(getattr(data, "carriers", ()))
        self._data = data
        self._transition_provider = transition_provider
        self._transition_pool_source = (
            "transition_outcomes" if transition_provider is not None else "historical_outcomes"
        )
        self._maxsize = int(maxsize)
        self._cache: OrderedDict[FlightCandidate, tuple[SampledOutcome, ...]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @property
    def metadata(self) -> object:
        return getattr(self._data, "metadata", {})

    @property
    def transition_pool_source(self) -> str:
        return self._transition_pool_source

    def candidates(
        self, origin: str, earliest_utc: int, horizon_utc: int, limit: int = 64
    ) -> tuple[FlightCandidate, ...]:
        return tuple(self._data.candidates(origin, earliest_utc, horizon_utc, limit))

    def outcome_summary(self, flight: FlightCandidate) -> OutcomeSummary:
        return self._data.outcome_summary(flight)

    def source_flight(self, flight_id: str) -> FlightCandidate | None:
        return _source_method(self._data, "source_flight")(flight_id)

    def source_transition_outcome(self, flight_id: str, donor_id: str) -> SampledOutcome | None:
        return _source_method(self._data, "source_transition_outcome")(flight_id, donor_id)

    def historical_outcomes(self, flight: FlightCandidate) -> tuple[SampledOutcome, ...]:
        return tuple(self._data.historical_outcomes(flight))

    def transition_outcomes(self, flight: FlightCandidate) -> tuple[SampledOutcome, ...]:
        if self._transition_provider is not None:
            return tuple(self._transition_provider(flight))
        return tuple(self._data.historical_outcomes(flight))

    def _scenario_ordered_outcomes(self, flight: FlightCandidate) -> tuple[SampledOutcome, ...]:
        if not isinstance(flight, FlightCandidate):
            raise TypeError("flight must be a FlightCandidate")
        cached = self._cache.get(flight)
        if cached is not None:
            self._hits += 1
            self._cache.move_to_end(flight)
            return cached
        self._misses += 1
        ordered = _ordered_outcomes(self.transition_outcomes(flight))
        self._cache[flight] = ordered
        self._cache.move_to_end(flight)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        return ordered

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def cache_info(self) -> dict[str, int]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "maxsize": self._maxsize,
            "currsize": len(self._cache),
        }


class ScenarioData:
    """Wrap flight data with deterministic paired transition outcomes.

    ``dependence`` is the mixing weight of an independence/comonotonic copula.
    A scenario-level deterministic gate selects either flight-keyed independent
    ranks or a common severity rank. It is a synthetic stress parameter, not an
    estimate of empirical cross-flight correlation. Both branches give every
    donor in a flight's empirical pool equal probability across scenario keys.
    """

    def __init__(self, data: object, scenario_seed: int, dependence: float = 0.0) -> None:
        if not isinstance(scenario_seed, Integral) or _is_bool(scenario_seed):
            raise TypeError("scenario_seed must be a non-boolean integer")
        if not _is_finite_real(dependence) or not 0.0 <= float(dependence) <= 1.0:
            raise ValueError("dependence must be a finite number in [0, 1]")

        for method_name in ("candidates", "outcome_summary", "historical_outcomes"):
            if not callable(getattr(data, method_name, None)):
                raise TypeError(f"Wrapped data must provide callable {method_name}()")
        transition_provider = getattr(data, "transition_outcomes", None)
        if transition_provider is not None and not callable(transition_provider):
            raise TypeError("transition_outcomes must be callable when provided")

        self.airports = tuple(getattr(data, "airports", ()))
        self.carriers = tuple(getattr(data, "carriers", ()))
        if not self.airports or any(
            not isinstance(value, str) or not value for value in self.airports
        ):
            raise ValueError("Wrapped data must expose nonempty airport strings")
        if not self.carriers or any(
            not isinstance(value, str) or not value for value in self.carriers
        ):
            raise ValueError("Wrapped data must expose nonempty carrier strings")

        self._data = data
        self._scenario_seed = int(scenario_seed)
        self._dependence = float(dependence)
        self._transition_provider = transition_provider
        self._pool_source = (
            data.transition_pool_source
            if isinstance(data, ScenarioPoolCache)
            else "transition_outcomes"
            if transition_provider is not None
            else "historical_outcomes"
        )
        self._coupling_gate = _hashed_uniform(self._scenario_seed, "coupling-gate")
        self._common_quantile = _hashed_uniform(self._scenario_seed, "common-rank")
        self._use_common_rank = self._coupling_gate < self._dependence
        self._ordered_pool_cache: dict[str, tuple[SampledOutcome, ...]] = {}

    @property
    def scenario_seed(self) -> int:
        return self._scenario_seed

    @property
    def dependence(self) -> float:
        return self._dependence

    def candidates(
        self, origin: str, earliest_utc: int, horizon_utc: int, limit: int = 64
    ) -> tuple[FlightCandidate, ...]:
        return tuple(self._data.candidates(origin, earliest_utc, horizon_utc, limit))

    def outcome_summary(self, flight: FlightCandidate) -> OutcomeSummary:
        return self._data.outcome_summary(flight)

    def source_flight(self, flight_id: str) -> FlightCandidate | None:
        return _source_method(self._data, "source_flight")(flight_id)

    def source_transition_outcome(self, flight_id: str, donor_id: str) -> SampledOutcome | None:
        return _source_method(self._data, "source_transition_outcome")(flight_id, donor_id)

    def historical_outcomes(self, flight: FlightCandidate) -> tuple[SampledOutcome, ...]:
        """Return only policy-facing fit history from the wrapped source."""

        return tuple(self._data.historical_outcomes(flight))

    def transition_outcomes(self, flight: FlightCandidate) -> tuple[SampledOutcome, ...]:
        """Return the simulator-only pool, using held-out outcomes when available."""

        if self._transition_provider is not None:
            return tuple(self._transition_provider(flight))
        return tuple(self._data.historical_outcomes(flight))

    def sample_outcome(self, flight: FlightCandidate, rng: np.random.Generator) -> SampledOutcome:
        if not isinstance(flight, FlightCandidate):
            raise TypeError("flight must be a FlightCandidate")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        if not flight.flight_id:
            raise ValueError("flight_id must be nonempty")

        ordered = self._ordered_pool_cache.get(flight.flight_id)
        if ordered is None:
            ordered = (
                self._data._scenario_ordered_outcomes(flight)
                if isinstance(self._data, ScenarioPoolCache)
                else _ordered_outcomes(self.transition_outcomes(flight))
            )
            self._ordered_pool_cache[flight.flight_id] = ordered
        quantile = (
            self._common_quantile
            if self._use_common_rank
            else _hashed_uniform(self._scenario_seed, "flight-rank", flight.flight_id)
        )
        return ordered[_quantile_index(quantile, len(ordered))]

    def diagnostics(self) -> dict[str, object]:
        """Return JSON-safe configuration diagnostics without exposing fit history."""

        result: dict[str, object] = {
            "scenario_seed": self._scenario_seed,
            "dependence": self._dependence,
            "dependence_interpretation": "synthetic mixture weight; not estimated correlation",
            "coupling": "mixture of flight-keyed independent ranks and one common severity rank",
            "coupling_mode": (
                "common_severity_rank" if self._use_common_rank else "flight_keyed_independent"
            ),
            "coupling_gate": self._coupling_gate,
            "common_quantile": self._common_quantile,
            "severity_order": list(SEVERITY_ORDER),
            "transition_pool_source": self._pool_source,
            "policy_history_source": "wrapped historical_outcomes",
            "environment_rng_selects_donor": False,
            "marginal_mapping": "equal-width empirical quantile bins",
        }
        if isinstance(self._data, ScenarioPoolCache):
            result["ordered_pool_cache"] = self._data.cache_info()
        return result
