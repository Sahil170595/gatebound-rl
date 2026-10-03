"""Outcome-blind local Ollama policy with an explicit deterministic fallback."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests

from flight_rl.baselines import NonstopFirstPolicy
from flight_rl.models import CANDIDATE_FEATURES

_PROMPT_VERSION = "compact-visible-decision-aid-v4"
_SYSTEM_PROMPT = (
    "You select one currently legal scheduled flight for a stranded passenger. "
    "Use only the supplied current state and candidate features. A row whose destination equals "
    "final_destination completes the trip; other rows require an unknown future connection. "
    "If any completing row has arrival_in_min plus mean_arrival_delay_min no greater than "
    "remaining_deadline_min, choose the completing row with the smallest such expected arrival, "
    "using lower p_cancelled plus p_diverted for close ties. Otherwise choose the row most likely "
    "to enable eventual arrival within remaining_horizon_min. Return exactly the requested JSON "
    "object and no explanation."
)
_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"action": {"type": "integer"}},
    "required": ["action"],
    "additionalProperties": False,
}
_GENERATION_OPTIONS: dict[str, int | float] = {
    "temperature": 0,
    "seed": 42,  # Replaced with the configured seed in each request.
    "num_predict": 16,
    "num_ctx": 8192,
}
_SERIALIZED_CANDIDATE_FEATURES = (
    "action",
    "destination",
    "carrier",
    "departure_in_min",
    "arrival_in_min",
    "scheduled_elapsed_min",
    "p_cancelled",
    "p_diverted",
    "mean_arrival_delay_min",
    "support",
)

_DESTINATION = CANDIDATE_FEATURES.index("destination_index")
_CARRIER = CANDIDATE_FEATURES.index("carrier_index")
_DEPARTURE = CANDIDATE_FEATURES.index("departure_in_min")
_ARRIVAL = CANDIDATE_FEATURES.index("arrival_in_min")
_ELAPSED = CANDIDATE_FEATURES.index("scheduled_elapsed_min")
_P_CANCELLED = CANDIDATE_FEATURES.index("p_cancelled")
_P_DIVERTED = CANDIDATE_FEATURES.index("p_diverted")
_MEAN_DELAY = CANDIDATE_FEATURES.index("mean_arrival_delay_min")
_SUPPORT = CANDIDATE_FEATURES.index("support")


def _vocabulary(value: tuple[str, ...], label: str, *, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{label} must be a tuple")  # noqa: TRY004 - public contract
    if not allow_empty and not value:
        raise ValueError(f"{label} must not be empty")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{label} must contain nonempty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} must not contain duplicates")
    return value


def _loopback_base_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("base_url must be a nonempty loopback HTTP URL")
    parsed = urlparse(value.strip())
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("base_url contains an invalid port") from exc
    del port
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.params
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("base_url must identify an uncredentialed loopback Ollama HTTP server")
    return value.strip().rstrip("/")


def _index(value: Any, size: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer index")  # noqa: TRY004 - contract
    result = int(value)
    if not 0 <= result < size:
        raise ValueError(f"{label} is outside its vocabulary")
    return result


def _row_index(value: float, size: int, label: str) -> int:
    result = int(value)
    if float(result) != value or not 0 <= result < size:
        raise ValueError(f"legal candidate {label} is outside its vocabulary")
    return result


def _json_number(value: float) -> int | float:
    integer = int(value)
    return integer if float(integer) == value else float(value)


class OllamaPolicy:
    """Choose from visible candidates using one bounded local generation request.

    Model failures never trigger retries. They take the same action as
    :class:`~flight_rl.baselines.NonstopFirstPolicy` and are retained in diagnostics.
    """

    def __init__(
        self,
        *,
        airports: tuple[str, ...],
        carriers: tuple[str, ...],
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_s: float = 60,
        seed: int = 42,
    ) -> None:
        self._airports = _vocabulary(airports, "airports", allow_empty=False)
        self._carriers = _vocabulary(carriers, "carriers", allow_empty=True)
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty Ollama model tag")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, Real):
            raise ValueError(  # noqa: TRY004 - constructor contract
                "timeout_s must be a positive finite number"
            )
        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_s must be a positive finite number")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed <= 2**31 - 1:
            raise ValueError("seed must be an integer in 0..2147483647")

        self._model = model.strip()
        self._base_url = _loopback_base_url(base_url)
        self._timeout_s = timeout
        self._seed = int(seed)
        self._fallback_policy = NonstopFirstPolicy()
        self._session = requests.Session()
        # A loopback request must not be redirected through an environment proxy.
        self._session.trust_env = False

        self._calls = 0
        self._responses_received = 0
        self._raw_valid = 0
        self._valid_actions = 0
        self._timeouts = 0
        self._request_failures = 0
        self._http_failures = 0
        self._parse_failures = 0
        self._mask_failures = 0
        self._fallbacks = 0
        self._fallback_reasons: Counter[str] = Counter()
        self._call_records: list[dict[str, Any]] = []
        self._last_prompt_sha256: str | None = None
        self._metadata: dict[str, Any] = {
            "digest": None,
            "size_bytes": None,
            "format": None,
            "family": None,
            "parameter_size": None,
            "quantization_level": None,
            "context_length": None,
        }
        self._metadata_lookup: dict[str, Any] = {"status": "not_attempted"}
        self._runtime: dict[str, Any] = {
            "version": None,
            "lookup": {"status": "not_attempted"},
        }
        self._load_model_metadata()
        self._load_runtime_metadata()

    def _load_model_metadata(self) -> None:
        started = time.perf_counter()
        try:
            response = self._session.get(
                f"{self._base_url}/api/tags",
                timeout=min(self._timeout_s, 5.0),
                allow_redirects=False,
            )
            status_code = int(response.status_code)
            if status_code != 200:
                self._metadata_lookup = {
                    "status": "http_failure",
                    "http_status": status_code,
                }
                return
            payload = response.json()
            if not isinstance(payload, Mapping) or not isinstance(payload.get("models"), list):
                raise ValueError(  # noqa: TRY004 - invalid external response
                    "invalid model-list envelope"
                )
            match = next(
                (
                    item
                    for item in payload["models"]
                    if isinstance(item, Mapping)
                    and self._model in {item.get("name"), item.get("model")}
                ),
                None,
            )
            if match is None:
                self._metadata_lookup = {
                    "status": "model_not_listed",
                    "http_status": status_code,
                }
                return
            details = match.get("details")
            details = details if isinstance(details, Mapping) else {}
            self._metadata = {
                "digest": match.get("digest") if isinstance(match.get("digest"), str) else None,
                "size_bytes": (
                    int(match["size"])
                    if isinstance(match.get("size"), Integral)
                    and not isinstance(match.get("size"), bool)
                    else None
                ),
                "format": details.get("format"),
                "family": details.get("family"),
                "parameter_size": details.get("parameter_size"),
                "quantization_level": details.get("quantization_level"),
                "context_length": details.get("context_length"),
            }
            self._metadata_lookup = {
                "status": "resolved",
                "http_status": status_code,
            }
        except requests.Timeout:
            self._metadata_lookup = {"status": "timeout"}
        except requests.RequestException as exc:
            self._metadata_lookup = {
                "status": "request_failure",
                "error_type": type(exc).__name__,
            }
        except (TypeError, ValueError, KeyError) as exc:
            self._metadata_lookup = {
                "status": "invalid_response",
                "error_type": type(exc).__name__,
            }
        finally:
            self._metadata_lookup["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)

    def _load_runtime_metadata(self) -> None:
        started = time.perf_counter()
        try:
            response = self._session.get(
                f"{self._base_url}/api/version",
                timeout=min(self._timeout_s, 5.0),
                allow_redirects=False,
            )
            status_code = int(response.status_code)
            if status_code != 200:
                self._runtime["lookup"] = {
                    "status": "http_failure",
                    "http_status": status_code,
                }
                return
            payload = response.json()
            if not isinstance(payload, Mapping) or not isinstance(payload.get("version"), str):
                raise ValueError(  # noqa: TRY004 - invalid external response
                    "invalid version response envelope"
                )
            self._runtime["version"] = payload["version"]
            self._runtime["lookup"] = {
                "status": "resolved",
                "http_status": status_code,
            }
        except requests.Timeout:
            self._runtime["lookup"] = {"status": "timeout"}
        except requests.RequestException as exc:
            self._runtime["lookup"] = {
                "status": "request_failure",
                "error_type": type(exc).__name__,
            }
        except (TypeError, ValueError) as exc:
            self._runtime["lookup"] = {
                "status": "invalid_response",
                "error_type": type(exc).__name__,
            }
        finally:
            self._runtime["lookup"]["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)

    def _prompt(self, observation: Any) -> tuple[str, np.ndarray, np.ndarray]:
        if not isinstance(observation, Mapping):
            raise ValueError("observation must be a mapping")  # noqa: TRY004 - contract
        required = {
            "current_airport",
            "destination",
            "time",
            "disrupted",
            "candidates",
            "action_mask",
        }
        if not required.issubset(observation):
            missing = sorted(required - set(observation))
            raise ValueError(f"observation is missing fields: {missing}")

        current_index = _index(
            observation["current_airport"], len(self._airports), "current_airport"
        )
        destination_index = _index(observation["destination"], len(self._airports), "destination")
        disrupted = observation["disrupted"]
        if isinstance(disrupted, bool) or not isinstance(disrupted, Integral):
            raise ValueError(  # noqa: TRY004 - observation contract
                "disrupted must be the integer 0 or 1"
            )
        disrupted_index = int(disrupted)
        if disrupted_index not in {0, 1}:
            raise ValueError("disrupted must be the integer 0 or 1")

        raw_time = np.asarray(observation["time"])
        if raw_time.dtype.kind not in "iuIf" or raw_time.dtype.kind == "b":
            raise ValueError("time must be a numeric vector")
        time_values = raw_time.astype(np.float64, copy=False)
        if time_values.shape != (4,) or not bool(np.all(np.isfinite(time_values))):
            raise ValueError("time must contain four finite values")

        raw_mask = np.asarray(observation["action_mask"])
        if raw_mask.ndim != 1 or raw_mask.dtype.kind not in "biuIf":
            raise ValueError("action_mask must be a one-dimensional binary numeric vector")
        try:
            mask_finite = np.isfinite(raw_mask)
        except TypeError as exc:
            raise ValueError("action_mask must contain only 0 and 1") from exc
        if not bool(np.all(mask_finite)) or not bool(np.all((raw_mask == 0) | (raw_mask == 1))):
            raise ValueError("action_mask must contain only 0 and 1")
        mask = raw_mask.astype(np.int8, copy=False)
        legal = np.flatnonzero(mask)

        raw_candidates = np.asarray(observation["candidates"])
        if raw_candidates.dtype.kind not in "iuIf" or raw_candidates.dtype.kind == "b":
            raise ValueError("candidates must be a numeric matrix")
        candidates = raw_candidates.astype(np.float64, copy=False)
        if candidates.ndim != 2 or candidates.shape[1] != len(CANDIDATE_FEATURES):
            raise ValueError(f"candidates must have shape (N, {len(CANDIDATE_FEATURES)})")
        if candidates.shape[0] != mask.shape[0]:
            raise ValueError("candidates and action_mask must have the same row count")
        if legal.size and not bool(np.all(np.isfinite(candidates[legal]))):
            raise ValueError("legal candidate features must be finite")

        serialized_candidates: list[list[Any]] = []
        feasible_final_destination: list[list[int | float]] = []
        for action in legal:
            row = candidates[int(action)]
            candidate_destination = _row_index(
                row[_DESTINATION], len(self._airports), "destination_index"
            )
            candidate_carrier = _row_index(row[_CARRIER], len(self._carriers), "carrier_index")
            if row[_ARRIVAL] <= row[_DEPARTURE] or row[_ELAPSED] <= 0:
                raise ValueError("legal candidate timing features are impossible")
            if not 0 <= row[_P_CANCELLED] <= 1 or not 0 <= row[_P_DIVERTED] <= 1:
                raise ValueError("legal candidate probability features must be in [0, 1]")
            if row[_SUPPORT] < 1 or float(int(row[_SUPPORT])) != row[_SUPPORT]:
                raise ValueError("legal candidate support must be a positive integer")
            serialized_candidates.append(
                [
                    int(action),
                    self._airports[candidate_destination],
                    self._carriers[candidate_carrier],
                    _json_number(row[_DEPARTURE]),
                    _json_number(row[_ARRIVAL]),
                    _json_number(row[_ELAPSED]),
                    round(float(row[_P_CANCELLED]), 6),
                    round(float(row[_P_DIVERTED]), 6),
                    round(float(row[_MEAN_DELAY]), 3),
                    int(row[_SUPPORT]),
                ]
            )
            expected_arrival = row[_ARRIVAL] + row[_MEAN_DELAY]
            if candidate_destination == destination_index and expected_arrival <= time_values[1]:
                feasible_final_destination.append(
                    [
                        int(action),
                        round(float(expected_arrival), 3),
                        round(float(time_values[1] - expected_arrival), 3),
                        round(float(row[_P_CANCELLED] + row[_P_DIVERTED]), 6),
                    ]
                )
        feasible_final_destination.sort(key=lambda row: (row[1], row[3], row[0]))

        prompt_payload = {
            "response_schema": _ACTION_SCHEMA,
            "candidate_feature_order": list(_SERIALIZED_CANDIDATE_FEATURES),
            "legal_candidates": serialized_candidates,
            "visible_decision_aid": {
                "feature_order": [
                    "action",
                    "expected_arrival_in_min",
                    "deadline_slack_min",
                    "p_disrupted",
                ],
                "deadline_feasible_final_destination_actions_sorted_by_expected_arrival": (
                    feasible_final_destination
                ),
            },
            # Keep the decision objective adjacent to the completion instead of burying it
            # before a long candidate list.
            "current_state": {
                "current_airport": self._airports[current_index],
                "final_destination": self._airports[destination_index],
                "elapsed_min": _json_number(time_values[0]),
                "remaining_deadline_min": _json_number(time_values[1]),
                "remaining_horizon_min": _json_number(time_values[2]),
                "attempts_left": _json_number(time_values[3]),
                "previously_disrupted": bool(disrupted_index),
            },
        }
        return json.dumps(prompt_payload, separators=(",", ":")), mask, legal

    def _fallback(self, observation: Any, reason: str) -> int:
        self._fallbacks += 1
        self._fallback_reasons[reason] += 1
        return self._fallback_policy.act(observation)

    def act(self, obs: Any) -> int:
        """Return one legal action after zero or one local generation request."""
        prompt, mask, legal = self._prompt(obs)
        if legal.size == 0:
            return self._fallback(obs, "empty_legal_mask")

        self._calls += 1
        call_index = self._calls
        prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        self._last_prompt_sha256 = prompt_digest
        record: dict[str, Any] = {
            "call": call_index,
            "prompt_sha256": prompt_digest,
            "prompt": prompt,
            "http_status": None,
            "raw_output": None,
            "raw_output_sha256": None,
            "raw_output_truncated": False,
            "raw_valid": False,
            "action_valid": False,
            "selected_action": None,
            "fallback_reason": None,
        }
        options = dict(_GENERATION_OPTIONS)
        options["seed"] = self._seed
        request_payload = {
            "model": self._model,
            "system": _SYSTEM_PROMPT,
            "prompt": prompt,
            "format": _ACTION_SCHEMA,
            "stream": False,
            "keep_alive": "10m",
            "options": options,
        }

        started = time.perf_counter()
        try:
            response = self._session.post(
                f"{self._base_url}/api/generate",
                json=request_payload,
                timeout=self._timeout_s,
                allow_redirects=False,
            )
            record["http_status"] = int(response.status_code)
        except requests.Timeout:
            record["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            record["fallback_reason"] = "timeout"
            self._timeouts += 1
            self._call_records.append(record)
            return self._fallback(obs, "timeout")
        except requests.RequestException as exc:
            record["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            record["fallback_reason"] = "request_failure"
            record["error_type"] = type(exc).__name__
            self._request_failures += 1
            self._call_records.append(record)
            return self._fallback(obs, "request_failure")
        record["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)

        if response.status_code != 200:
            record["fallback_reason"] = "http_failure"
            self._http_failures += 1
            self._call_records.append(record)
            return self._fallback(obs, "http_failure")

        self._responses_received += 1
        try:
            envelope = response.json()
            if (
                not isinstance(envelope, Mapping)
                or envelope.get("done") is not True
                or not isinstance(envelope.get("response"), str)
            ):
                raise ValueError("invalid generation response envelope")
            raw_output = envelope["response"]
            record["raw_output"] = raw_output[:256]
            record["raw_output_truncated"] = len(raw_output) > 256
            record["raw_output_sha256"] = hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
            for source, target in (
                ("total_duration", "ollama_total_duration_ms"),
                ("load_duration", "ollama_load_duration_ms"),
                ("prompt_eval_duration", "ollama_prompt_eval_duration_ms"),
                ("eval_duration", "ollama_eval_duration_ms"),
            ):
                value = envelope.get(source)
                if isinstance(value, Integral) and not isinstance(value, bool) and value >= 0:
                    record[target] = round(int(value) / 1_000_000, 3)
            for field in ("prompt_eval_count", "eval_count"):
                value = envelope.get(field)
                if isinstance(value, Integral) and not isinstance(value, bool) and value >= 0:
                    record[field] = int(value)
            parsed = json.loads(raw_output)
            if (
                not isinstance(parsed, dict)
                or set(parsed) != {"action"}
                or isinstance(parsed["action"], bool)
                or not isinstance(parsed["action"], int)
            ):
                raise ValueError("model output did not match the exact action schema")
        except (json.JSONDecodeError, TypeError, ValueError):
            record["fallback_reason"] = "parse_failure"
            self._parse_failures += 1
            self._call_records.append(record)
            return self._fallback(obs, "parse_failure")

        self._raw_valid += 1
        record["raw_valid"] = True
        action = int(parsed["action"])
        if not 0 <= action < mask.size or mask[action] != 1:
            record["fallback_reason"] = "mask_failure"
            self._mask_failures += 1
            self._call_records.append(record)
            return self._fallback(obs, "mask_failure")

        self._valid_actions += 1
        record["action_valid"] = True
        record["selected_action"] = action
        self._call_records.append(record)
        return action

    def diagnostics(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot of model identity, validity, and fallback use."""
        latencies = [float(record["latency_ms"]) for record in self._call_records]
        if latencies:
            latency_summary: dict[str, float | int | None] = {
                "count": len(latencies),
                "mean": float(np.mean(latencies)),
                "p50": float(np.percentile(latencies, 50)),
                "p95": float(np.percentile(latencies, 95)),
                "min": min(latencies),
                "max": max(latencies),
                "total": sum(latencies),
            }
        else:
            latency_summary = {
                "count": 0,
                "mean": None,
                "p50": None,
                "p95": None,
                "min": None,
                "max": None,
                "total": 0.0,
            }
        generation_options = dict(_GENERATION_OPTIONS)
        generation_options["seed"] = self._seed
        payload = {
            "policy": type(self).__name__,
            "behavior_label": (
                "ollama_with_observation_derived_decision_aid"
                if self._fallbacks == 0
                else "ollama_with_observation_derived_decision_aid_and_nonstop_first_fallback"
            ),
            "fallback_policy": type(self._fallback_policy).__name__,
            "input_scope": "current observation and currently legal candidates only",
            "base_url": self._base_url,
            "local_loopback_only": True,
            "model": self._model,
            "model_digest": self._metadata["digest"],
            "model_size_bytes": self._metadata["size_bytes"],
            "model_details": {
                key: self._metadata[key]
                for key in (
                    "format",
                    "family",
                    "parameter_size",
                    "quantization_level",
                    "context_length",
                )
            },
            "ollama_runtime": self._runtime,
            "metadata_lookup": self._metadata_lookup,
            "generation_config": {
                "endpoint": "/api/generate",
                "stream": False,
                "format": _ACTION_SCHEMA,
                "keep_alive": "10m",
                "timeout_s": self._timeout_s,
                "options": generation_options,
                "maximum_generation_calls_per_action": 1,
                "retries": 0,
            },
            "prompt": {
                "template_version": _PROMPT_VERSION,
                "system": _SYSTEM_PROMPT,
                "candidate_features": list(CANDIDATE_FEATURES),
                "serialized_candidate_feature_order": list(_SERIALIZED_CANDIDATE_FEATURES),
                "visible_derived_fields": {
                    "expected_arrival_in_min": ("arrival_in_min + mean_arrival_delay_min"),
                    "deadline_slack_min": ("remaining_deadline_min - expected_arrival_in_min"),
                    "p_disrupted": "p_cancelled + p_diverted",
                    "decision_aid_scope": (
                        "Only current legal final-destination rows whose visible expected "
                        "arrival meets the current remaining deadline"
                    ),
                },
                "response_schema": _ACTION_SCHEMA,
                "last_prompt_sha256": self._last_prompt_sha256,
            },
            "calls": self._calls,
            "responses_received": self._responses_received,
            "raw_valid": self._raw_valid,
            "raw_valid_rate": self._raw_valid / self._calls if self._calls else None,
            "valid_actions": self._valid_actions,
            "valid_action_rate": self._valid_actions / self._calls if self._calls else None,
            "timeouts": self._timeouts,
            "request_failures": self._request_failures,
            "http_failures": self._http_failures,
            "parse_failures": self._parse_failures,
            "mask_failures": self._mask_failures,
            "fallbacks": self._fallbacks,
            "fallback_counts": dict(sorted(self._fallback_reasons.items())),
            "latencies_ms": latencies,
            "latency_summary_ms": latency_summary,
            "call_records": self._call_records,
        }
        return copy.deepcopy(payload)

    def close(self) -> None:
        """Close the policy's local HTTP connection pool."""
        self._session.close()
