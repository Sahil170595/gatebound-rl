"""Focused contract tests for the outcome-blind local Ollama policy."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
import requests

from flight_rl.llm_policy import OllamaPolicy

MODEL = "qwen2.5:1.5b-instruct-q4_K_M"
DIGEST = "65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b"


class FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        if isinstance(self._payload, BaseException):
            raise self._payload
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


class FakeSession:
    def __init__(
        self,
        outputs: list[Any] | None = None,
        metadata_response: FakeResponse | None = None,
    ) -> None:
        self.outputs = list(outputs or [])
        self.metadata_response = metadata_response
        self.gets: list[dict[str, Any]] = []
        self.posts: list[dict[str, Any]] = []
        self.trust_env = True
        self.closed = False

    def get(self, url: str, *, timeout: float, allow_redirects: bool) -> FakeResponse:
        self.gets.append({"url": url, "timeout": timeout, "allow_redirects": allow_redirects})
        if self.metadata_response is not None and url.endswith("/api/tags"):
            return self.metadata_response
        if url.endswith("/api/version"):
            return FakeResponse({"version": "0.32.1"})
        return FakeResponse(
            {
                "models": [
                    {
                        "name": MODEL,
                        "model": MODEL,
                        "size": 986_061_892,
                        "digest": DIGEST,
                        "details": {
                            "format": "gguf",
                            "family": "qwen2",
                            "parameter_size": "1.5B",
                            "quantization_level": "Q4_K_M",
                            "context_length": 32_768,
                        },
                    }
                ]
            }
        )

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, Any],
        timeout: float,
        allow_redirects: bool,
    ) -> FakeResponse:
        self.posts.append(
            {
                "url": url,
                "json": json,
                "timeout": timeout,
                "allow_redirects": allow_redirects,
            }
        )
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return output

    def close(self) -> None:
        self.closed = True


def observation() -> dict[str, Any]:
    candidates = np.zeros((4, 9), dtype=np.float32)
    candidates[0] = [1, 0, 60, 180, 120, 0.1, 0.02, 8, 100]
    # Masked rows may contain arbitrary finite padding; they must never reach the prompt.
    candidates[1] = [999_999, 999_999, 12_345, 23_456, 11_111, -7, -8, 9_999, -3]
    candidates[2] = [2, 1, 90, 390, 300, 0.03, 0.01, 4, 250]
    candidates[3] = [-999_999, -999_999, -12_345, -23_456, -11_111, 7, 8, -9_999, -3]
    return {
        "current_airport": 0,
        "destination": 2,
        "time": np.asarray([0, 480, 720, 3], dtype=np.float32),
        "disrupted": 0,
        "candidates": candidates,
        "action_mask": np.asarray([1, 0, 1, 0], dtype=np.int8),
    }


def build_policy(
    monkeypatch: pytest.MonkeyPatch, outputs: list[Any]
) -> tuple[OllamaPolicy, FakeSession]:
    session = FakeSession(outputs)
    monkeypatch.setattr(requests, "Session", lambda: session)
    policy = OllamaPolicy(
        airports=("SFO", "DEN", "JFK"),
        carriers=("UA", "DL"),
        model=MODEL,
        timeout_s=7.5,
        seed=123,
    )
    return policy, session


def generation(raw: str, *, done: bool = True) -> FakeResponse:
    return FakeResponse(
        {
            "model": MODEL,
            "done": done,
            "response": raw,
            "total_duration": 2_500_000,
            "load_duration": 1_000_000,
            "prompt_eval_duration": 900_000,
            "eval_duration": 500_000,
            "prompt_eval_count": 210,
            "eval_count": 5,
        }
    )


def test_valid_action_uses_one_bounded_call_and_only_visible_legal_rows(monkeypatch):
    policy, session = build_policy(monkeypatch, [generation('{"action":2}')])

    assert policy.act(observation()) == 2
    assert len(session.gets) == 2
    assert len(session.posts) == 1
    request = session.posts[0]
    assert request["url"] == "http://127.0.0.1:11434/api/generate"
    assert request["timeout"] == 7.5
    assert request["allow_redirects"] is False
    assert all(request["allow_redirects"] is False for request in session.gets)
    body = request["json"]
    assert body["stream"] is False
    assert body["options"] == {
        "temperature": 0,
        "seed": 123,
        "num_predict": 16,
        "num_ctx": 8192,
    }
    assert body["format"]["additionalProperties"] is False
    serialized = json.loads(body["prompt"])
    assert serialized["current_state"] == {
        "attempts_left": 3,
        "current_airport": "SFO",
        "elapsed_min": 0,
        "final_destination": "JFK",
        "previously_disrupted": False,
        "remaining_deadline_min": 480,
        "remaining_horizon_min": 720,
    }
    assert serialized["candidate_feature_order"] == [
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
    ]
    assert [candidate[0] for candidate in serialized["legal_candidates"]] == [0, 2]
    assert len(serialized["legal_candidates"]) == 2
    assert {candidate[1] for candidate in serialized["legal_candidates"]} == {"DEN", "JFK"}
    assert serialized["visible_decision_aid"] == {
        "feature_order": [
            "action",
            "expected_arrival_in_min",
            "deadline_slack_min",
            "p_disrupted",
        ],
        "deadline_feasible_final_destination_actions_sorted_by_expected_arrival": [
            [2, 394.0, 86.0, 0.04]
        ],
    }
    assert "donor" not in body["prompt"].lower()
    assert "outcome" not in body["prompt"].lower()

    diagnostics = policy.diagnostics()
    assert diagnostics["model_digest"] == DIGEST
    assert diagnostics["model_details"]["quantization_level"] == "Q4_K_M"
    assert diagnostics["calls"] == diagnostics["raw_valid"] == diagnostics["valid_actions"] == 1
    assert diagnostics["fallbacks"] == 0
    assert diagnostics["behavior_label"] == "ollama_with_observation_derived_decision_aid"
    assert diagnostics["ollama_runtime"]["version"] == "0.32.1"
    assert diagnostics["call_records"][0]["ollama_total_duration_ms"] == 2.5
    json.dumps(diagnostics, allow_nan=False)


@pytest.mark.parametrize(
    ("response", "counter"),
    [
        (generation('{"action":"2"}'), "parse_failures"),
        (generation('{"action":2,"reason":"extra"}'), "parse_failures"),
        (generation('{"action":1}'), "mask_failures"),
        (generation('{"action":9}'), "mask_failures"),
        (generation('{"action":2}', done=False), "parse_failures"),
    ],
)
def test_invalid_model_output_is_counted_and_uses_nonstop_fallback(monkeypatch, response, counter):
    policy, session = build_policy(monkeypatch, [response])

    # Candidate 2 is the final-destination nonstop selected by the explicit fallback.
    assert policy.act(observation()) == 2
    assert len(session.posts) == 1
    diagnostics = policy.diagnostics()
    assert diagnostics["calls"] == 1
    assert diagnostics[counter] == 1
    assert diagnostics["fallbacks"] == 1
    assert diagnostics["valid_actions"] == 0
    assert diagnostics["behavior_label"].endswith("_and_nonstop_first_fallback")


def test_timeout_makes_no_retry_and_records_fallback(monkeypatch):
    policy, session = build_policy(monkeypatch, [requests.Timeout("bounded timeout")])

    assert policy.act(observation()) == 2
    assert len(session.posts) == 1
    diagnostics = policy.diagnostics()
    assert diagnostics["calls"] == 1
    assert diagnostics["timeouts"] == 1
    assert diagnostics["fallback_counts"] == {"timeout": 1}
    assert diagnostics["raw_valid"] == 0


def test_redirect_response_is_not_followed_and_uses_fallback(monkeypatch):
    policy, session = build_policy(monkeypatch, [FakeResponse({}, status_code=302)])

    assert policy.act(observation()) == 2
    assert len(session.posts) == 1
    assert session.posts[0]["allow_redirects"] is False
    diagnostics = policy.diagnostics()
    assert diagnostics["http_failures"] == 1
    assert diagnostics["fallback_counts"] == {"http_failure": 1}


def test_metadata_redirect_is_not_followed_or_accepted(monkeypatch):
    session = FakeSession(metadata_response=FakeResponse({}, status_code=307))
    monkeypatch.setattr(requests, "Session", lambda: session)

    policy = OllamaPolicy(
        airports=("SFO", "JFK"),
        carriers=("UA",),
        model=MODEL,
    )

    assert len(session.gets) == 2
    assert all(request["allow_redirects"] is False for request in session.gets)
    metadata = policy.diagnostics()["metadata_lookup"]
    assert metadata["status"] == "http_failure"
    assert metadata["http_status"] == 307
    assert metadata["latency_ms"] >= 0


def test_empty_mask_uses_counted_fallback_without_a_model_call(monkeypatch):
    policy, session = build_policy(monkeypatch, [])
    obs = observation()
    obs["action_mask"][:] = 0

    assert policy.act(obs) == 0
    assert session.posts == []
    diagnostics = policy.diagnostics()
    assert diagnostics["calls"] == 0
    assert diagnostics["fallbacks"] == 1
    assert diagnostics["fallback_counts"] == {"empty_legal_mask": 1}
    assert diagnostics["raw_valid_rate"] is None


def test_nonfinite_legal_features_fail_before_any_model_call(monkeypatch):
    policy, session = build_policy(monkeypatch, [])
    obs = observation()
    obs["candidates"][0, 5] = np.nan

    with pytest.raises(ValueError, match="legal candidate features must be finite"):
        policy.act(obs)
    assert session.posts == []


@pytest.mark.parametrize(
    "base_url",
    [
        "https://127.0.0.1:11434",
        "http://api.ollama.com",
        "http://user:password@127.0.0.1:11434",
        "http://127.0.0.1:11434/api",
    ],
)
def test_constructor_rejects_nonlocal_or_credentialed_endpoint(monkeypatch, base_url):
    session = FakeSession()
    monkeypatch.setattr(requests, "Session", lambda: session)
    with pytest.raises(ValueError, match="loopback"):
        OllamaPolicy(
            airports=("SFO", "JFK"),
            carriers=("UA",),
            model=MODEL,
            base_url=base_url,
        )
    assert session.gets == []


def test_diagnostics_is_a_snapshot_and_close_releases_session(monkeypatch):
    policy, session = build_policy(monkeypatch, [generation('{"action":2}')])
    policy.act(observation())
    first = policy.diagnostics()
    first["call_records"][0]["selected_action"] = 999

    assert policy.diagnostics()["call_records"][0]["selected_action"] == 2
    policy.close()
    assert session.closed is True
