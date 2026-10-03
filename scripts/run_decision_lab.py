#!/usr/bin/env python
"""Serve the local Gatebound Decision Lab over a small loopback HTTP API."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flight_rl.experiments import load_split_data
from flight_rl.fixtures import make_demo_scenario
from flight_rl.models import TripRequest
from flight_rl.verifier import DEADLINE_FIRST_V1_WEIGHTS, PRIMARY_SCORE_PROFILE

BODY_LIMIT_BYTES = 64 * 1024
CASE_FILE_LIMIT_BYTES = 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 10.0
MAX_TRIALS = 64
MAX_CHOICES = 64

POLICIES = (
    {"id": "nonstop_first", "label": "Nonstop first"},
    {"id": "shortest_scheduled", "label": "Shortest scheduled arrival"},
    {"id": "deadline_planner", "label": "Deadline planner"},
)
POLICY_IDS = frozenset(item["id"] for item in POLICIES)

RUBRIC_LABELS = {
    "on_time_arrival": "Met deadline",
    "arrived": "Reached destination",
    "earliness": "Earlier arrival",
}
RUBRIC = tuple(
    {"name": name, "label": RUBRIC_LABELS[name], "weight": float(weight)}
    for name, weight in DEADLINE_FIRST_V1_WEIGHTS.items()
)

CASE_REQUIRED_KEYS = frozenset(
    {
        "id",
        "label",
        "description",
        "origin",
        "destination",
        "ready_utc",
        "deadline_utc",
        "horizon_utc",
        "scenario_seed",
        "policy_id",
    }
)
CASE_OPTIONAL_KEYS = frozenset({"max_attempts", "min_connection_min", "delay_budget_min"})


class RequestError(ValueError):
    """A client request failed boundary validation."""


class UnknownResource(LookupError):
    """A client named an endpoint that is not available."""


@dataclass(frozen=True)
class LabCase:
    case_id: str
    label: str
    description: str
    request: TripRequest
    scenario_seed: int
    policy_id: str

    def catalog_entry(self) -> dict[str, Any]:
        return {
            "id": self.case_id,
            "label": self.label,
            "description": self.description,
            "origin": self.request.origin,
            "destination": self.request.destination,
            "ready_utc": self.request.ready_utc,
            "deadline_utc": self.request.deadline_utc,
            "horizon_utc": self.request.horizon_utc,
            "scenario_seed": self.scenario_seed,
            "policy_id": self.policy_id,
        }


@dataclass
class LabRuntime:
    engine: Any
    cases: dict[str, LabCase]
    catalog: dict[str, Any]
    html: bytes
    compute_lock: threading.Lock


def _reject_json_constant(value: str) -> None:
    raise RequestError(f"JSON number {value!r} is not supported")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RequestError(f"Duplicate JSON field {key!r}")
        result[key] = value
    return result


def _decode_json(raw: bytes) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RequestError("JSON body must use UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise RequestError("Request body is not valid JSON") from exc


def _require_object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RequestError(f"{label} must be a JSON object")
    return value


def _require_exact_keys(
    value: dict[str, Any], required: frozenset[str], *, optional: frozenset[str] = frozenset()
) -> None:
    missing = sorted(required - value.keys())
    extra = sorted(value.keys() - required - optional)
    if missing:
        raise RequestError(f"Missing fields: {', '.join(missing)}")
    if extra:
        raise RequestError(f"Unknown fields: {', '.join(extra)}")


def _require_string(value: Any, name: str, *, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RequestError(f"{name} must be a nonempty string")
    if len(value) > limit:
        raise RequestError(f"{name} is too long")
    return value


def _require_integer(value: Any, name: str) -> int:
    if type(value) is not int:
        raise RequestError(f"{name} must be an integer")
    return value


def _parse_case(value: Any) -> LabCase:
    row = _require_object(value, label="Each case")
    _require_exact_keys(row, CASE_REQUIRED_KEYS, optional=CASE_OPTIONAL_KEYS)

    case_id = _require_string(row["id"], "case id", limit=128)
    policy_id = _require_string(row["policy_id"], "policy_id", limit=64)
    if policy_id not in POLICY_IDS:
        raise RequestError(f"Unsupported case policy {policy_id!r}")

    request_kwargs: dict[str, Any] = {
        "origin": _require_string(row["origin"], "origin", limit=16),
        "destination": _require_string(row["destination"], "destination", limit=16),
        "ready_utc": _require_integer(row["ready_utc"], "ready_utc"),
        "deadline_utc": _require_integer(row["deadline_utc"], "deadline_utc"),
        "horizon_utc": _require_integer(row["horizon_utc"], "horizon_utc"),
    }
    for key in CASE_OPTIONAL_KEYS:
        if key in row:
            request_kwargs[key] = _require_integer(row[key], key)
    try:
        request = TripRequest(**request_kwargs)
    except (TypeError, ValueError) as exc:
        raise RequestError(f"Invalid trip request for case {case_id!r}: {exc}") from exc

    return LabCase(
        case_id=case_id,
        label=_require_string(row["label"], "label", limit=160),
        description=_require_string(row["description"], "description", limit=1000),
        request=request,
        scenario_seed=_require_integer(row["scenario_seed"], "scenario_seed"),
        policy_id=policy_id,
    )


def load_cases(path: Path) -> list[LabCase]:
    size = path.stat().st_size
    if size > CASE_FILE_LIMIT_BYTES:
        raise ValueError("Decision Lab case catalog is larger than 1 MiB")
    try:
        payload = _decode_json(path.read_bytes())
    except RequestError as exc:
        raise ValueError(f"Invalid Decision Lab case catalog: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("Decision Lab case catalog must be a nonempty JSON array")
    try:
        cases = [_parse_case(row) for row in payload]
    except RequestError as exc:
        raise ValueError(f"Invalid Decision Lab case catalog: {exc}") from exc
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Decision Lab case IDs must be unique")
    return cases


def _fixture_case(request: TripRequest) -> LabCase:
    return LabCase(
        case_id="synthetic_demo",
        label="Synthetic direct-versus-connection demo",
        description="A portable five-flight fixture for checking the interface without BTS data.",
        request=request,
        scenario_seed=42,
        policy_id="deadline_planner",
    )


def _catalog(
    cases: list[LabCase], *, mode: str, fit_window: str | None, outcome_window: str | None
) -> dict[str, Any]:
    first = cases[0]
    scope = (
        "Synthetic five-flight fixture; interface demonstration only."
        if mode == "synthetic_fixture"
        else "Configured synthetic passenger requests using 2025 schedules/outcomes and 2020-2024 "
        "fitted history; conditional simulator evidence, not predictions or causal effects."
    )
    return {
        "title": "Gatebound Decision Lab",
        "score_profile": PRIMARY_SCORE_PROFILE,
        "mode": mode,
        "cases": [case.catalog_entry() for case in cases],
        "policies": [dict(item) for item in POLICIES],
        "defaults": {
            "case_id": first.case_id,
            "scenario_seed": first.scenario_seed,
            "policy_id": first.policy_id,
        },
        "provenance": {
            "fit_window": fit_window,
            "outcome_window": outcome_window,
            "scope": scope,
        },
        "rubric": [dict(item) for item in RUBRIC],
    }


def _request_case(runtime: LabRuntime, value: Any) -> LabCase:
    case_id = _require_string(value, "case_id", limit=128)
    try:
        return runtime.cases[case_id]
    except KeyError as exc:
        raise RequestError(f"Unknown case_id {case_id!r}") from exc


def _request_policy(value: Any) -> str:
    policy_id = _require_string(value, "policy_id", limit=64)
    if policy_id not in POLICY_IDS:
        raise RequestError(f"Unknown policy_id {policy_id!r}")
    return policy_id


def _request_choices(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise RequestError("choices must be a JSON array of flight IDs")
    if len(value) > MAX_CHOICES:
        raise RequestError(f"choices may contain at most {MAX_CHOICES} flight IDs")
    return tuple(_require_string(item, "flight ID", limit=512) for item in value)


def replay(runtime: LabRuntime, payload: Any) -> dict[str, Any]:
    body = _require_object(payload, label="Replay request")
    _require_exact_keys(body, frozenset({"case_id", "scenario_seed", "policy_id", "choices"}))
    case = _request_case(runtime, body["case_id"])
    scenario_seed = _require_integer(body["scenario_seed"], "scenario_seed")
    policy_id = _request_policy(body["policy_id"])
    choices = _request_choices(body["choices"])

    with runtime.compute_lock:
        result = runtime.engine.replay(
            case.request,
            scenario_seed,
            policy_id=policy_id,
            choices=choices,
        )
    if not isinstance(result, dict):
        raise TypeError("Lab engine returned an invalid replay response")
    response = dict(result)
    response["case_id"] = case.case_id
    return response


def compare(runtime: LabRuntime, payload: Any) -> dict[str, Any]:
    body = _require_object(payload, label="Comparison request")
    _require_exact_keys(
        body,
        frozenset(
            {
                "case_id",
                "scenario_seed",
                "policy_id",
                "reference_flight_id",
                "alternative_flight_id",
                "trials",
            }
        ),
    )
    case = _request_case(runtime, body["case_id"])
    scenario_seed = _require_integer(body["scenario_seed"], "scenario_seed")
    policy_id = _request_policy(body["policy_id"])
    reference = _require_string(body["reference_flight_id"], "reference_flight_id", limit=512)
    alternative = _require_string(body["alternative_flight_id"], "alternative_flight_id", limit=512)
    if reference == alternative:
        raise RequestError("reference_flight_id and alternative_flight_id must differ")
    trials = _require_integer(body["trials"], "trials")
    if not 1 <= trials <= MAX_TRIALS:
        raise RequestError(f"trials must be between 1 and {MAX_TRIALS}")

    with runtime.compute_lock:
        result = runtime.engine.compare_first_actions(
            case.request,
            scenario_seed,
            policy_id,
            reference,
            alternative,
            trials=trials,
        )
    if not isinstance(result, dict):
        raise TypeError("Lab engine returned an invalid comparison response")
    return result


def make_handler(runtime: LabRuntime) -> type[BaseHTTPRequestHandler]:
    class DecisionLabHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "FlightDecisionLab"
        sys_version = ""

        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(REQUEST_TIMEOUT_SECONDS)

        def log_message(self, _format: str, *args: Any) -> None:
            return

        def _send_bytes(self, status: int, payload: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Connection", "close")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)
            self.close_connection = True

        def _send_json(self, status: int, payload: Any) -> None:
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            self._send_bytes(status, encoded, "application/json; charset=utf-8")

        def _send_error_json(self, status: int, message: str) -> None:
            self._send_json(status, {"error": message})

        def send_error(
            self,
            code: int,
            message: str | None = None,
            explain: str | None = None,
        ) -> None:
            del explain
            safe_message = message if code < 500 and message else HTTPStatus(code).phrase
            self._send_error_json(code, safe_message)

        def _route(self) -> str:
            parsed = urlsplit(self.path)
            if parsed.query or parsed.fragment:
                raise UnknownResource("Unknown endpoint")
            return parsed.path

        def _read_json(self) -> Any:
            transfer_encoding = self.headers.get("Transfer-Encoding")
            if transfer_encoding:
                raise RequestError("Transfer-Encoding is not supported")
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != "application/json":
                raise RequestError("Content-Type must be application/json")
            content_lengths = self.headers.get_all("Content-Length", [])
            if not content_lengths:
                raise RequestError("Content-Length is required")
            if len(content_lengths) != 1:
                raise RequestError("Content-Length must appear exactly once")
            raw_length = content_lengths[0]
            try:
                length = int(raw_length, 10)
            except ValueError as exc:
                raise RequestError("Content-Length must be an integer") from exc
            if length < 1:
                raise RequestError("Request body must not be empty")
            if length > BODY_LIMIT_BYTES:
                raise RequestError(f"Request body exceeds {BODY_LIMIT_BYTES} bytes")
            try:
                raw = self.rfile.read(length)
            except TimeoutError as exc:
                raise RequestError("Timed out while reading the request body") from exc
            if len(raw) != length:
                raise RequestError("Request body ended before Content-Length bytes were received")
            return _decode_json(raw)

        def do_HEAD(self) -> None:
            try:
                route = self._route()
                if route == "/":
                    self._send_bytes(HTTPStatus.OK, runtime.html, "text/html; charset=utf-8")
                else:
                    raise UnknownResource("Unknown endpoint")
            except UnknownResource as exc:
                self._send_error_json(HTTPStatus.NOT_FOUND, str(exc))

        def do_GET(self) -> None:
            try:
                route = self._route()
                if route == "/":
                    self._send_bytes(HTTPStatus.OK, runtime.html, "text/html; charset=utf-8")
                elif route == "/api/catalog":
                    self._send_json(HTTPStatus.OK, runtime.catalog)
                else:
                    raise UnknownResource("Unknown endpoint")
            except UnknownResource as exc:
                self._send_error_json(HTTPStatus.NOT_FOUND, str(exc))
            except Exception:  # noqa: BLE001 - this is the HTTP 500 isolation boundary
                self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, "Unexpected server error")

        def do_POST(self) -> None:
            try:
                route = self._route()
                if route not in {"/api/replay", "/api/compare"}:
                    raise UnknownResource("Unknown endpoint")
                payload = self._read_json()
                response = (
                    replay(runtime, payload)
                    if route == "/api/replay"
                    else compare(runtime, payload)
                )
                self._send_json(HTTPStatus.OK, response)
            except UnknownResource as exc:
                self._send_error_json(HTTPStatus.NOT_FOUND, str(exc))
            except (RequestError, ValueError) as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            except Exception:  # noqa: BLE001 - this is the HTTP 500 isolation boundary
                self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, "Unexpected server error")

        def do_PUT(self) -> None:
            self._send_error_json(HTTPStatus.METHOD_NOT_ALLOWED, "Method not allowed")

        do_DELETE = do_PUT
        do_PATCH = do_PUT
        do_OPTIONS = do_PUT

    return DecisionLabHandler


class DecisionLabHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def build_runtime(args: argparse.Namespace) -> LabRuntime:
    from flight_rl.decision_lab import LabEngine

    html_path = ROOT / "src" / "flight_rl" / "lab.html"
    html = html_path.read_bytes()
    if args.fixture:
        request, data = make_demo_scenario()
        cases = [_fixture_case(request)]
        mode = "synthetic_fixture"
        fit_window = None
        outcome_window = None
    else:
        fit_path = args.fit_data if args.fit_data.is_absolute() else ROOT / args.fit_data
        evaluation_path = (
            args.evaluation_data
            if args.evaluation_data.is_absolute()
            else ROOT / args.evaluation_data
        )
        cases_path = args.cases_file if args.cases_file.is_absolute() else ROOT / args.cases_file
        cases = load_cases(cases_path)
        data, _lineage = load_split_data(fit_path, evaluation_path)
        mode = "held_out_real_data"
        fit_window = "2020-2024"
        outcome_window = "2025"

    engine = LabEngine(data, max_candidates=64)
    cases_by_id = {case.case_id: case for case in cases}
    return LabRuntime(
        engine=engine,
        cases=cases_by_id,
        catalog=_catalog(
            cases,
            mode=mode,
            fit_window=fit_window,
            outcome_window=outcome_window,
        ),
        html=html,
        compute_lock=threading.Lock(),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--fit-data", type=Path, default=Path("data/processed/bts_v1_default_airports")
    )
    parser.add_argument(
        "--evaluation-data",
        type=Path,
        default=Path("data/processed/bts2025_default_airports"),
    )
    parser.add_argument("--cases-file", type=Path, default=Path("examples/decision_lab_cases.json"))
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Use the explicitly synthetic five-flight fixture instead of BTS data.",
    )
    args = parser.parse_args(argv)
    if not args.host.strip():
        parser.error("host must be nonempty")
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    runtime = build_runtime(args)
    handler = make_handler(runtime)
    server = DecisionLabHTTPServer((args.host, args.port), handler)
    host, port = server.server_address[:2]
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    print(f"Gatebound Decision Lab ready: http://{display_host}:{port}/", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
