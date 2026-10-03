#!/usr/bin/env python
"""Download monthly BTS Marketing Carrier On-Time Performance archives.

Files are streamed to .part paths, validated as ZIP archives (including CRC
checks), and atomically renamed. Interrupted partial files are resumed when
the BTS server honours HTTP Range requests. A deterministic JSON manifest
records the URL, month, byte count, SHA-256, and CSV member.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import zipfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import requests

BASE_URL = "https://transtats.bts.gov/PREZIP"
FILENAME_TEMPLATE = (
    "On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_{year}_{month}.zip"
)
MANIFEST_FILENAME = "download_manifest.json"
DEFAULT_WORKERS = 3
DEFAULT_RETRIES = 3
DEFAULT_CHUNK_BYTES = 1024 * 1024
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
_PARTIAL_STATE_VERSION = 1


@dataclass(frozen=True)
class DownloadRecord:
    """Manifest entry for one complete monthly archive."""

    month: str
    url: str
    filename: str
    bytes: int
    sha256: str
    csv_member: str
    zip_members: tuple[str, ...]
    status: str
    duration_seconds: float


@dataclass(frozen=True)
class _PartialState:
    validator_header: str
    validator: str
    total_bytes: int | None


def month_range(start: date, end: date) -> list[date]:
    """Return first-of-month dates from start through end inclusive."""

    current = start.replace(day=1)
    last = end.replace(day=1)
    if last < current:
        raise ValueError("end month must not precede start month")
    result: list[date] = []
    while current <= last:
        result.append(current)
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return result


def build_url(year: int, month: int) -> str:
    """Return the documented BTS prezip URL for a month."""

    if year < 2018 or not 1 <= month <= 12:
        raise ValueError("BTS marketing-carrier prezip requires year >= 2018 and month 1..12")
    filename = FILENAME_TEMPLATE.format(year=year, month=month)
    return f"{BASE_URL}/{filename}"


def parse_year_month(value: str) -> date:
    """Parse exactly YYYY-MM into a first-of-month date."""

    try:
        parsed = date.fromisoformat(f"{value}-01")
    except ValueError as exc:
        raise ValueError(f"invalid year-month {value!r}; expected YYYY-MM") from exc
    if parsed.strftime("%Y-%m") != value:
        raise ValueError(f"invalid year-month {value!r}; expected YYYY-MM")
    return parsed.replace(day=1)


def sha256_file(path: Path, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> str:
    """Hash a local file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def validate_zip(path: Path) -> tuple[str, tuple[str, ...]]:
    """Validate ZIP structure and CRCs and return its single CSV member."""

    if not path.is_file() or not zipfile.is_zipfile(path):
        raise ValueError(f"not a valid ZIP archive: {path}")
    with zipfile.ZipFile(path) as archive:
        members = tuple(info.filename for info in archive.infolist() if not info.is_dir())
        csv_members = tuple(name for name in members if name.lower().endswith(".csv"))
        if len(csv_members) != 1:
            raise ValueError(f"expected exactly one CSV member in {path}, found {csv_members}")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"CRC failure in {path}: {bad_member}")
    return csv_members[0], members


def _expected_total(response: requests.Response, offset: int) -> int | None:
    """Resolve the complete object length from a GET response."""

    if response.status_code == 206:
        content_range = response.headers.get("Content-Range", "")
        match = _CONTENT_RANGE.fullmatch(content_range)
        if match is None:
            raise RuntimeError(f"invalid Content-Range for resume: {content_range!r}")
        start, end, total = (int(value) for value in match.groups())
        content_length = response.headers.get("Content-Length")
        invalid_length = content_length is not None and int(content_length) != end - start + 1
        if start != offset or end < start or end >= total or invalid_length:
            raise RuntimeError(f"invalid Content-Range for resume: {content_range!r}")
        return total
    content_length = response.headers.get("Content-Length")
    if content_length is None:
        return None
    total = int(content_length)
    if total < 0:
        raise RuntimeError(f"invalid Content-Length: {content_length!r}")
    return total


def _response_validator(response: requests.Response) -> tuple[str, str] | None:
    etag = response.headers.get("ETag", "").strip()
    if etag and not etag.startswith("W/"):
        return "ETag", etag
    last_modified = response.headers.get("Last-Modified", "").strip()
    if last_modified:
        return "Last-Modified", last_modified
    return None


def _load_partial_state(path: Path, url: str, offset: int) -> _PartialState | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    validator_header = payload.get("validator_header")
    validator = payload.get("validator")
    total_bytes = payload.get("total_bytes")
    if (
        payload.get("schema_version") != _PARTIAL_STATE_VERSION
        or payload.get("url") != url
        or validator_header not in {"ETag", "Last-Modified"}
        or not isinstance(validator, str)
        or not validator
        or (
            total_bytes is not None
            and (
                isinstance(total_bytes, bool)
                or not isinstance(total_bytes, int)
                or total_bytes < offset
            )
        )
    ):
        return None
    return _PartialState(
        validator_header=validator_header,
        validator=validator,
        total_bytes=total_bytes,
    )


def _write_partial_state(
    path: Path,
    url: str,
    validator: tuple[str, str],
    total_bytes: int | None,
) -> None:
    payload = {
        "schema_version": _PARTIAL_STATE_VERSION,
        "url": url,
        "validator_header": validator[0],
        "validator": validator[1],
        "total_bytes": total_bytes,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _discard_partial(part_path: Path, state_path: Path) -> None:
    part_path.unlink(missing_ok=True)
    state_path.unlink(missing_ok=True)


def _record_for_complete(
    month: date,
    url: str,
    path: Path,
    *,
    status: str,
    duration_seconds: float,
) -> DownloadRecord:
    csv_member, members = validate_zip(path)
    return DownloadRecord(
        month=f"{month:%Y-%m}",
        url=url,
        filename=path.name,
        bytes=path.stat().st_size,
        sha256=sha256_file(path),
        csv_member=csv_member,
        zip_members=members,
        status=status,
        duration_seconds=round(duration_seconds, 3),
    )


def download_month(
    month: date,
    out_dir: Path,
    *,
    retries: int = DEFAULT_RETRIES,
    timeout: tuple[float, float] = (20.0, 120.0),
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> DownloadRecord:
    """Download, resume, validate, and atomically complete one month."""

    if retries < 1 or chunk_bytes < 1:
        raise ValueError("retries and chunk_bytes must be positive")
    month = month.replace(day=1)
    url = build_url(month.year, month.month)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / url.rsplit("/", 1)[-1]
    part_path = final_path.with_suffix(final_path.suffix + ".part")
    state_path = part_path.with_suffix(part_path.suffix + ".json")

    if final_path.exists():
        if part_path.exists():
            raise ValueError(f"both complete and partial archives exist for {month:%Y-%m}")
        state_path.unlink(missing_ok=True)
        return _record_for_complete(month, url, final_path, status="reused", duration_seconds=0.0)

    started = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        resumed = False
        try:
            restarted_stale_partial = False
            while True:
                offset = part_path.stat().st_size if part_path.exists() else 0
                state = _load_partial_state(state_path, url, offset) if offset else None
                if offset and state is None:
                    _discard_partial(part_path, state_path)
                    offset = 0
                headers: dict[str, str] = {}
                if offset:
                    assert state is not None
                    headers = {
                        "Range": f"bytes={offset}-",
                        "If-Range": state.validator,
                    }
                with requests.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=timeout,
                    allow_redirects=True,
                ) as response:
                    if response.status_code == 416 and offset:
                        _discard_partial(part_path, state_path)
                        if restarted_stale_partial:
                            raise RuntimeError("server repeatedly rejected a restarted download")
                        restarted_stale_partial = True
                        continue
                    response.raise_for_status()
                    if response.status_code not in {200, 206}:
                        raise RuntimeError(
                            f"unexpected HTTP status {response.status_code} for {url}"
                        )
                    resumed = offset > 0 and response.status_code == 206
                    if resumed:
                        assert state is not None
                        expected_total = _expected_total(response, offset)
                        returned_validator = response.headers.get(
                            state.validator_header, ""
                        ).strip()
                        identity_changed = bool(
                            returned_validator and returned_validator != state.validator
                        )
                        length_changed = (
                            state.total_bytes is not None and expected_total != state.total_bytes
                        )
                        if identity_changed or length_changed:
                            _discard_partial(part_path, state_path)
                            if restarted_stale_partial:
                                raise RuntimeError("server repeatedly changed a restarted download")
                            restarted_stale_partial = True
                            continue
                        if state.total_bytes is None:
                            _write_partial_state(
                                state_path,
                                url,
                                (state.validator_header, state.validator),
                                expected_total,
                            )
                    else:
                        offset = 0
                        expected_total = _expected_total(response, offset)
                        validator = _response_validator(response)
                        if validator is None:
                            state_path.unlink(missing_ok=True)
                        else:
                            _write_partial_state(state_path, url, validator, expected_total)
                    mode = "ab" if resumed else "wb"
                    with part_path.open(mode) as output:
                        for chunk in response.iter_content(chunk_size=chunk_bytes):
                            if chunk:
                                output.write(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                break

            actual_total = part_path.stat().st_size
            if expected_total is not None and actual_total != expected_total:
                if actual_total > expected_total:
                    _discard_partial(part_path, state_path)
                raise OSError(
                    f"incomplete download for {month:%Y-%m}: "
                    f"expected {expected_total} bytes, found {actual_total}"
                )
            try:
                validate_zip(part_path)
            except ValueError:
                _discard_partial(part_path, state_path)
                raise
            os.replace(part_path, final_path)
            state_path.unlink(missing_ok=True)
            return _record_for_complete(
                month,
                url,
                final_path,
                status="resumed" if resumed else "downloaded",
                duration_seconds=time.perf_counter() - started,
            )
        except (OSError, requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))

    raise RuntimeError(
        f"failed to download {month:%Y-%m} after {retries} attempts; "
        f"partial retained at {part_path}: {last_error}"
    ) from last_error


def _load_manifest_entries(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read existing download manifest {path}: {exc}") from exc
    if payload.get("schema_version") != 1 or not isinstance(payload.get("files"), list):
        raise ValueError(f"unsupported download manifest format: {path}")
    return {str(entry["month"]): entry for entry in payload["files"]}


def write_manifest(out_dir: Path, records: Iterable[DownloadRecord]) -> Path:
    """Merge records into the deterministic download manifest atomically."""

    manifest_path = out_dir / MANIFEST_FILENAME
    entries = _load_manifest_entries(manifest_path)
    for record in records:
        entries[record.month] = asdict(record)
    payload = {
        "schema_version": 1,
        "dataset": "BTS Marketing Carrier On-Time Performance",
        "base_url": BASE_URL,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "files": [entries[key] for key in sorted(entries)],
    }
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, manifest_path)
    return manifest_path


def download_range(
    months: Iterable[date],
    out_dir: Path,
    *,
    workers: int = DEFAULT_WORKERS,
    retries: int = DEFAULT_RETRIES,
    timeout: tuple[float, float] = (20.0, 120.0),
) -> tuple[DownloadRecord, ...]:
    """Download a bounded set of months and update the manifest."""

    requested = sorted({month.replace(day=1) for month in months})
    if not requested:
        raise ValueError("at least one month is required")
    if workers < 1:
        raise ValueError("workers must be positive")
    records: list[DownloadRecord] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(requested))) as executor:
        future_to_month = {
            executor.submit(
                download_month,
                month,
                out_dir,
                retries=retries,
                timeout=timeout,
            ): month
            for month in requested
        }
        for future in as_completed(future_to_month):
            month = future_to_month[future]
            try:
                record = future.result()
            except Exception as exc:  # noqa: BLE001 - collect every worker failure
                failures.append(f"{month:%Y-%m}: {exc}")
                print(f"FAILED {month:%Y-%m}: {exc}", flush=True)
            else:
                records.append(record)
                print(
                    f"{record.status.upper():10s} {record.month} "
                    f"{record.bytes} bytes sha256={record.sha256}",
                    flush=True,
                )
    records.sort(key=lambda record: record.month)
    if records:
        manifest_path = write_manifest(out_dir, records)
        print(f"manifest: {manifest_path.resolve()}", flush=True)
    if failures:
        raise RuntimeError("one or more monthly downloads failed:\n" + "\n".join(failures))
    return tuple(records)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="First month, YYYY-MM.")
    parser.add_argument("--end", required=True, help="Last month, YYYY-MM, inclusive.")
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--connect-timeout", type=float, default=20.0)
    parser.add_argument("--read-timeout", type=float, default=120.0)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print URLs without downloading files."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="HEAD-check the first URL and exit without downloading.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        start = parse_year_month(args.start)
        end = parse_year_month(args.end)
        months = month_range(start, end)
    except ValueError as exc:
        parser.error(str(exc))
    urls = [build_url(month.year, month.month) for month in months]
    if args.dry_run:
        for month, url in zip(months, urls, strict=True):
            print(f"{month:%Y-%m} {url}")
        return
    if args.verify:
        response = requests.head(urls[0], timeout=args.connect_timeout, allow_redirects=True)
        print(f"status: {response.status_code}")
        print(f"content-length: {response.headers.get('Content-Length')}")
        response.raise_for_status()
        return
    download_range(
        months,
        args.out,
        workers=args.workers,
        retries=args.retries,
        timeout=(args.connect_timeout, args.read_timeout),
    )


if __name__ == "__main__":
    main()
