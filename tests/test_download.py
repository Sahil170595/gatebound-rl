from __future__ import annotations

import importlib.util
import io
import json
import sys
import zipfile
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Self

import pytest


def load_downloader() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "download_bts.py"
    spec = importlib.util.spec_from_file_location("download_bts_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def zip_bytes(row: str = "2020-01-01") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("month.csv", f"FlightDate\n{row}\n")
        archive.writestr("readme.html", "source")
    return buffer.getvalue()


class FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        start: int = 0,
        status_code: int = 200,
        etag: str = '"fixture-v1"',
        interrupt_after_first_chunk: bool = False,
    ) -> None:
        self.status_code = status_code
        self.headers = {"Content-Length": str(len(payload) - start), "ETag": etag}
        if status_code == 206:
            self.headers["Content-Range"] = f"bytes {start}-{len(payload) - 1}/{len(payload)}"
        self._body = payload[start:]
        self._interrupt_after_first_chunk = interrupt_after_first_chunk

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        for offset in range(0, len(self._body), chunk_size):
            yield self._body[offset : offset + chunk_size]
            if self._interrupt_after_first_chunk:
                raise OSError("simulated interrupted transfer")


def test_url_and_month_range_are_exact() -> None:
    downloader = load_downloader()

    assert downloader.build_url(2020, 1).endswith(
        "On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_2020_1.zip"
    )
    assert downloader.month_range(date(2020, 11, 1), date(2021, 2, 1)) == [
        date(2020, 11, 1),
        date(2020, 12, 1),
        date(2021, 1, 1),
        date(2021, 2, 1),
    ]
    with pytest.raises(ValueError):
        downloader.parse_year_month("2020-1")


def test_download_month_resumes_validates_and_completes_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloader = load_downloader()
    payload = zip_bytes()
    final = tmp_path / downloader.FILENAME_TEMPLATE.format(year=2020, month=1)
    partial = final.with_suffix(".zip.part")
    split = len(payload) // 3
    requested_headers: list[dict[str, str]] = []

    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        del url
        headers = dict(kwargs["headers"])
        requested_headers.append(headers)
        if len(requested_headers) == 1:
            return FakeResponse(
                payload,
                interrupt_after_first_chunk=True,
            )
        return FakeResponse(payload, start=split, status_code=206)

    monkeypatch.setattr(downloader.requests, "get", fake_get)
    monkeypatch.setattr(downloader.time, "sleep", lambda _: None)
    record = downloader.download_month(date(2020, 1, 1), tmp_path, retries=2, chunk_bytes=split)

    assert requested_headers == [
        {},
        {"Range": f"bytes={split}-", "If-Range": '"fixture-v1"'},
    ]
    assert record.status == "resumed"
    assert record.bytes == len(payload)
    assert record.csv_member == "month.csv"
    assert final.read_bytes() == payload
    assert not partial.exists()
    assert not partial.with_suffix(partial.suffix + ".json").exists()
    assert downloader.validate_zip(final)[0] == "month.csv"


def test_download_month_restarts_when_server_object_changes_during_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloader = load_downloader()
    old_payload = zip_bytes()
    new_payload = zip_bytes("2020-01-01\n2020-01-02\n2020-01-03")
    assert len(new_payload) != len(old_payload)
    split = len(old_payload) // 3
    requested_headers: list[dict[str, str]] = []

    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        del url
        headers = dict(kwargs["headers"])
        requested_headers.append(headers)
        if len(requested_headers) == 1:
            return FakeResponse(
                old_payload,
                etag='"old-object"',
                interrupt_after_first_chunk=True,
            )
        return FakeResponse(new_payload, etag='"new-object"')

    monkeypatch.setattr(downloader.requests, "get", fake_get)
    monkeypatch.setattr(downloader.time, "sleep", lambda _: None)

    record = downloader.download_month(date(2020, 1, 1), tmp_path, retries=2, chunk_bytes=split)

    assert requested_headers == [
        {},
        {"Range": f"bytes={split}-", "If-Range": '"old-object"'},
    ]
    assert record.status == "downloaded"
    assert record.bytes == len(new_payload)
    assert (tmp_path / record.filename).read_bytes() == new_payload


def test_download_month_restarts_legacy_partial_without_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloader = load_downloader()
    old_payload = zip_bytes()
    new_payload = zip_bytes("2020-01-02")
    final = tmp_path / downloader.FILENAME_TEMPLATE.format(year=2020, month=1)
    partial = final.with_suffix(".zip.part")
    partial.write_bytes(old_payload[: len(old_payload) // 3])
    requested_headers: list[dict[str, str]] = []

    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        del url
        requested_headers.append(dict(kwargs["headers"]))
        return FakeResponse(new_payload, etag='"current-object"')

    monkeypatch.setattr(downloader.requests, "get", fake_get)

    record = downloader.download_month(date(2020, 1, 1), tmp_path, retries=1)

    assert requested_headers == [{}]
    assert record.status == "downloaded"
    assert final.read_bytes() == new_payload


def test_complete_archive_is_reused_and_manifest_merges(tmp_path: Path) -> None:
    downloader = load_downloader()
    payload = zip_bytes()
    first_path = tmp_path / downloader.FILENAME_TEMPLATE.format(year=2020, month=1)
    second_path = tmp_path / downloader.FILENAME_TEMPLATE.format(year=2020, month=2)
    first_path.write_bytes(payload)
    second_path.write_bytes(payload)

    first = downloader.download_month(date(2020, 1, 1), tmp_path)
    second = downloader.download_month(date(2020, 2, 1), tmp_path)
    manifest = downloader.write_manifest(tmp_path, [first])
    downloader.write_manifest(tmp_path, [second])
    payload_json = json.loads(manifest.read_text())

    assert first.status == second.status == "reused"
    assert [entry["month"] for entry in payload_json["files"]] == [
        "2020-01",
        "2020-02",
    ]
    assert not manifest.with_suffix(".json.part").exists()
