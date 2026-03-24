from __future__ import annotations

from types import SimpleNamespace

from app.api.errors import _sanitize_request_path


def test_sanitize_request_path_strips_query_text() -> None:
    request = SimpleNamespace(url=SimpleNamespace(path="/api/reports/consult?api_key=secret"))

    assert _sanitize_request_path(request) == "/api/reports/consult"
