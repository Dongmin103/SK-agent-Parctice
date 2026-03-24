from __future__ import annotations

import app.ui.main as ui_main
from app.ui.client import UiApiError
from app.ui.presenters import FindingViewModel

from app.ui.main import _coerce_ui_api_error, _sanitize_svg_markup


def test_coerce_ui_api_error_accepts_current_error_type() -> None:
    error = UiApiError(
        code="timeout",
        message="FastAPI request timed out.",
        details="timed out",
        status_code=504,
    )

    coerced = _coerce_ui_api_error(error)

    assert coerced is error


def test_coerce_ui_api_error_accepts_stale_error_shape_after_reload() -> None:
    class UiApiErrorReloaded(Exception):
        def __init__(self) -> None:
            self.code = "network_error"
            self.message = "Unable to reach FastAPI."
            self.details = "connection refused"
            self.status_code = None

    UiApiErrorReloaded.__name__ = "UiApiError"

    coerced = _coerce_ui_api_error(UiApiErrorReloaded())

    assert isinstance(coerced, UiApiError)
    assert coerced is not None
    assert coerced.code == "network_error"
    assert coerced.message == "Unable to reach FastAPI."
    assert coerced.details == "connection refused"


def test_coerce_ui_api_error_rejects_unrelated_errors() -> None:
    coerced = _coerce_ui_api_error(RuntimeError("boom"))

    assert coerced is None


def test_sanitize_svg_markup_removes_script_and_event_handlers() -> None:
    sanitized = _sanitize_svg_markup(
        """
        <svg xmlns="http://www.w3.org/2000/svg" onload="alert('boom')">
          <script>alert('boom')</script>
          <rect
            width="10"
            height="10"
            onclick="alert('boom')"
            style="fill:#ffffff;stroke:#000000;stroke-width:1;fill:url(javascript:alert('boom'))"
          />
        </svg>
        """
    )

    assert sanitized is not None
    assert "<script" not in sanitized
    assert "onload" not in sanitized
    assert "onclick" not in sanitized
    assert "javascript:" not in sanitized
    assert "<rect" in sanitized


def test_render_findings_shows_visible_confidence_text(monkeypatch) -> None:
    expander_labels: list[str] = []
    markdown_calls: list[str] = []
    write_calls: list[str] = []

    class DummyExpander:
        def __enter__(self) -> "DummyExpander":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    monkeypatch.setattr(
        ui_main.st,
        "expander",
        lambda label, expanded=True: expander_labels.append(label) or DummyExpander(),
    )
    monkeypatch.setattr(ui_main.st, "markdown", lambda body: markdown_calls.append(body))
    monkeypatch.setattr(ui_main.st, "write", lambda body: write_calls.append(body))
    monkeypatch.setattr(ui_main.st, "info", lambda body: None)

    ui_main._render_findings(
        [
            FindingViewModel(
                agent_id="walter",
                summary="Walter summary",
                confidence=0.81,
            )
        ]
    )

    assert expander_labels == ["walter"]
    assert "**Confidence:** 0.81" in markdown_calls
    assert write_calls == ["Walter summary"]
