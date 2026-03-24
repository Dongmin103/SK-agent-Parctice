from __future__ import annotations

import json

import httpx
import pytest

from app.ui.client import UiApiClient, UiApiError


def test_ui_api_client_posts_consult_payload_and_returns_response() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            status_code=200,
            json={
                "selected_agents": ["house"],
                "routing_reason": "Safety question",
                "predictions": {
                    "source": "txgemma_stub",
                    "target": "KRAS G12C",
                    "compound_name": "ABC-101",
                    "canonical_smiles": "CCO",
                    "signals": [],
                    "missing_signals": [],
                    "generated_at": "2026-03-24T12:00:00+00:00",
                },
                "agent_findings": [],
                "consulting_answer": "House answer",
                "citations": ["https://example.org/house"],
                "review_required": True,
            },
        )

    transport = httpx.MockTransport(handler)
    client = UiApiClient(
        base_url="http://127.0.0.1:8000",
        http_client=httpx.Client(transport=transport),
    )

    payload = client.submit_consult(
        smiles="CCO",
        target="KRAS G12C",
        question="이 화합물의 hERG 위험은?",
        compound_name="ABC-101",
    )

    assert captured["url"] == "http://127.0.0.1:8000/api/reports/consult"
    assert captured["payload"] == {
        "smiles": "CCO",
        "target": "KRAS G12C",
        "question": "이 화합물의 hERG 위험은?",
        "compound_name": "ABC-101",
    }
    assert payload["selected_agents"] == ["house"]


def test_ui_api_client_raises_typed_error_for_structured_api_failures() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            status_code=400,
            json={
                "error": {
                    "code": "invalid_smiles",
                    "message": "Invalid SMILES: not-a-smiles",
                }
            },
        )

    transport = httpx.MockTransport(handler)
    client = UiApiClient(
        base_url="http://127.0.0.1:8000",
        http_client=httpx.Client(transport=transport),
    )

    with pytest.raises(UiApiError) as exc_info:
        client.submit_executive(
            smiles="not-a-smiles",
            target="KRAS G12C",
            compound_name="ABC-101",
        )

    assert exc_info.value.code == "invalid_smiles"
    assert str(exc_info.value) == "Invalid SMILES: not-a-smiles"


def test_ui_api_client_maps_timeout_to_typed_timeout_error() -> None:
    class TimeoutingClient:
        def post(self, url: str, *, json: dict[str, object], timeout: float) -> httpx.Response:
            del json
            raise httpx.ReadTimeout(
                "timed out",
                request=httpx.Request("POST", url),
            )

    client = UiApiClient(
        base_url="http://127.0.0.1:8000",
        http_client=TimeoutingClient(),
        timeout_seconds=45.0,
    )

    with pytest.raises(UiApiError) as exc_info:
        client.submit_executive(
            smiles="CCO",
            target="KRAS G12C",
            compound_name="ABC-101",
        )

    assert exc_info.value.code == "timeout"
    assert "45" in str(exc_info.value)
    assert "http://127.0.0.1:8000" in str(exc_info.value)


def test_ui_api_client_streams_consult_trace_and_result_chunks() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://127.0.0.1:8000/api/reports/consult/stream"
        return httpx.Response(
            status_code=200,
            content=(
                json.dumps(
                    {
                        "type": "trace",
                        "trace": {
                            "stage": "routing",
                            "message": "Selected agents: house, harvey",
                            "level": "info",
                            "details": {},
                        },
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "type": "result",
                        "result": {
                            "selected_agents": ["house", "harvey"],
                            "routing_reason": "Safety and clinical evidence are both needed.",
                        },
                    }
                )
                + "\n"
            ).encode("utf-8"),
        )

    transport = httpx.MockTransport(handler)
    client = UiApiClient(
        base_url="http://127.0.0.1:8000",
        http_client=httpx.Client(transport=transport),
    )

    chunks = list(
        client.stream_consult(
            smiles="CCO",
            target="KRAS G12C",
            question="이 화합물의 hERG 위험은?",
            compound_name="ABC-101",
        )
    )

    assert [chunk["type"] for chunk in chunks] == ["trace", "result"]
    assert chunks[0]["trace"]["message"] == "Selected agents: house, harvey"
    assert chunks[1]["result"]["selected_agents"] == ["house", "harvey"]


def test_ui_api_client_raises_typed_error_for_stream_error_chunks() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            status_code=200,
            content=(
                json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "code": "upstream_unavailable",
                            "message": "Consult workflow dependency was unavailable.",
                            "details": {"dependency": "bedrock"},
                        },
                    }
                )
                + "\n"
            ).encode("utf-8"),
        )

    transport = httpx.MockTransport(handler)
    client = UiApiClient(
        base_url="http://127.0.0.1:8000",
        http_client=httpx.Client(transport=transport),
    )

    with pytest.raises(UiApiError) as exc_info:
        list(
            client.stream_consult(
                smiles="CCO",
                target="KRAS G12C",
                question="이 화합물의 hERG 위험은?",
                compound_name="ABC-101",
            )
        )

    assert exc_info.value.code == "upstream_unavailable"
    assert exc_info.value.details == {"dependency": "bedrock"}


def test_ui_api_client_reads_streaming_error_response_before_parsing_error_body() -> None:
    class DeferredJsonResponse:
        def __init__(self) -> None:
            self.status_code = 503
            self.text = '{"error":{"code":"upstream_unavailable","message":"Consult workflow dependency was unavailable.","details":{"dependency":"bedrock"}}}'
            self.read_called = False

        def json(self) -> dict[str, object]:
            if not self.read_called:
                raise httpx.ResponseNotRead()
            return {
                "error": {
                    "code": "upstream_unavailable",
                    "message": "Consult workflow dependency was unavailable.",
                    "details": {"dependency": "bedrock"},
                }
            }

        def read(self) -> bytes:
            self.read_called = True
            return self.text.encode("utf-8")

    client = UiApiClient(base_url="http://127.0.0.1:8000")

    error = client._build_error(DeferredJsonResponse())

    assert error.code == "upstream_unavailable"
    assert error.status_code == 503
