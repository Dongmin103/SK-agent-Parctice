from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(slots=True)
class UiApiError(Exception):
    code: str
    message: str
    details: object | None = None
    status_code: int | None = None

    def __str__(self) -> str:
        return self.message


class UiApiClient:
    def __init__(
        self,
        *,
        base_url: str,
        http_client: httpx.Client | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.http_client = http_client or httpx.Client()

    def submit_consult(
        self,
        *,
        smiles: str,
        target: str,
        question: str,
        compound_name: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "smiles": smiles,
            "target": target,
            "question": question,
            "compound_name": compound_name,
        }
        return self._post_json("/api/reports/consult", payload)

    def submit_executive(
        self,
        *,
        smiles: str,
        target: str,
        compound_name: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "smiles": smiles,
            "target": target,
            "compound_name": compound_name,
        }
        return self._post_json("/api/reports/executive", payload)

    def stream_consult(
        self,
        *,
        smiles: str,
        target: str,
        question: str,
        compound_name: str | None = None,
    ):
        payload = {
            "smiles": smiles,
            "target": target,
            "question": question,
            "compound_name": compound_name,
        }
        yield from self._stream_json("/api/reports/consult/stream", payload)

    def stream_executive(
        self,
        *,
        smiles: str,
        target: str,
        compound_name: str | None = None,
    ):
        payload = {
            "smiles": smiles,
            "target": target,
            "compound_name": compound_name,
        }
        yield from self._stream_json("/api/reports/executive/stream", payload)

    def _post_json(self, path: str, payload: dict[str, object]) -> dict[str, Any]:
        try:
            response = self.http_client.post(
                f"{self.base_url}{path}",
                json=payload,
                timeout=self.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise UiApiError(
                code="timeout",
                message=(
                    f"FastAPI at {self.base_url} did not respond within "
                    f"{self.timeout_seconds:.0f}s"
                ),
                details=str(exc),
            ) from exc
        except httpx.RequestError as exc:
            raise UiApiError(
                code="network_error",
                message=f"Unable to reach FastAPI at {self.base_url}",
                details=str(exc),
            ) from exc

        if response.is_success:
            return self._response_json(response)

        raise self._build_error(response)

    def _stream_json(self, path: str, payload: dict[str, object]):
        try:
            with self.http_client.stream(
                "POST",
                f"{self.base_url}{path}",
                json=payload,
                timeout=self.timeout_seconds,
            ) as response:
                if not response.is_success:
                    raise self._build_error(response)

                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if not isinstance(chunk, dict):
                        continue
                    if chunk.get("type") == "error":
                        raise self._stream_error_to_ui_error(chunk)
                    yield chunk
        except httpx.TimeoutException as exc:
            raise UiApiError(
                code="timeout",
                message=(
                    f"FastAPI at {self.base_url} did not respond within "
                    f"{self.timeout_seconds:.0f}s"
                ),
                details=str(exc),
            ) from exc
        except httpx.RequestError as exc:
            raise UiApiError(
                code="network_error",
                message=f"Unable to reach FastAPI at {self.base_url}",
                details=str(exc),
            ) from exc

    def _build_error(self, response: httpx.Response) -> UiApiError:
        payload = self._response_json(response)
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            return UiApiError(
                code=str(error.get("code", "api_error")),
                message=str(error.get("message", f"API request failed with {response.status_code}")),
                details=error.get("details"),
                status_code=response.status_code,
            )
        return UiApiError(
            code="api_error",
            message=f"API request failed with status {response.status_code}",
            details=self._response_text(response),
            status_code=response.status_code,
        )

    def _response_json(self, response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except httpx.ResponseNotRead:
            response.read()
            try:
                payload = response.json()
            except ValueError:
                return {}
        except ValueError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _response_text(self, response: httpx.Response) -> str:
        try:
            return response.text
        except httpx.ResponseNotRead:
            response.read()
            return response.text

    def _stream_error_to_ui_error(self, chunk: dict[str, Any]) -> UiApiError:
        error = chunk.get("error")
        if isinstance(error, dict):
            return UiApiError(
                code=str(error.get("code", "api_error")),
                message=str(error.get("message", "Streamed API request failed.")),
                details=error.get("details"),
            )
        return UiApiError(
            code="api_error",
            message="Streamed API request failed.",
        )
