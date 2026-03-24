from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Generator

import httpx

from app.api.settings import load_settings


def test_load_settings_reads_runtime_configuration_from_env() -> None:
    settings = load_settings(
        {
            "API_HOST": "0.0.0.0",
            "API_PORT": "8111",
            "API_USE_STUB_WORKFLOWS": "true",
            "EUTILS_TOOL": "codex-smoke",
            "EUTILS_EMAIL": "lab@example.com",
            "TXGEMMA_ENDPOINT_NAME": "txgemma-endpoint",
            "AWS_REGION": "ap-northeast-2",
        }
    )

    assert settings.host == "0.0.0.0"
    assert settings.port == 8111
    assert settings.use_stub_workflows is True
    assert settings.eutils_tool == "codex-smoke"
    assert settings.eutils_email == "lab@example.com"
    assert settings.txgemma_endpoint_name == "txgemma-endpoint"
    assert settings.txgemma_region_name == "ap-northeast-2"


def test_runtime_app_can_start_local_server_and_serve_consult_and_executive_smoke_requests() -> None:
    port = _find_free_port()
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["API_USE_STUB_WORKFLOWS"] = "true"

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.api.main:create_runtime_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=".",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        _wait_for_server(f"http://127.0.0.1:{port}/api/reports/consult")

        consult_response = httpx.post(
            f"http://127.0.0.1:{port}/api/reports/consult",
            json={
                "smiles": "CCO",
                "target": "KRAS G12C",
                "question": "이 화합물의 hERG 위험은?",
                "compound_name": "ABC-101",
            },
            timeout=5.0,
        )
        executive_response = httpx.post(
            f"http://127.0.0.1:{port}/api/reports/executive",
            json={
                "smiles": "CCO",
                "target": "KRAS G12C",
                "compound_name": "ABC-101",
            },
            timeout=5.0,
        )

        assert consult_response.status_code == 200
        assert consult_response.json()["selected_agents"] == ["house"]
        assert executive_response.status_code == 200
        assert executive_response.json()["executive_decision"]["decision"] == "conditional_go"
    finally:
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)


def _wait_for_server(url: str, timeout_seconds: float = 10.0) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = httpx.post(
                url,
                json={
                    "smiles": "CCO",
                    "target": "KRAS G12C",
                    "question": "warmup",
                    "compound_name": "ABC-101",
                },
                timeout=1.0,
            )
            if response.status_code in {200, 422, 500}:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(0.2)
    raise AssertionError(f"server did not become ready in time: {last_error}")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
