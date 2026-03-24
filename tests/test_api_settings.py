from __future__ import annotations

from app.api.settings import AppSettings, load_settings


def test_app_settings_load_defaults_for_local_app_bootstrap(monkeypatch) -> None:
    monkeypatch.delenv("APP_TOOL_NAME", raising=False)
    monkeypatch.delenv("APP_CONTACT_EMAIL", raising=False)
    monkeypatch.delenv("APP_HTTP_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("APP_CACHE_TTL_SECONDS", raising=False)
    monkeypatch.delenv("APP_USE_STUB_PREDICTIONS", raising=False)
    monkeypatch.delenv("TXGEMMA_SAGEMAKER_ENDPOINT_NAME", raising=False)
    monkeypatch.delenv("TXGEMMA_ENDPOINT_NAME", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    settings = AppSettings.from_env()

    assert settings.tool_name == "agentic-ai-poc"
    assert settings.contact_email == "local@example.com"
    assert settings.http_timeout_seconds == 10.0
    assert settings.cache_ttl_seconds == 86_400
    assert settings.use_stub_predictions is False
    assert settings.txgemma_endpoint_name is None
    assert settings.txgemma_region_name is None


def test_app_settings_read_env_overrides_and_bool_flags(monkeypatch) -> None:
    monkeypatch.setenv("APP_TOOL_NAME", "sk-poc")
    monkeypatch.setenv("APP_CONTACT_EMAIL", "team@example.com")
    monkeypatch.setenv("APP_HTTP_TIMEOUT_SECONDS", "3.5")
    monkeypatch.setenv("APP_CACHE_TTL_SECONDS", "120")
    monkeypatch.setenv("APP_USE_STUB_PREDICTIONS", "true")
    monkeypatch.setenv("TXGEMMA_ENDPOINT_NAME", "txgemma-endpoint")
    monkeypatch.setenv("AWS_REGION", "ap-northeast-2")
    monkeypatch.setenv("BEDROCK_WALTER_AGENT_MODEL_ID", "walter-model")
    monkeypatch.setenv("BEDROCK_PUBMED_QUERY_MODEL_ID", "pubmed-query-model")

    settings = AppSettings.from_env()

    assert settings.tool_name == "sk-poc"
    assert settings.contact_email == "team@example.com"
    assert settings.http_timeout_seconds == 3.5
    assert settings.cache_ttl_seconds == 120
    assert settings.use_stub_predictions is True
    assert settings.txgemma_endpoint_name == "txgemma-endpoint"
    assert settings.txgemma_region_name == "ap-northeast-2"
    assert settings.walter_agent_model_id == "walter-model"
    assert settings.pubmed_query_planner_model_id == "pubmed-query-model"


def test_load_settings_reads_dotenv_local_and_allows_env_override(tmp_path) -> None:
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "\n".join(
            [
                "# local runtime wiring",
                "APP_TOOL_NAME=file-tool",
                "TXGEMMA_SAGEMAKER_ENDPOINT_NAME=huggingface-pytorch-tgi-inference-2026-03-23-07-40-06-070",
                "TXGEMMA_AWS_REGION=ap-southeast-2",
                "",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(
        {
            "APP_TOOL_NAME": "env-tool",
        },
        cwd=tmp_path,
    )

    assert settings.tool_name == "env-tool"
    assert settings.txgemma_endpoint_name == "huggingface-pytorch-tgi-inference-2026-03-23-07-40-06-070"
    assert settings.txgemma_region_name == "ap-southeast-2"
