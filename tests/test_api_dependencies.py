from __future__ import annotations

from app.api.dependencies import _build_evidence_coordinator
from app.api.settings import AppSettings


def test_build_evidence_coordinator_wires_pubmed_query_planner_when_model_is_configured(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    planner = object()

    def fake_build_bedrock_pubmed_query_agent(pubmed_client, **kwargs):
        captured["pubmed_client"] = pubmed_client
        captured.update(kwargs)
        return planner

    monkeypatch.setattr(
        "app.api.dependencies.build_bedrock_pubmed_query_agent",
        fake_build_bedrock_pubmed_query_agent,
    )

    coordinator = _build_evidence_coordinator(
        AppSettings(
            tool_name="agentic-ai-poc-tests",
            contact_email="tests@example.com",
            pubmed_query_planner_model_id="anthropic.planner-v1",
            bedrock_region_name="ap-northeast-2",
        )
    )

    assert coordinator.pubmed_query_planner is planner
    assert captured["pubmed_client"] is coordinator.pubmed_client
    assert captured["model_id"] == "anthropic.planner-v1"
    assert captured["region_name"] == "ap-northeast-2"
