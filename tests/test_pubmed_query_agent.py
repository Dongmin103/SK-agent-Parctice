from __future__ import annotations

import os

import pytest

from app.agents.pubmed_query_agent import (
    PubMedQueryAgentResult,
    PubMedQueryPlan,
    PubMedQueryPlannerAgent,
    build_bedrock_pubmed_query_agent,
)
from app.clients.pubmed import PubMedClient
from app.domain.models import PubMedQueryInput


class SequenceRunnable:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        if not self._responses:
            raise RuntimeError("No more stubbed responses")
        return self._responses.pop(0)


def make_client() -> PubMedClient:
    return PubMedClient(tool="agentic-ai-poc-tests", email="tests@example.com")


def test_agent_chooses_best_query_from_dry_run(monkeypatch) -> None:
    client = make_client()
    planner = SequenceRunnable(
        [
            PubMedQueryPlan(
                question_type="safety",
                primary_terms=["cardiotoxicity", "hERG"],
                optional_terms=["QT prolongation"],
                reasoning="toxicity-focused query",
                confidence=0.9,
            )
        ]
    )
    agent = PubMedQueryPlannerAgent(client, planner_runnable=planner)

    hit_map = {
        '"KRAS G12C" AND ("cardiotoxicity" OR "hERG")': ["1", "2"],
        '"KRAS G12C" AND ("cardiotoxicity" OR "hERG" OR "QT prolongation")': ["1"],
        '"KRAS G12C"': [],
    }
    monkeypatch.setattr(client, "search_pubmed", lambda query, retmax=10: hit_map.get(query, []))

    result = agent.plan(
        PubMedQueryInput(
            question="이 화합물의 심장독성 위험은?",
            question_type="safety",
            target="KRAS G12C",
        )
    )

    assert isinstance(result, PubMedQueryAgentResult)
    assert result.selected_query == '"KRAS G12C" AND ("cardiotoxicity" OR "hERG")'
    assert result.fallback_used is False
    assert result.used_llm is True


def test_agent_revises_when_first_attempt_has_no_hits(monkeypatch) -> None:
    client = make_client()
    planner = SequenceRunnable(
        [
            PubMedQueryPlan(
                question_type="safety",
                primary_terms=["overly specific toxicity phrase"],
                reasoning="too narrow first plan",
                confidence=0.7,
            )
        ]
    )
    reviser = SequenceRunnable(
        [
            PubMedQueryPlan(
                question_type="safety",
                primary_terms=["cardiotoxicity", "hERG"],
                reasoning="broadened plan after zero hits",
                confidence=0.85,
            )
        ]
    )
    agent = PubMedQueryPlannerAgent(
        client,
        planner_runnable=planner,
        reviser_runnable=reviser,
        max_revision_attempts=2,
    )

    hit_map = {
        '"KRAS G12C" AND "overly specific toxicity phrase"': [],
        '"KRAS G12C"': [],
        '"KRAS G12C" AND ("cardiotoxicity" OR "hERG")': ["1", "2", "3"],
    }
    monkeypatch.setattr(client, "search_pubmed", lambda query, retmax=10: hit_map.get(query, []))

    result = agent.plan(
        PubMedQueryInput(
            question="이 화합물의 심장독성 위험은?",
            question_type="safety",
            target="KRAS G12C",
        )
    )

    assert result.selected_query == '"KRAS G12C" AND ("cardiotoxicity" OR "hERG")'
    assert result.revision_attempts == 1


def test_agent_falls_back_when_plan_contains_smiles_like_terms(monkeypatch) -> None:
    client = make_client()
    planner = SequenceRunnable(
        [
            PubMedQueryPlan(
                question_type="safety",
                primary_terms=["C1=CC=CC=C1", "hERG"],
                reasoning="contains a raw SMILES-like term",
                confidence=0.5,
            )
        ]
    )
    agent = PubMedQueryPlannerAgent(client, planner_runnable=planner, reviser_runnable=None)

    hit_map = {
        '"KRAS G12C" AND "hERG"': [],
        '"KRAS G12C" AND (cardiotoxicity OR hERG OR QT prolongation OR hepatotoxicity OR CYP)': ["1"],
        '"KRAS G12C"': [],
    }
    monkeypatch.setattr(client, "search_pubmed", lambda query, retmax=10: hit_map.get(query, []))

    result = agent.plan(
        PubMedQueryInput(
            question="이 화합물의 심장독성 위험은?",
            question_type="safety",
            target="KRAS G12C",
        )
    )

    assert result.fallback_used is True
    assert any("smiles_like" in issue for issue in result.validation_issues)
    assert result.selected_query.startswith('"KRAS G12C"')


def test_agent_escapes_quotes_and_backslashes_when_compiling_queries(monkeypatch) -> None:
    client = make_client()
    planner = SequenceRunnable(
        [
            PubMedQueryPlan(
                question_type="safety",
                primary_terms=["DNA damage response", "BCR ABL"],
                reasoning="preserve literal quote and backslash characters in compiled queries",
                confidence=0.82,
            )
        ]
    )
    agent = PubMedQueryPlannerAgent(client, planner_runnable=planner, max_revision_attempts=0)

    seen_queries: list[str] = []

    def fake_search(query: str, retmax: int = 10) -> list[str]:
        seen_queries.append(query)
        return []

    monkeypatch.setattr(client, "search_pubmed", fake_search)

    result = agent.plan(
        PubMedQueryInput(
            question="따옴표와 백슬래시가 들어간 질의를 안전하게 만들어줘",
            question_type="safety",
            target='KRAS "G12C"',
            compound_name=r'ABC\123',
        )
    )

    assert result.candidate_queries
    assert seen_queries == result.candidate_queries
    assert '"KRAS \\"G12C\\""' in result.candidate_queries[0]
    assert '"ABC\\\\123"' in result.candidate_queries[0]
    assert '"DNA damage response"' in result.candidate_queries[0]
    assert '"BCR ABL"' in result.candidate_queries[0]


def test_build_bedrock_agent_uses_region_from_environment(monkeypatch) -> None:
    client = make_client()
    captured: dict[str, object] = {}

    def fake_from_bedrock(cls, pubmed_client, **kwargs):
        captured["pubmed_client"] = pubmed_client
        captured.update(kwargs)
        return "agent-from-bedrock"

    monkeypatch.setenv("BEDROCK_PUBMED_QUERY_MODEL_ID", "anthropic.test-model-v1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-northeast-2")
    monkeypatch.setattr(PubMedQueryPlannerAgent, "from_bedrock", classmethod(fake_from_bedrock))

    result = build_bedrock_pubmed_query_agent(client)

    assert result == "agent-from-bedrock"
    assert captured["pubmed_client"] is client
    assert captured["model_id"] == "anthropic.test-model-v1"
    assert captured["region_name"] == "ap-northeast-2"


@pytest.mark.bedrock_live
def test_live_bedrock_planner_generates_query_plan(monkeypatch) -> None:
    model_id = os.environ.get("BEDROCK_PUBMED_QUERY_MODEL_ID")
    region_name = (
        os.environ.get("BEDROCK_AWS_REGION")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
    )
    if not model_id or not region_name:
        pytest.skip(
            "Set BEDROCK_PUBMED_QUERY_MODEL_ID and one of "
            "BEDROCK_AWS_REGION/AWS_REGION/AWS_DEFAULT_REGION to run this test."
        )

    client = make_client()
    monkeypatch.setattr(client, "search_pubmed", lambda query, retmax=10: ["1"] if query else [])

    agent = build_bedrock_pubmed_query_agent(
        client,
        max_revision_attempts=0,
        retmax=1,
    )

    result = agent.plan(
        PubMedQueryInput(
            question="KRAS G12C 억제제의 심장독성 위험 근거를 찾아줘",
            question_type="safety",
            target="KRAS G12C",
            compound_name="adagrasib",
        )
    )

    assert result.used_llm is True
    assert result.candidate_queries
    assert result.selected_query
