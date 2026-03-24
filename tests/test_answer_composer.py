from __future__ import annotations

from app.agents.answer_composer import AnswerComposer
from app.domain import models as domain_models
from app.domain.models import AgentFinding


def make_finding(
    agent_id: str,
    summary: str,
    *,
    recommendations: list[str] | None = None,
    citations: list[str] | None = None,
) -> AgentFinding:
    return AgentFinding(
        agent_id=agent_id,
        summary=summary,
        recommendations=recommendations or [],
        citations=citations or [],
    )


def test_answer_composer_preserves_agent_order_and_deduplicates_recommendations_and_citations() -> None:
    composer = AnswerComposer()
    findings = [
        make_finding(
            "harvey",
            "Harvey summary.",
            recommendations=[
                "Confirm trial feasibility.",
                "Coordinate regulatory strategy.",
            ],
            citations=["https://example.org/c1", "https://example.org/c2"],
        ),
        make_finding(
            "house",
            "House summary.",
            recommendations=[
                "Coordinate regulatory strategy.",
                "Review safety follow-up.",
            ],
            citations=["https://example.org/c2", "https://example.org/c3"],
        ),
    ]

    result = composer.compose(selected_agents=["harvey", "house"], findings=findings)

    assert isinstance(result, domain_models.ComposedAnswer)
    assert result.answer == (
        "질문에 대해 Harvey, House 관점에서 검토했습니다. "
        "Harvey summary. House summary. "
        "권고: Confirm trial feasibility.; Coordinate regulatory strategy.; Review safety follow-up."
    )
    assert result.citations == [
        "https://example.org/c1",
        "https://example.org/c2",
        "https://example.org/c3",
    ]
    assert result.citation_validation.complete is True
    assert result.citation_validation.missing_agent_ids == []
    assert result.citation_validation.total_citations == 3


def test_answer_composer_returns_limited_evidence_message_for_empty_findings() -> None:
    composer = AnswerComposer()

    result = composer.compose(selected_agents=["walter"], findings=[])

    assert result.answer == "질문에 대해 검토했지만 현재 수집된 근거가 제한적입니다."
    assert result.citations == []
    assert result.citation_validation.complete is False
    assert result.citation_validation.missing_agent_ids == []
    assert result.citation_validation.total_citations == 0


def test_answer_composer_marks_incomplete_citations_when_any_finding_is_uncited() -> None:
    composer = AnswerComposer()

    result = composer.compose(
        selected_agents=["walter", "house"],
        findings=[
            make_finding("walter", "Walter summary.", citations=["https://example.org/c1"]),
            make_finding("house", "House summary.", citations=[]),
        ],
    )

    assert result.citation_validation.complete is False
    assert result.citation_validation.missing_agent_ids == ["house"]
    assert result.citation_validation.total_citations == 1
