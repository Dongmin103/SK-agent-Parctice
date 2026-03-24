from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.models import AgentFinding


@dataclass(slots=True)
class CitationValidationStub:
    complete: bool
    missing_agent_ids: list[str] = field(default_factory=list)
    total_citations: int = 0


def test_review_policy_always_requires_human_review_and_reports_missing_agents() -> None:
    from app.agents.review_policy import ReviewPolicy

    policy = ReviewPolicy()
    decision = policy.evaluate(
        citation_validation=CitationValidationStub(
            complete=False,
            missing_agent_ids=["house", "walter"],
            total_citations=1,
        ),
        findings=[
            AgentFinding(
                agent_id="house",
                summary="Safety review completed.",
                citations=["https://example.org/a"],
            )
        ],
    )

    assert decision.review_required is True
    assert decision.reasons[0] == "연구용 결과이므로 사람 검토가 필요합니다."
    assert "인용이 누락된 전문가: house, walter" in decision.reasons


def test_review_policy_marks_insufficient_evidence_when_no_findings() -> None:
    from app.agents.review_policy import ReviewPolicy

    policy = ReviewPolicy()
    decision = policy.evaluate(
        citation_validation=CitationValidationStub(complete=True, total_citations=0),
        findings=[],
    )

    assert decision.review_required is True
    assert decision.reasons == [
        "연구용 결과이므로 사람 검토가 필요합니다.",
        "근거가 충분하지 않아 추가 검토가 필요합니다.",
    ]
