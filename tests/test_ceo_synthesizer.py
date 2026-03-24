from __future__ import annotations

from app.agents.ceo_synthesizer import CEOSynthesizer
from app.domain.models import AgentFinding


def test_ceo_synthesizer_merges_citations_and_next_steps_with_conservative_decision() -> None:
    synth = CEOSynthesizer()
    findings = [
        AgentFinding(
            agent_id="walter",
            summary="Walter summary.",
            risks=["개발상 주의 필요"],
            recommendations=["추가 PK 확인", "추가 PK 확인"],
            citations=[
                "https://example.org/a",
                "https://example.org/b",
            ],
        ),
        AgentFinding(
            agent_id="house",
            summary="House summary.",
            risks=["주의 필요"],
            recommendations=["추가 PK 확인", "규제 검토"],
            citations=[
                "https://example.org/b",
                "https://example.org/c",
            ],
        ),
    ]

    result = synth.synthesize(findings)

    assert result.citations == [
        "https://example.org/a",
        "https://example.org/b",
        "https://example.org/c",
    ]
    assert result.citation_validation.complete is True
    assert result.citation_validation.total_citations == 3
    assert result.decision_draft.decision == "conditional_go"
    assert result.decision_draft.next_steps == ["추가 PK 확인", "규제 검토"]
    assert "종합" in result.summary


def test_ceo_synthesizer_caps_uncited_findings_at_conditional_go() -> None:
    synth = CEOSynthesizer()
    findings = [
        AgentFinding(
            agent_id="walter",
            summary="Walter summary.",
            recommendations=["후속 실험 필요"],
            citations=["https://example.org/a"],
        ),
        AgentFinding(
            agent_id="harvey",
            summary="Harvey summary.",
            recommendations=["규제 검토"],
            citations=[],
        ),
    ]

    result = synth.synthesize(findings)

    assert result.citation_validation.complete is False
    assert result.citation_validation.missing_agent_ids == ["harvey"]
    assert result.decision_draft.decision == "conditional_go"
    assert result.citations == ["https://example.org/a"]


def test_ceo_synthesizer_returns_no_go_for_empty_or_blocking_findings() -> None:
    synth = CEOSynthesizer()

    empty_result = synth.synthesize([])
    assert empty_result.decision_draft.decision == "no_go"
    assert empty_result.citation_validation.complete is False
    assert empty_result.citation_validation.total_citations == 0

    blocking_result = synth.synthesize(
        [
            AgentFinding(
                agent_id="house",
                summary="프로그램 중단이 필요하다.",
                risks=["명확한 중단 사유"],
                citations=["https://example.org/block"],
            )
        ]
    )
    assert blocking_result.decision_draft.decision == "no_go"
