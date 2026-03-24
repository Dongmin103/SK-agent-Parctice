from __future__ import annotations

from typing import Iterable

from app.agents.citation_validator import CitationValidator
from app.domain.models import AgentFinding, DecisionDraft, ExecutiveSynthesis


BLOCKING_KEYWORDS = (
    "block",
    "blocking",
    "halt",
    "stop",
    "no-go",
    "nogo",
    "cannot advance",
    "not advance",
    "중단",
    "진행 불가",
    "개발 중단",
    "불가",
    "차단",
)
class CEOSynthesizer:
    """Combine expert findings into a conservative executive recommendation."""

    def __init__(self, *, citation_validator: CitationValidator | None = None) -> None:
        self.citation_validator = citation_validator or CitationValidator()

    def synthesize(self, findings: list[AgentFinding]) -> ExecutiveSynthesis:
        citation_validation = self.citation_validator.validate(findings)
        citations = self._collect_citations(findings)
        next_steps = self._collect_next_steps(findings)

        if not findings:
            decision_draft = DecisionDraft(
                decision="no_go",
                rationale="수집된 전문가 근거가 없어 현재 시점의 실행 판단을 내리기 어렵습니다.",
                next_steps=next_steps or ["추가 근거를 수집한 뒤 다시 검토합니다."],
            )
            return ExecutiveSynthesis(
                summary="CEO 합성 결과: 현재는 검토 가능한 근거가 부족합니다.",
                decision_draft=decision_draft,
                citations=citations,
                citation_validation=citation_validation,
            )

        if self._has_blocking_language(findings):
            decision_draft = DecisionDraft(
                decision="no_go",
                rationale="명확한 중단 또는 차단 신호가 확인되어 진행을 권장하지 않습니다.",
                next_steps=next_steps or ["중단 사유를 정리하고 추가 검증을 수행합니다."],
            )
            return ExecutiveSynthesis(
                summary="CEO 합성 결과: 차단성 리스크가 확인되었습니다.",
                decision_draft=decision_draft,
                citations=citations,
                citation_validation=citation_validation,
            )

        if not citation_validation.complete:
            decision = "conditional_go"
            rationale = "일부 근거가 출처 없이 남아 있어 조건부 검토가 필요합니다."
        elif self._has_risk_or_recommendation(findings):
            decision = "conditional_go"
            rationale = "근거는 충분하지만 추가 확인이 필요한 리스크와 후속 조치가 남아 있습니다."
        else:
            decision = "go"
            rationale = "일관된 근거와 권고가 확인되어 진행 가능성이 높습니다."

        decision_draft = DecisionDraft(
            decision=decision,
            rationale=rationale,
            next_steps=next_steps,
        )
        summary = self._build_summary(findings, decision)
        return ExecutiveSynthesis(
            summary=summary,
            decision_draft=decision_draft,
            citations=citations,
            citation_validation=citation_validation,
        )

    def _build_summary(self, findings: list[AgentFinding], decision: str) -> str:
        agent_ids = [finding.agent_id for finding in findings]
        if not agent_ids:
            return "CEO 합성 결과: 검토 대상이 없습니다."
        joined_agents = ", ".join(agent_ids)
        return f"CEO 합성 결과: {joined_agents} 관점을 종합한 판단은 {decision} 입니다."

    def _collect_next_steps(self, findings: list[AgentFinding]) -> list[str]:
        next_steps: list[str] = []
        seen: set[str] = set()
        for finding in findings:
            for recommendation in finding.recommendations:
                normalized = " ".join(str(recommendation).split())
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                next_steps.append(normalized)
        return next_steps

    def _collect_citations(self, findings: list[AgentFinding]) -> list[str]:
        citations: list[str] = []
        seen: set[str] = set()
        for finding in findings:
            for citation in finding.citations:
                if citation in seen:
                    continue
                seen.add(citation)
                citations.append(citation)
        return citations

    def _has_risk_or_recommendation(self, findings: list[AgentFinding]) -> bool:
        return any(finding.risks or finding.recommendations for finding in findings)

    def _has_blocking_language(self, findings: list[AgentFinding]) -> bool:
        for finding in findings:
            for text in self._iter_text_fragments(finding):
                lowered = text.lower()
                if any(keyword in lowered for keyword in BLOCKING_KEYWORDS):
                    return True
        return False

    def _iter_text_fragments(self, finding: AgentFinding) -> Iterable[str]:
        yield finding.summary
        yield from finding.risks
        yield from finding.recommendations


__all__ = [
    "CEOSynthesizer",
]
