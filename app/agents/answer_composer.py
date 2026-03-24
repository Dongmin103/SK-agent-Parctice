from __future__ import annotations

from app.agents.citation_validator import CitationValidator
from app.domain.models import AgentFinding, ComposedAnswer

AGENT_LABELS = {
    "walter": "Walter",
    "house": "House",
    "harvey": "Harvey",
}

class AnswerComposer:
    """Compose a consult answer from selected expert findings."""

    def __init__(self, *, citation_validator: CitationValidator | None = None) -> None:
        self.citation_validator = citation_validator or CitationValidator()

    def compose(
        self,
        *,
        selected_agents: list[str],
        findings: list[AgentFinding],
    ) -> ComposedAnswer:
        citation_validation = self.citation_validator.validate(findings)
        citations = self._collect_citations(findings)
        consulting_answer = self._compose_answer_text(selected_agents, findings)
        return ComposedAnswer(
            answer=consulting_answer,
            citations=citations,
            citation_validation=citation_validation,
        )

    def _compose_answer_text(
        self,
        selected_agents: list[str],
        findings: list[AgentFinding],
    ) -> str:
        if not findings:
            return "질문에 대해 검토했지만 현재 수집된 근거가 제한적입니다."

        labels = [AGENT_LABELS[agent_id] for agent_id in selected_agents if agent_id in AGENT_LABELS]
        parts: list[str] = []
        if labels:
            parts.append(f"질문에 대해 {', '.join(labels)} 관점에서 검토했습니다.")
        else:
            parts.append("질문에 대해 검토했습니다.")

        parts.extend(finding.summary for finding in findings if finding.summary)

        recommendations: list[str] = []
        seen_recommendations: set[str] = set()
        for finding in findings:
            for recommendation in finding.recommendations:
                normalized = " ".join(str(recommendation).split())
                if not normalized:
                    continue
                lowered = normalized.lower()
                if lowered in seen_recommendations:
                    continue
                seen_recommendations.add(lowered)
                recommendations.append(normalized)
                if len(recommendations) >= 3:
                    break
            if len(recommendations) >= 3:
                break

        if recommendations:
            parts.append(f"권고: {'; '.join(recommendations)}")

        return " ".join(parts)

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
