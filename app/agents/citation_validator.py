from __future__ import annotations

from app.domain.models import AgentFinding, CitationValidation


class CitationValidator:
    """Validate citation completeness for synthesized answers."""

    def validate(self, findings: list[AgentFinding]) -> CitationValidation:
        unique_citations: list[str] = []
        seen_citations: set[str] = set()
        missing_agent_ids: list[str] = []

        for finding in findings:
            normalized_citations = self._normalize_citations(finding.citations)
            if not normalized_citations:
                missing_agent_ids.append(finding.agent_id)
            for citation in normalized_citations:
                if citation in seen_citations:
                    continue
                seen_citations.add(citation)
                unique_citations.append(citation)

        complete = bool(findings) and not missing_agent_ids
        return CitationValidation(
            complete=complete,
            missing_agent_ids=missing_agent_ids,
            total_citations=len(unique_citations),
        )

    def _normalize_citations(self, citations: list[str]) -> list[str]:
        normalized: list[str] = []
        for citation in citations:
            item = str(citation).strip()
            if not item:
                continue
            normalized.append(item)
        return normalized
