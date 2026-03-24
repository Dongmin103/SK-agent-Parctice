from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.domain.models import AgentFinding, CitationValidation

GENERAL_REVIEW_REASON = "연구용 결과이므로 사람 검토가 필요합니다."
INSUFFICIENT_EVIDENCE_REASON = "근거가 충분하지 않아 추가 검토가 필요합니다."
INCOMPLETE_CITATION_REASON = "일부 결과의 인용 완전성이 확인되지 않았습니다."


@dataclass(slots=True)
class ReviewDecision:
    review_required: bool = True
    reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.review_required is not True:
            raise ValueError("review_required must always be True")


class ReviewPolicy:
    """Deterministic review gate for consult and executive outputs."""

    def evaluate(
        self,
        *,
        citation_validation: CitationValidation,
        findings: list[AgentFinding],
    ) -> ReviewDecision:
        reasons: list[str] = []

        self._append_reason(reasons, GENERAL_REVIEW_REASON)

        if not findings:
            self._append_reason(reasons, INSUFFICIENT_EVIDENCE_REASON)

        if not getattr(citation_validation, "complete", False):
            missing_agent_ids = self._normalize_missing_agent_ids(
                getattr(citation_validation, "missing_agent_ids", [])
            )
            if missing_agent_ids:
                self._append_reason(
                    reasons,
                    f"인용이 누락된 전문가: {', '.join(missing_agent_ids)}",
                )
            else:
                self._append_reason(reasons, INCOMPLETE_CITATION_REASON)

        return ReviewDecision(review_required=True, reasons=reasons)

    def _append_reason(self, reasons: list[str], reason: str) -> None:
        cleaned = " ".join(str(reason).split())
        if cleaned and cleaned not in reasons:
            reasons.append(cleaned)

    def _normalize_missing_agent_ids(self, missing_agent_ids: Any) -> list[str]:
        normalized: list[str] = []
        for agent_id in missing_agent_ids or []:
            cleaned = " ".join(str(agent_id).split())
            if cleaned:
                normalized.append(cleaned)
        return normalized
