from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Protocol

from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError, field_validator

from app.domain.models import AgentFinding, EvidencePacket, PredictionBundle, PredictionSignal

LOGGER = logging.getLogger(__name__)

HARVEY_AGENT_ID = "harvey"
LOW_RISK_MARKERS = {"", "low", "none", "negative", "normal", "not_detected"}
STRATEGY_LABELS = {
    "clinical": "clinical development",
    "regulatory": "regulatory approval",
    "strategy": "competitive strategy",
}

HARVEY_SYSTEM_PROMPT = """You are Harvey, a clinical development, regulatory, and program strategy expert.

Rules:
- Use only the provided prediction bundle and evidence items.
- Focus on clinical development feasibility, approval hurdles, and strategic prioritization.
- Return only the structured schema fields.
- Keep risks and recommendations concise and action-oriented.
- Cite only URLs already present in the evidence items.
"""

HARVEY_USER_TEMPLATE = """Question: {question}
Target: {target}
Compound name: {compound_name}
Prediction bundle: {prediction_bundle}
Evidence items: {evidence_items}

Produce a machine-readable Harvey clinical/regulatory finding."""


class HarveyFindingOutput(BaseModel):
    summary: str = Field(min_length=1)
    risks: list[str] = Field(default_factory=list, min_length=1, max_length=6)
    recommendations: list[str] = Field(default_factory=list, min_length=1, max_length=6)
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[str] = Field(default_factory=list, max_length=6)

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("summary must not be blank")
        return cleaned

    @field_validator("risks", "recommendations", "citations")
    @classmethod
    def normalize_string_list(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            item = " ".join(str(value).split())
            if not item:
                continue
            lowered = item.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            cleaned.append(item)
        return cleaned


class HarveyAnalyzerRunnable(Protocol):
    def invoke(self, payload: Mapping[str, object]) -> HarveyFindingOutput | Mapping[str, object]:
        ...


class HarveyAgent:
    """Analyze clinical development, regulatory, and prioritization evidence for Harvey."""

    def __init__(
        self,
        *,
        analyzer_runnable: HarveyAnalyzerRunnable | None = None,
        max_citations: int = 5,
    ) -> None:
        self.analyzer_runnable = analyzer_runnable
        self.max_citations = max_citations

    @classmethod
    def from_bedrock(
        cls,
        *,
        model_id: str,
        region_name: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 900,
        max_citations: int = 5,
    ) -> "HarveyAgent":
        llm = ChatBedrockConverse(
            model=model_id,
            region_name=region_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        chain = (
            ChatPromptTemplate.from_messages(
                [("system", HARVEY_SYSTEM_PROMPT), ("user", HARVEY_USER_TEMPLATE)]
            )
            | llm.with_structured_output(HarveyFindingOutput)
        )
        return cls(analyzer_runnable=chain, max_citations=max_citations)

    def analyze(
        self,
        question: str,
        *,
        target: str | None = None,
        compound_name: str | None = None,
        evidence_packet: EvidencePacket | None = None,
        prediction_bundle: PredictionBundle | None = None,
    ) -> AgentFinding:
        packet = evidence_packet or EvidencePacket(
            source="clinical_regulatory",
            query="",
            items=[],
            source_health="degraded",
            missing_reason="harvey_evidence_unavailable",
        )
        allowed_citations = self._allowed_citations(packet)
        allowed_citation_set = set(allowed_citations)
        payload = {
            "question": question,
            "target": target or "",
            "compound_name": compound_name or "",
            "prediction_bundle": self._serialize_prediction_bundle(prediction_bundle),
            "evidence_items": self._serialize_evidence_items(packet),
        }

        if self.analyzer_runnable is None:
            return self._fallback(packet, prediction_bundle)

        try:
            output = self.analyzer_runnable.invoke(payload)
            if not isinstance(output, HarveyFindingOutput):
                output = HarveyFindingOutput.model_validate(output)
        except (ValidationError, ValueError, TypeError) as exc:
            LOGGER.warning("Harvey agent output validation failed; using deterministic fallback: %s", exc)
            return self._fallback(packet, prediction_bundle)
        except Exception as exc:
            LOGGER.warning("Harvey agent runnable failed; using deterministic fallback: %s", exc)
            return self._fallback(packet, prediction_bundle)

        citations = [url for url in output.citations if url in allowed_citation_set]
        if not citations:
            citations = allowed_citations

        return AgentFinding(
            agent_id=HARVEY_AGENT_ID,
            summary=output.summary,
            risks=output.risks,
            recommendations=output.recommendations,
            confidence=output.confidence,
            citations=citations,
        )

    def _fallback(
        self,
        evidence_packet: EvidencePacket,
        prediction_bundle: PredictionBundle | None,
    ) -> AgentFinding:
        categories = self._detect_categories(evidence_packet, prediction_bundle)
        citations = self._allowed_citations(evidence_packet)

        if not categories:
            return AgentFinding(
                agent_id=HARVEY_AGENT_ID,
                summary="Harvey found limited clinical/regulatory evidence for a stronger conclusion.",
                risks=["Insufficient clinical, regulatory, or strategy signals collected yet."],
                recommendations=[
                    "Collect additional ClinicalTrials.gov/openFDA evidence and program strategy inputs."
                ],
                confidence=0.3 if citations else 0.25,
                citations=citations,
            )

        ordered_categories = [
            name for name in ("clinical", "regulatory", "strategy") if name in categories
        ]
        risks: list[str] = []
        recommendations: list[str] = []

        if "clinical" in categories:
            risks.append(
                "Clinical development evidence still leaves execution risk around trial design or durability."
            )
            recommendations.append("Review ongoing trial design, endpoints, and enrollment feasibility.")
        if "regulatory" in categories:
            risks.append("Regulatory approval hurdles remain open and need explicit evidence.")
            recommendations.append("Map the approval pathway with confirmatory evidence requirements.")
        if "strategy" in categories:
            risks.append("Competitive strategy pressure could weaken program prioritization.")
            recommendations.append("Clarify differentiation versus competitors before major investment.")

        summary = (
            "Harvey prioritized "
            + ", ".join(STRATEGY_LABELS[name] for name in ordered_categories)
            + " signals from evidence and predictions."
        )

        confidence = 0.45
        if prediction_bundle and prediction_bundle.signals:
            confidence += 0.15
        if citations:
            confidence += 0.15
        confidence += min(0.1, 0.03 * len(ordered_categories))

        return AgentFinding(
            agent_id=HARVEY_AGENT_ID,
            summary=summary,
            risks=risks,
            recommendations=recommendations,
            confidence=min(confidence, 0.85),
            citations=citations,
        )

    def _allowed_citations(self, evidence_packet: EvidencePacket) -> list[str]:
        citations: list[str] = []
        seen: set[str] = set()
        for item in evidence_packet.items:
            if not item.url or item.url in seen:
                continue
            seen.add(item.url)
            citations.append(item.url)
            if len(citations) >= self.max_citations:
                break
        return citations

    def _serialize_prediction_bundle(
        self, prediction_bundle: PredictionBundle | None
    ) -> dict[str, object]:
        if prediction_bundle is None:
            return {"source": "", "signals": [], "generated_at": ""}
        return {
            "source": prediction_bundle.source,
            "generated_at": prediction_bundle.generated_at,
            "signals": [
                {
                    "name": signal.name,
                    "value": signal.value,
                    "unit": signal.unit or "",
                    "confidence": signal.confidence,
                    "risk_level": signal.risk_level or "",
                }
                for signal in prediction_bundle.signals
            ],
        }

    def _serialize_evidence_items(self, evidence_packet: EvidencePacket) -> list[dict[str, object]]:
        return [
            {
                "pmid": item.pmid,
                "title": item.title,
                "abstract": item.abstract,
                "journal": item.journal,
                "pub_year": item.pub_year,
                "score": item.score,
                "url": item.url,
                "missing_reason": item.missing_reason or "",
            }
            for item in evidence_packet.items
        ]

    def _detect_categories(
        self,
        evidence_packet: EvidencePacket,
        prediction_bundle: PredictionBundle | None,
    ) -> set[str]:
        categories = set()
        categories.update(self._prediction_categories(prediction_bundle))
        categories.update(self._evidence_categories(evidence_packet))
        return categories

    def _prediction_categories(self, prediction_bundle: PredictionBundle | None) -> set[str]:
        if prediction_bundle is None:
            return set()

        categories: set[str] = set()
        for signal in prediction_bundle.signals:
            if not self._signal_is_flagged(signal):
                continue

            text = f"{signal.name} {signal.value} {signal.risk_level or ''}".lower()
            if any(
                keyword in text
                for keyword in ("clinical", "trial", "phase", "enrollment", "endpoint", "durability")
            ):
                categories.add("clinical")
            if any(
                keyword in text
                for keyword in ("approval", "regulatory", "fda", "confirmatory", "label")
            ):
                categories.add("regulatory")
            if any(
                keyword in text
                for keyword in ("competitive", "competition", "priority", "prioritization", "strategy", "different")
            ):
                categories.add("strategy")
        return categories

    def _signal_is_flagged(self, signal: PredictionSignal) -> bool:
        risk_level = (signal.risk_level or "").strip().lower()
        if risk_level not in LOW_RISK_MARKERS:
            return True

        value = signal.value
        if isinstance(value, bool):
            return value
        value_text = str(value).strip().lower()
        if not value_text:
            return False
        return any(marker in value_text for marker in ("high", "elevated", "positive", "moderate"))

    def _evidence_categories(self, evidence_packet: EvidencePacket) -> set[str]:
        categories: set[str] = set()
        for item in evidence_packet.items:
            text = f"{item.title} {item.abstract}".lower()
            if any(
                keyword in text
                for keyword in ("phase", "trial", "enrollment", "clinical development", "durability")
            ):
                categories.add("clinical")
            if any(
                keyword in text
                for keyword in ("approval", "regulatory", "fda", "confirmatory", "label", "briefing")
            ):
                categories.add("regulatory")
            if any(
                keyword in text
                for keyword in ("competitive", "competition", "priority", "prioritization", "strategy", "different")
            ):
                categories.add("strategy")
        return categories


def build_bedrock_harvey_agent(
    *,
    model_id: str | None = None,
    region_name: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 900,
    max_citations: int = 5,
) -> HarveyAgent:
    resolved_model_id = model_id or os.environ.get("BEDROCK_HARVEY_AGENT_MODEL_ID")
    if not resolved_model_id:
        raise ValueError(
            "A Bedrock model id is required. Set BEDROCK_HARVEY_AGENT_MODEL_ID or pass model_id."
        )
    return HarveyAgent.from_bedrock(
        model_id=resolved_model_id,
        region_name=region_name,
        temperature=temperature,
        max_tokens=max_tokens,
        max_citations=max_citations,
    )
