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

WALTER_AGENT_ID = "walter"
LOW_RISK_MARKERS = {"", "low", "none", "negative", "normal", "not_detected"}
CHEMISTRY_LABELS = {
    "sar": "SAR/scaffold",
    "solubility": "solubility",
    "lipophilicity": "lipophilicity",
    "permeability": "permeability",
}

WALTER_SYSTEM_PROMPT = """You are Walter, a medicinal chemistry, SAR, and structure-property expert.

Rules:
- Use only the provided prediction bundle and evidence items.
- Focus on scaffold changes, SAR, and structure/property optimization guidance.
- Return only the structured schema fields.
- Keep risks and recommendations concise and action-oriented.
- Cite only URLs already present in the evidence items.
"""

WALTER_USER_TEMPLATE = """Question: {question}
Target: {target}
Compound name: {compound_name}
Prediction bundle: {prediction_bundle}
Evidence items: {evidence_items}

Produce a machine-readable Walter chemistry finding."""


class WalterFindingOutput(BaseModel):
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


class WalterAnalyzerRunnable(Protocol):
    def invoke(self, payload: Mapping[str, object]) -> WalterFindingOutput | Mapping[str, object]:
        ...


class WalterAgent:
    """Analyze chemistry, SAR, and structure-property evidence for Walter."""

    def __init__(
        self,
        *,
        analyzer_runnable: WalterAnalyzerRunnable | None = None,
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
    ) -> "WalterAgent":
        llm = ChatBedrockConverse(
            model=model_id,
            region_name=region_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        chain = (
            ChatPromptTemplate.from_messages(
                [("system", WALTER_SYSTEM_PROMPT), ("user", WALTER_USER_TEMPLATE)]
            )
            | llm.with_structured_output(WalterFindingOutput)
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
            source="chemistry",
            query="",
            items=[],
            source_health="degraded",
            missing_reason="walter_evidence_unavailable",
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
            if not isinstance(output, WalterFindingOutput):
                output = WalterFindingOutput.model_validate(output)
        except (ValidationError, ValueError, TypeError) as exc:
            LOGGER.warning("Walter agent output validation failed; using deterministic fallback: %s", exc)
            return self._fallback(packet, prediction_bundle)
        except Exception as exc:
            LOGGER.warning("Walter agent runnable failed; using deterministic fallback: %s", exc)
            return self._fallback(packet, prediction_bundle)

        citations = [url for url in output.citations if url in allowed_citation_set]
        if not citations:
            citations = allowed_citations

        return AgentFinding(
            agent_id=WALTER_AGENT_ID,
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
                agent_id=WALTER_AGENT_ID,
                summary="Walter found limited chemistry evidence for a stronger conclusion.",
                risks=["Insufficient SAR or structure/property signals collected yet."],
                recommendations=[
                    "Collect additional PubChem/ChEMBL chemistry evidence and property predictions."
                ],
                confidence=0.3 if citations else 0.25,
                citations=citations,
            )

        ordered_categories = [
            name
            for name in ("sar", "solubility", "lipophilicity", "permeability")
            if name in categories
        ]
        risks: list[str] = []
        recommendations: list[str] = []

        if "sar" in categories:
            risks.append("SAR/scaffold evidence suggests more analog refinement is needed.")
            recommendations.append(
                "Expand analog exploration around the most informative scaffold changes."
            )
        if "solubility" in categories:
            risks.append("Low solubility could limit chemistry progression.")
            recommendations.append(
                "Prioritize solubility-focused analog design and confirm experimentally."
            )
        if "lipophilicity" in categories:
            risks.append("High lipophilicity/logD may reduce developability.")
            recommendations.append(
                "Reduce lipophilicity/logD with targeted substituent or scaffold edits."
            )
        if "permeability" in categories:
            risks.append("Permeability trade-offs should be checked during optimization.")
            recommendations.append("Balance permeability against polarity during lead optimization.")

        summary = (
            "Walter prioritized "
            + ", ".join(CHEMISTRY_LABELS[name] for name in ordered_categories)
            + " signals from chemistry evidence and property predictions."
        )

        confidence = 0.45
        if prediction_bundle and prediction_bundle.signals:
            confidence += 0.15
        if citations:
            confidence += 0.15
        confidence += min(0.1, 0.03 * len(ordered_categories))

        return AgentFinding(
            agent_id=WALTER_AGENT_ID,
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
            if any(keyword in text for keyword in ("sar", "scaffold", "analog", "analogue", "matched pair")):
                categories.add("sar")
            if any(keyword in text for keyword in ("solubility", "aqueous", "insoluble")):
                categories.add("solubility")
            if any(keyword in text for keyword in ("logp", "logd", "lipophilicity", "lipophilic")):
                categories.add("lipophilicity")
            if any(keyword in text for keyword in ("permeability", "efflux", "polar surface area", "psa")):
                categories.add("permeability")
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
        return any(
            marker in value_text
            for marker in ("high", "elevated", "positive", "moderate", "low", "poor", "insoluble")
        )

    def _evidence_categories(self, evidence_packet: EvidencePacket) -> set[str]:
        categories: set[str] = set()
        for item in evidence_packet.items:
            text = f"{item.title} {item.abstract}".lower()
            if any(
                keyword in text
                for keyword in ("sar", "scaffold", "analog", "analogue", "matched pair", "potency")
            ):
                categories.add("sar")
            if any(keyword in text for keyword in ("solubility", "aqueous", "insoluble")):
                categories.add("solubility")
            if any(keyword in text for keyword in ("logp", "logd", "lipophilicity", "lipophilic")):
                categories.add("lipophilicity")
            if any(keyword in text for keyword in ("permeability", "efflux", "polar surface area", "psa")):
                categories.add("permeability")
        return categories


def build_bedrock_walter_agent(
    *,
    model_id: str | None = None,
    region_name: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 900,
    max_citations: int = 5,
) -> WalterAgent:
    resolved_model_id = model_id or os.environ.get("BEDROCK_WALTER_AGENT_MODEL_ID")
    if not resolved_model_id:
        raise ValueError(
            "A Bedrock model id is required. Set BEDROCK_WALTER_AGENT_MODEL_ID or pass model_id."
        )
    return WalterAgent.from_bedrock(
        model_id=resolved_model_id,
        region_name=region_name,
        temperature=temperature,
        max_tokens=max_tokens,
        max_citations=max_citations,
    )
