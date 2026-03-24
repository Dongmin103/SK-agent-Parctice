from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PredictionRow:
    label: str
    value: str
    meta: str = ""


@dataclass(slots=True)
class FindingViewModel:
    agent_id: str
    summary: str
    risks: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    confidence: float = 0.0
    citations: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvidenceSourceSummary:
    source: str
    health: str
    item_count: int
    query: str = ""
    missing_reason: str | None = None


@dataclass(slots=True)
class ConsultViewModel:
    selected_agents: list[str]
    routing_reason: str
    answer: str
    prediction_rows: list[PredictionRow] = field(default_factory=list)
    findings: list[FindingViewModel] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    missing_signals: list[str] = field(default_factory=list)
    review_label: str = ""
    citation_count: int = 0


@dataclass(slots=True)
class ExecutiveViewModel:
    decision: str
    decision_label: str
    rationale: str
    summary: str
    next_steps: list[str] = field(default_factory=list)
    molecule_svg: str | None = None
    canonical_smiles: str | None = None
    prediction_rows: list[PredictionRow] = field(default_factory=list)
    evidence_sources: list[EvidenceSourceSummary] = field(default_factory=list)
    findings: list[FindingViewModel] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    review_required: bool = True
    review_reasons: list[str] = field(default_factory=list)


def build_consult_view_model(payload: dict[str, Any]) -> ConsultViewModel:
    predictions = payload.get("predictions", {})
    citations = _string_list(payload.get("citations"))
    return ConsultViewModel(
        selected_agents=_string_list(payload.get("selected_agents")),
        routing_reason=str(payload.get("routing_reason", "")),
        answer=str(payload.get("consulting_answer", "")),
        prediction_rows=_build_prediction_rows(predictions),
        findings=_build_findings(payload.get("agent_findings")),
        citations=citations,
        missing_signals=_string_list(predictions.get("missing_signals") if isinstance(predictions, dict) else []),
        review_label="Human review required" if bool(payload.get("review_required", True)) else "Ready",
        citation_count=len(citations),
    )


def build_executive_view_model(payload: dict[str, Any]) -> ExecutiveViewModel:
    decision = payload.get("executive_decision", {})
    predictions = payload.get("predictions", {})
    evidence_bundle = payload.get("evidence_bundle", {})
    return ExecutiveViewModel(
        decision=str(decision.get("decision", "no_go")),
        decision_label=_humanize_decision(str(decision.get("decision", "no_go"))),
        rationale=str(decision.get("rationale", "")),
        summary=str(payload.get("executive_summary", "")),
        next_steps=_string_list(decision.get("next_steps") if isinstance(decision, dict) else []),
        molecule_svg=_optional_str(payload.get("molecule_svg")),
        canonical_smiles=_optional_str(payload.get("canonical_smiles")),
        prediction_rows=_build_prediction_rows(predictions),
        evidence_sources=_build_evidence_sources(evidence_bundle),
        findings=_build_findings(payload.get("agent_findings")),
        citations=_string_list(payload.get("citations")),
        review_required=bool(payload.get("review_required", True)),
        review_reasons=_string_list(payload.get("review_reasons")),
    )


def _build_prediction_rows(predictions: Any) -> list[PredictionRow]:
    if not isinstance(predictions, dict):
        return []

    rows: list[PredictionRow] = []
    for signal in predictions.get("signals", []):
        if not isinstance(signal, dict):
            continue
        meta_parts: list[str] = []
        if signal.get("risk_level"):
            meta_parts.append(f"risk: {signal['risk_level']}")
        if signal.get("confidence") is not None:
            try:
                meta_parts.append(f"confidence: {float(signal['confidence']):.2f}")
            except (TypeError, ValueError):
                meta_parts.append(f"confidence: {signal['confidence']}")
        if signal.get("unit"):
            meta_parts.append(f"unit: {signal['unit']}")

        rows.append(
            PredictionRow(
                label=str(signal.get("name", "")),
                value=_stringify_value(signal.get("value")),
                meta=" | ".join(meta_parts),
            )
        )
    return rows


def _build_findings(payload: Any) -> list[FindingViewModel]:
    if not isinstance(payload, list):
        return []

    findings: list[FindingViewModel] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        findings.append(
            FindingViewModel(
                agent_id=str(item.get("agent_id", "")),
                summary=str(item.get("summary", "")),
                risks=_string_list(item.get("risks")),
                recommendations=_string_list(item.get("recommendations")),
                confidence=_float_value(item.get("confidence")),
                citations=_string_list(item.get("citations")),
            )
        )
    return findings


def _build_evidence_sources(payload: Any) -> list[EvidenceSourceSummary]:
    if not isinstance(payload, dict):
        return []

    packets = payload.get("packets", {})
    if not isinstance(packets, dict):
        return []

    rows: list[EvidenceSourceSummary] = []
    for source, packet in packets.items():
        if not isinstance(packet, dict):
            continue
        items = packet.get("items")
        item_count = len(items) if isinstance(items, list) else 0
        rows.append(
            EvidenceSourceSummary(
                source=str(source),
                health=str(packet.get("source_health", "ok")),
                item_count=item_count,
                query=str(packet.get("query", "")),
                missing_reason=_optional_str(packet.get("missing_reason")),
            )
        )
    return rows


def _humanize_decision(decision: str) -> str:
    mapping = {
        "go": "Go",
        "conditional_go": "Conditional go",
        "no_go": "No go",
    }
    return mapping.get(decision, decision.replace("_", " ").title())


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _stringify_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
