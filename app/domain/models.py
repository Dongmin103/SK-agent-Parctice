from __future__ import annotations

from typing import Any
from dataclasses import dataclass, field
from datetime import datetime, timezone


def utc_now_iso() -> str:
    """Return a compact ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


JsonValue = Any


@dataclass(slots=True)
class PubMedQueryInput:
    question: str
    question_type: str
    target: str | None = None
    compound_name: str | None = None
    prediction_flags: dict[str, str | bool | float] | None = None


@dataclass(slots=True)
class PubMedArticleRaw:
    pmid: str
    title: str
    abstract: str
    journal: str
    pub_year: int | None
    authors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvidenceItem:
    source: str
    pmid: str
    title: str
    abstract: str
    journal: str
    pub_year: int | None
    authors: list[str] = field(default_factory=list)
    url: str = ""
    score: float = 0.0
    fetched_at: str = field(default_factory=utc_now_iso)
    missing_reason: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(slots=True)
class EvidencePacket:
    source: str
    query: str
    items: list[EvidenceItem] = field(default_factory=list)
    fetched_at: str = field(default_factory=utc_now_iso)
    source_health: str = "ok"
    missing_reason: str | None = None
    diagnostics: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class EvidenceBundle:
    query: str
    packets: dict[str, EvidencePacket] = field(default_factory=dict)
    items: list[EvidenceItem] = field(default_factory=list)
    fetched_at: str = field(default_factory=utc_now_iso)
    source_health: str = "ok"
    missing_sources: list[str] = field(default_factory=list)
    partial_failures: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CompoundContext:
    smiles: str
    target: str
    compound_name: str | None = None
    canonical_smiles: str | None = None
    molecule_svg: str | None = None


@dataclass(slots=True)
class PredictionSignal:
    name: str
    value: str | float | int | bool
    unit: str | None = None
    confidence: float | None = None
    risk_level: str | None = None


@dataclass(slots=True)
class PredictionBundle:
    source: str = "txgemma"
    target: str | None = None
    compound_name: str | None = None
    canonical_smiles: str | None = None
    signals: list[PredictionSignal] = field(default_factory=list)
    missing_signals: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=utc_now_iso)


@dataclass(slots=True)
class AgentFinding:
    agent_id: str
    summary: str
    risks: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    confidence: float = 0.0
    citations: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CitationValidation:
    complete: bool = False
    missing_agent_ids: list[str] = field(default_factory=list)
    total_citations: int = 0


@dataclass(slots=True)
class ComposedAnswer:
    answer: str
    citations: list[str] = field(default_factory=list)
    citation_validation: CitationValidation = field(default_factory=CitationValidation)


ALLOWED_DECISIONS = {"go", "conditional_go", "no_go"}


@dataclass(slots=True)
class DecisionDraft:
    decision: str
    rationale: str
    next_steps: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.decision not in ALLOWED_DECISIONS:
            raise ValueError(f"decision must be one of {sorted(ALLOWED_DECISIONS)}")


@dataclass(slots=True)
class ExecutiveSynthesis:
    summary: str
    decision_draft: DecisionDraft
    citations: list[str] = field(default_factory=list)
    citation_validation: CitationValidation = field(default_factory=CitationValidation)
