from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.domain.models import JsonValue

SMILES_MAX_LENGTH = 4096
TARGET_MAX_LENGTH = 512
QUESTION_MAX_LENGTH = 4000
COMPOUND_NAME_MAX_LENGTH = 256


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class ConsultRequest(BaseModel):
    smiles: str = Field(min_length=1, max_length=SMILES_MAX_LENGTH)
    target: str = Field(min_length=1, max_length=TARGET_MAX_LENGTH)
    question: str = Field(min_length=1, max_length=QUESTION_MAX_LENGTH)
    compound_name: str | None = Field(default=None, max_length=COMPOUND_NAME_MAX_LENGTH)


class ExecutiveRequest(BaseModel):
    smiles: str = Field(min_length=1, max_length=SMILES_MAX_LENGTH)
    target: str = Field(min_length=1, max_length=TARGET_MAX_LENGTH)
    compound_name: str | None = Field(default=None, max_length=COMPOUND_NAME_MAX_LENGTH)


class PredictionSignalResponse(BaseModel):
    name: str
    value: str | float | int | bool
    unit: str | None = None
    confidence: float | None = None
    risk_level: str | None = None


class PredictionBundleResponse(BaseModel):
    source: str
    target: str | None = None
    compound_name: str | None = None
    canonical_smiles: str | None = None
    signals: list[PredictionSignalResponse] = Field(default_factory=list)
    missing_signals: list[str] = Field(default_factory=list)
    generated_at: str


class AgentFindingResponse(BaseModel):
    agent_id: str
    summary: str
    risks: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    citations: list[str] = Field(default_factory=list)


class EvidenceItemResponse(BaseModel):
    source: str
    pmid: str
    title: str
    abstract: str
    journal: str
    pub_year: int | None = None
    authors: list[str] = Field(default_factory=list)
    url: str = ""
    score: float = 0.0
    fetched_at: str
    missing_reason: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class EvidencePacketResponse(BaseModel):
    source: str
    query: str
    items: list[EvidenceItemResponse] = Field(default_factory=list)
    fetched_at: str
    source_health: str = "ok"
    missing_reason: str | None = None
    diagnostics: dict[str, str] = Field(default_factory=dict)


class EvidenceBundleResponse(BaseModel):
    query: str
    packets: dict[str, EvidencePacketResponse] = Field(default_factory=dict)
    items: list[EvidenceItemResponse] = Field(default_factory=list)
    fetched_at: str
    source_health: str = "ok"
    missing_sources: list[str] = Field(default_factory=list)
    partial_failures: list[str] = Field(default_factory=list)


class DecisionDraftResponse(BaseModel):
    decision: str
    rationale: str
    next_steps: list[str] = Field(default_factory=list)


class ConsultResponse(BaseModel):
    selected_agents: list[str]
    routing_reason: str
    predictions: PredictionBundleResponse
    agent_findings: list[AgentFindingResponse] = Field(default_factory=list)
    consulting_answer: str
    citations: list[str] = Field(default_factory=list)
    review_required: bool


class ExecutiveResponse(BaseModel):
    canonical_smiles: str | None = None
    molecule_svg: str | None = None
    predictions: PredictionBundleResponse
    evidence_bundle: EvidenceBundleResponse
    agent_findings: list[AgentFindingResponse] = Field(default_factory=list)
    executive_summary: str
    executive_decision: DecisionDraftResponse
    citations: list[str] = Field(default_factory=list)
    review_required: bool
    review_reasons: list[str] = Field(default_factory=list)
