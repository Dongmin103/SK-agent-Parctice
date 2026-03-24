from __future__ import annotations

from dataclasses import asdict

import pytest

from app.domain.models import (
    AgentFinding,
    CitationValidation,
    ComposedAnswer,
    CompoundContext,
    EvidenceBundle,
    EvidenceItem,
    EvidencePacket,
    DecisionDraft,
    ExecutiveSynthesis,
    PredictionBundle,
    PredictionSignal,
)


def test_compound_context_captures_preprocessed_compound_metadata() -> None:
    context = CompoundContext(
        smiles="N#CC1=CC=CC=C1",
        target="KRAS G12C",
        compound_name="ABC-101",
        canonical_smiles="N#Cc1ccccc1",
        molecule_svg="<svg />",
    )

    assert asdict(context) == {
        "smiles": "N#CC1=CC=CC=C1",
        "target": "KRAS G12C",
        "compound_name": "ABC-101",
        "canonical_smiles": "N#Cc1ccccc1",
        "molecule_svg": "<svg />",
    }


def test_prediction_bundle_keeps_context_fields_with_partial_predictions() -> None:
    bundle = PredictionBundle(
        target="KRAS G12C",
        compound_name="ABC-101",
        canonical_smiles="N#Cc1ccccc1",
        missing_signals=["bbb"],
        signals=[
            PredictionSignal(
                name="hERG",
                value="high",
                unit=None,
                confidence=0.82,
                risk_level="high",
            )
        ],
    )

    assert bundle.source == "txgemma"
    assert bundle.target == "KRAS G12C"
    assert bundle.compound_name == "ABC-101"
    assert bundle.canonical_smiles == "N#Cc1ccccc1"
    assert bundle.missing_signals == ["bbb"]
    assert bundle.signals[0].name == "hERG"


def test_agent_finding_remains_machine_readable_for_downstream_workflows() -> None:
    finding = AgentFinding(
        agent_id="house",
        summary="Predicted hERG liability aligns with literature evidence.",
        risks=["High hERG liability"],
        recommendations=["Run a confirmatory patch clamp assay."],
        confidence=0.79,
        citations=["https://pubmed.ncbi.nlm.nih.gov/12345/"],
    )

    assert asdict(finding) == {
        "agent_id": "house",
        "summary": "Predicted hERG liability aligns with literature evidence.",
        "risks": ["High hERG liability"],
        "recommendations": ["Run a confirmatory patch clamp assay."],
        "confidence": 0.79,
        "citations": ["https://pubmed.ncbi.nlm.nih.gov/12345/"],
    }


def test_citation_validation_tracks_missing_agents_and_unique_total() -> None:
    validation = CitationValidation(
        complete=False,
        missing_agent_ids=["walter"],
        total_citations=2,
    )

    assert asdict(validation) == {
        "complete": False,
        "missing_agent_ids": ["walter"],
        "total_citations": 2,
    }


def test_composed_answer_and_executive_synthesis_share_machine_readable_result_contracts() -> None:
    validation = CitationValidation(
        complete=True,
        missing_agent_ids=[],
        total_citations=2,
    )
    composed = ComposedAnswer(
        answer="질문에 대해 House 관점에서 검토했습니다. 추가 검증이 필요합니다.",
        citations=[
            "https://pubmed.ncbi.nlm.nih.gov/12345/",
            "https://clinicaltrials.gov/study/NCT00000000",
        ],
        citation_validation=validation,
    )
    synthesis = ExecutiveSynthesis(
        summary="안전성 후속 검증 전까지는 조건부 진행이 적절합니다.",
        decision_draft=DecisionDraft(
            decision="conditional_go",
            rationale="안전성 리스크 완화와 임상 근거 보강이 필요합니다.",
            next_steps=["Repeat hERG assay", "Confirm DDI mitigation plan"],
        ),
        citations=[
            "https://pubmed.ncbi.nlm.nih.gov/12345/",
            "https://clinicaltrials.gov/study/NCT00000000",
        ],
        citation_validation=validation,
    )

    assert asdict(composed) == {
        "answer": "질문에 대해 House 관점에서 검토했습니다. 추가 검증이 필요합니다.",
        "citations": [
            "https://pubmed.ncbi.nlm.nih.gov/12345/",
            "https://clinicaltrials.gov/study/NCT00000000",
        ],
        "citation_validation": {
            "complete": True,
            "missing_agent_ids": [],
            "total_citations": 2,
        },
    }
    assert asdict(synthesis) == {
        "summary": "안전성 후속 검증 전까지는 조건부 진행이 적절합니다.",
        "decision_draft": {
            "decision": "conditional_go",
            "rationale": "안전성 리스크 완화와 임상 근거 보강이 필요합니다.",
            "next_steps": ["Repeat hERG assay", "Confirm DDI mitigation plan"],
        },
        "citations": [
            "https://pubmed.ncbi.nlm.nih.gov/12345/",
            "https://clinicaltrials.gov/study/NCT00000000",
        ],
        "citation_validation": {
            "complete": True,
            "missing_agent_ids": [],
            "total_citations": 2,
        },
    }


def test_decision_draft_validates_supported_outcomes() -> None:
    draft = DecisionDraft(
        decision="conditional_go",
        rationale="Safety follow-up is still required before advancing.",
        next_steps=["Repeat hERG assay", "Confirm DDI mitigation plan"],
    )

    assert asdict(draft) == {
        "decision": "conditional_go",
        "rationale": "Safety follow-up is still required before advancing.",
        "next_steps": ["Repeat hERG assay", "Confirm DDI mitigation plan"],
    }

    with pytest.raises(ValueError, match="decision"):
        DecisionDraft(decision="maybe", rationale="Unsupported outcome")


def test_evidence_bundle_tracks_packets_missing_sources_and_partial_failures() -> None:
    pubchem_item = EvidenceItem(
        source="pubchem",
        pmid="2244",
        title="PubChem compound summary for Aspirin",
        abstract="Formula C9H8O4; XLogP 1.2",
        journal="PubChem",
        pub_year=None,
        url="https://pubchem.ncbi.nlm.nih.gov/compound/2244",
        score=5.5,
        metadata={"cid": "2244", "formula": "C9H8O4"},
    )
    pubchem_packet = EvidencePacket(
        source="pubchem",
        query="Aspirin",
        items=[pubchem_item],
        source_health="ok",
    )
    clinical_packet = EvidencePacket(
        source="clinicaltrials",
        query="Aspirin",
        items=[],
        source_health="degraded",
        missing_reason="clinicaltrials_request_failed",
        diagnostics={"error": "HTTP 503"},
    )
    bundle = EvidenceBundle(
        query="Aspirin evidence",
        packets={
            "pubchem": pubchem_packet,
            "clinicaltrials": clinical_packet,
        },
        items=[pubchem_item],
        source_health="partial",
        missing_sources=["clinicaltrials"],
        partial_failures=["clinicaltrials"],
    )

    assert asdict(bundle) == {
        "query": "Aspirin evidence",
        "packets": {
            "pubchem": {
                "source": "pubchem",
                "query": "Aspirin",
                "items": [
                    {
                        "source": "pubchem",
                        "pmid": "2244",
                        "title": "PubChem compound summary for Aspirin",
                        "abstract": "Formula C9H8O4; XLogP 1.2",
                        "journal": "PubChem",
                        "pub_year": None,
                        "authors": [],
                        "url": "https://pubchem.ncbi.nlm.nih.gov/compound/2244",
                        "score": 5.5,
                        "fetched_at": pubchem_item.fetched_at,
                        "missing_reason": None,
                        "metadata": {"cid": "2244", "formula": "C9H8O4"},
                    }
                ],
                "fetched_at": pubchem_packet.fetched_at,
                "source_health": "ok",
                "missing_reason": None,
                "diagnostics": {},
            },
            "clinicaltrials": {
                "source": "clinicaltrials",
                "query": "Aspirin",
                "items": [],
                "fetched_at": clinical_packet.fetched_at,
                "source_health": "degraded",
                "missing_reason": "clinicaltrials_request_failed",
                "diagnostics": {"error": "HTTP 503"},
            },
        },
        "items": [
            {
                "source": "pubchem",
                "pmid": "2244",
                "title": "PubChem compound summary for Aspirin",
                "abstract": "Formula C9H8O4; XLogP 1.2",
                "journal": "PubChem",
                "pub_year": None,
                "authors": [],
                "url": "https://pubchem.ncbi.nlm.nih.gov/compound/2244",
                "score": 5.5,
                "fetched_at": pubchem_item.fetched_at,
                "missing_reason": None,
                "metadata": {"cid": "2244", "formula": "C9H8O4"},
            }
        ],
        "fetched_at": bundle.fetched_at,
        "source_health": "partial",
        "missing_sources": ["clinicaltrials"],
        "partial_failures": ["clinicaltrials"],
    }
