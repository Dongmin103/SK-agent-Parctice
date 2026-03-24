from __future__ import annotations

import pytest

from app.domain.models import (
    AgentFinding,
    EvidenceBundle,
    EvidenceItem,
    EvidencePacket,
    PredictionBundle,
    PredictionSignal,
)
from app.domain.compound import InvalidSmilesError
from app.workflows.executive import ExecutiveWorkflow


class StubTxGemmaClient:
    def __init__(self, bundle: PredictionBundle) -> None:
        self.bundle = bundle
        self.calls: list[dict[str, object]] = []

    def predict(
        self,
        *,
        smiles: str,
        target: str | None = None,
        compound_name: str | None = None,
    ) -> PredictionBundle:
        self.calls.append(
            {
                "smiles": smiles,
                "target": target,
                "compound_name": compound_name,
            }
        )
        return self.bundle


class StubEvidenceCoordinator:
    def __init__(self, bundle: EvidenceBundle, domain_packets: dict[str, EvidencePacket]) -> None:
        self.bundle = bundle
        self.domain_packets = domain_packets
        self.collect_calls: list[dict[str, object]] = []
        self.packet_calls: list[EvidenceBundle] = []

    def collect_evidence(
        self,
        *,
        question: str,
        question_type: str,
        target: str | None = None,
        compound_name: str | None = None,
        retmax: int = 10,
        top_k: int = 5,
    ) -> EvidenceBundle:
        self.collect_calls.append(
            {
                "question": question,
                "question_type": question_type,
                "target": target,
                "compound_name": compound_name,
                "retmax": retmax,
                "top_k": top_k,
            }
        )
        return self.bundle

    def build_domain_packets(self, bundle: EvidenceBundle) -> dict[str, EvidencePacket]:
        self.packet_calls.append(bundle)
        return self.domain_packets


class StubExpertAgent:
    def __init__(self, finding: AgentFinding) -> None:
        self.finding = finding
        self.calls: list[dict[str, object]] = []

    def analyze(
        self,
        question: str,
        *,
        target: str | None = None,
        compound_name: str | None = None,
        evidence_packet: EvidencePacket | None = None,
        prediction_bundle: PredictionBundle | None = None,
    ) -> AgentFinding:
        self.calls.append(
            {
                "question": question,
                "target": target,
                "compound_name": compound_name,
                "evidence_packet": evidence_packet,
                "prediction_bundle": prediction_bundle,
            }
        )
        return self.finding


def make_prediction_bundle(canonical_smiles: str = "CCO") -> PredictionBundle:
    return PredictionBundle(
        source="txgemma",
        target="KRAS G12C",
        compound_name="ABC-101",
        canonical_smiles=canonical_smiles,
        generated_at="2026-03-24T12:00:00+00:00",
        signals=[
            PredictionSignal(
                name="hERG",
                value="elevated",
                confidence=0.82,
                risk_level="high",
            )
        ],
    )


def make_packet(source: str, pmid: str, title: str) -> EvidencePacket:
    return EvidencePacket(
        source=source,
        query=f"{source} query",
        items=[
            EvidenceItem(
                source=source,
                pmid=pmid,
                title=title,
                abstract=f"{title} abstract",
                journal="Test Journal",
                pub_year=2025,
                url=f"https://example.org/{pmid}",
                score=8.2,
            )
        ],
    )


def make_bundle(*packets: EvidencePacket) -> EvidenceBundle:
    items = []
    for packet in packets:
        items.extend(packet.items)
    return EvidenceBundle(
        query="executive assessment query",
        packets={packet.source: packet for packet in packets},
        items=items,
        source_health="ok",
    )


def test_executive_workflow_runs_all_experts_and_synthesizes_decision() -> None:
    chemistry_packet = make_packet("chemistry", "chemistry", "Walter evidence")
    safety_packet = make_packet("safety", "safety", "House evidence")
    clinical_packet = make_packet("clinical_regulatory", "clinical", "Harvey evidence")
    canonical_smiles = "N#Cc1ccccc1"
    workflow = ExecutiveWorkflow(
        prediction_client=StubTxGemmaClient(make_prediction_bundle(canonical_smiles)),
        evidence_coordinator=StubEvidenceCoordinator(
            make_bundle(chemistry_packet, safety_packet, clinical_packet),
            {
                "chemistry": chemistry_packet,
                "safety": safety_packet,
                "clinical_regulatory": clinical_packet,
            },
        ),
        walter_agent=StubExpertAgent(
            AgentFinding(
                agent_id="walter",
                summary="Walter summary",
                citations=[chemistry_packet.items[0].url],
            )
        ),
        house_agent=StubExpertAgent(
            AgentFinding(
                agent_id="house",
                summary="House summary",
                recommendations=["Repeat hERG assay"],
                citations=[safety_packet.items[0].url],
            )
        ),
        harvey_agent=StubExpertAgent(
            AgentFinding(
                agent_id="harvey",
                summary="Harvey summary",
                recommendations=["Confirm regulatory path"],
                citations=[clinical_packet.items[0].url],
            )
        ),
    )

    report = workflow.run(
        smiles="N#CC1=CC=CC=C1",
        target="KRAS G12C",
        compound_name="ABC-101",
    )

    assert [finding.agent_id for finding in report.agent_findings] == ["walter", "house", "harvey"]
    assert report.canonical_smiles == canonical_smiles
    assert report.molecule_svg is not None
    assert "<svg" in report.molecule_svg
    assert report.executive_decision.decision == "conditional_go"
    assert report.executive_decision.next_steps == ["Repeat hERG assay", "Confirm regulatory path"]
    assert report.citations == [
        "https://example.org/chemistry",
        "https://example.org/safety",
        "https://example.org/clinical",
    ]
    assert report.review_required is True
    assert "사람 검토가 필요합니다." in " ".join(report.review_reasons)

    assert workflow.prediction_client.calls == [
        {
            "smiles": canonical_smiles,
            "target": "KRAS G12C",
            "compound_name": "ABC-101",
        }
    ]
    assert workflow.evidence_coordinator.collect_calls == [
        {
            "question": "Executive assessment for ABC-101 against KRAS G12C",
            "question_type": "multi_expert",
            "target": "KRAS G12C",
            "compound_name": "ABC-101",
            "retmax": 10,
            "top_k": 5,
        }
    ]
    assert workflow.walter_agent.calls[0]["evidence_packet"] == chemistry_packet
    assert workflow.house_agent.calls[0]["evidence_packet"] == safety_packet
    assert workflow.harvey_agent.calls[0]["evidence_packet"] == clinical_packet


def test_executive_workflow_rejects_invalid_smiles_before_downstream_calls() -> None:
    chemistry_packet = make_packet("chemistry", "chemistry", "Walter evidence")
    safety_packet = make_packet("safety", "safety", "House evidence")
    clinical_packet = make_packet("clinical_regulatory", "clinical", "Harvey evidence")
    workflow = ExecutiveWorkflow(
        prediction_client=StubTxGemmaClient(make_prediction_bundle()),
        evidence_coordinator=StubEvidenceCoordinator(
            make_bundle(chemistry_packet, safety_packet, clinical_packet),
            {
                "chemistry": chemistry_packet,
                "safety": safety_packet,
                "clinical_regulatory": clinical_packet,
            },
        ),
        walter_agent=StubExpertAgent(AgentFinding(agent_id="walter", summary="Walter summary")),
        house_agent=StubExpertAgent(AgentFinding(agent_id="house", summary="House summary")),
        harvey_agent=StubExpertAgent(AgentFinding(agent_id="harvey", summary="Harvey summary")),
    )

    with pytest.raises(InvalidSmilesError, match="Invalid SMILES"):
        workflow.run(
            smiles="not-a-smiles",
            target="KRAS G12C",
            compound_name="ABC-101",
        )

    assert workflow.prediction_client.calls == []
    assert workflow.evidence_coordinator.collect_calls == []
    assert workflow.walter_agent.calls == []
    assert workflow.house_agent.calls == []
    assert workflow.harvey_agent.calls == []
