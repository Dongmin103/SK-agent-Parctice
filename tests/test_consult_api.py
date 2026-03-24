from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.agents.router_agent import RoutingDecision
from app.api import create_app
from app.api.errors import AppServiceError
from app.domain.models import (
    AgentFinding,
    EvidenceBundle,
    EvidenceItem,
    EvidencePacket,
    PredictionBundle,
    PredictionSignal,
)
from app.workflows.consult import ConsultReport, ConsultWorkflow
from app.workflows.tracing import WorkflowTraceEvent


class StubTxGemmaClient:
    def __init__(self, bundle: PredictionBundle) -> None:
        self.bundle = bundle
        self.calls: list[dict[str, object]] = []

    def predict(self, *, smiles: str, target: str | None = None, compound_name: str | None = None) -> PredictionBundle:
        self.calls.append(
            {
                "smiles": smiles,
                "target": target,
                "compound_name": compound_name,
            }
        )
        return self.bundle


class StubRouterAgent:
    def __init__(self, decision: RoutingDecision) -> None:
        self.decision = decision
        self.calls: list[dict[str, object]] = []

    def route(
        self,
        question: str,
        *,
        target: str | None = None,
        compound_name: str | None = None,
        prediction_bundle: PredictionBundle | None = None,
    ) -> RoutingDecision:
        self.calls.append(
            {
                "question": question,
                "target": target,
                "compound_name": compound_name,
                "prediction_bundle": prediction_bundle,
            }
        )
        return self.decision


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


class GuardWorkflow:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run(self, *, smiles: str, target: str, question: str, compound_name: str | None = None) -> object:
        self.calls.append(
            {
                "smiles": smiles,
                "target": target,
                "question": question,
                "compound_name": compound_name,
            }
        )
        raise AssertionError("workflow should not be called for invalid requests")


class StubStreamingConsultWorkflow:
    def __init__(self, report: ConsultReport) -> None:
        self.report = report
        self.calls: list[dict[str, object]] = []

    def run(
        self,
        *,
        smiles: str,
        target: str,
        question: str,
        compound_name: str | None = None,
        event_sink=None,
    ) -> ConsultReport:
        self.calls.append(
            {
                "smiles": smiles,
                "target": target,
                "question": question,
                "compound_name": compound_name,
                "event_sink": event_sink,
            }
        )
        assert event_sink is not None
        event_sink(
            WorkflowTraceEvent(
                stage="routing",
                message="Selected agents: house, harvey",
            )
        )
        event_sink(
            WorkflowTraceEvent(
                stage="evidence",
                message="PubMed dry run hits: 12",
            )
        )
        return self.report


class RaisingStreamingConsultWorkflow:
    def run(
        self,
        *,
        smiles: str,
        target: str,
        question: str,
        compound_name: str | None = None,
        event_sink=None,
    ) -> ConsultReport:
        del smiles, target, question, compound_name, event_sink
        raise AppServiceError(
            code="upstream_unavailable",
            message="Consult workflow dependency was unavailable.",
            status_code=503,
            details={"dependency": "bedrock"},
        )


def make_prediction_bundle() -> PredictionBundle:
    return PredictionBundle(
        source="txgemma",
        target="KRAS G12C",
        compound_name="ABC-101",
        canonical_smiles="CCO",
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


def make_safety_packet() -> EvidencePacket:
    return EvidencePacket(
        source="safety",
        query='"ABC-101" AND hERG',
        source_health="ok",
        items=[
            EvidenceItem(
                source="pubmed",
                pmid="12345",
                title="ABC-101 hERG risk evidence",
                abstract="PubMed evidence supports follow-up QT review.",
                journal="Drug Safety",
                pub_year=2025,
                url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                score=8.5,
            )
        ],
    )


def make_bundle(packet: EvidencePacket) -> EvidenceBundle:
    return EvidenceBundle(
        query="이 화합물의 hERG 위험은?",
        packets={"pubmed": packet},
        items=list(packet.items),
        source_health="ok",
    )


def make_consult_report() -> ConsultReport:
    prediction_bundle = make_prediction_bundle()
    safety_packet = make_safety_packet()
    return ConsultReport(
        selected_agents=["house", "harvey"],
        routing_reason="Safety and clinical evidence are both needed.",
        predictions=prediction_bundle,
        agent_findings=[
            AgentFinding(
                agent_id="house",
                summary="House summary",
                citations=[safety_packet.items[0].url],
            )
        ],
        consulting_answer="Combined answer",
        citations=[safety_packet.items[0].url],
        review_required=True,
    )


def test_post_consult_report_wires_existing_agents_and_returns_response() -> None:
    prediction_bundle = make_prediction_bundle()
    safety_packet = make_safety_packet()
    evidence_bundle = make_bundle(safety_packet)
    house_finding = AgentFinding(
        agent_id="house",
        summary="House sees a material hERG liability that needs follow-up.",
        risks=["Elevated hERG risk may block progression."],
        recommendations=["Run a confirmatory patch clamp assay."],
        confidence=0.81,
        citations=[safety_packet.items[0].url],
    )

    workflow = ConsultWorkflow(
        prediction_client=StubTxGemmaClient(prediction_bundle),
        router_agent=StubRouterAgent(
            RoutingDecision(
                question_type="safety_pk",
                selected_agents=["house"],
                routing_reason="The question focuses on hERG safety risk.",
                confidence=0.84,
                fallback_used=False,
            )
        ),
        evidence_coordinator=StubEvidenceCoordinator(
            evidence_bundle,
            {
                "safety": safety_packet,
                "chemistry": EvidencePacket(source="chemistry", query="", items=[]),
                "clinical_regulatory": EvidencePacket(source="clinical_regulatory", query="", items=[]),
            },
        ),
        walter_agent=StubExpertAgent(
            AgentFinding(
                agent_id="walter",
                summary="unused",
                citations=[],
            )
        ),
        house_agent=StubExpertAgent(house_finding),
        harvey_agent=StubExpertAgent(
            AgentFinding(
                agent_id="harvey",
                summary="unused",
                citations=[],
            )
        ),
    )
    client = TestClient(create_app(consult_workflow=workflow))

    response = client.post(
        "/api/reports/consult",
        json={
            "smiles": "CCO",
            "target": "KRAS G12C",
            "question": "이 화합물의 hERG 위험은?",
            "compound_name": "ABC-101",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_agents"] == ["house"]
    assert payload["routing_reason"] == "The question focuses on hERG safety risk."
    assert payload["predictions"]["canonical_smiles"] == "CCO"
    assert payload["predictions"]["signals"] == [
        {
            "name": "hERG",
            "value": "elevated",
            "unit": None,
            "confidence": 0.82,
            "risk_level": "high",
        }
    ]
    assert payload["agent_findings"] == [
        {
            "agent_id": "house",
            "summary": "House sees a material hERG liability that needs follow-up.",
            "risks": ["Elevated hERG risk may block progression."],
            "recommendations": ["Run a confirmatory patch clamp assay."],
            "confidence": 0.81,
            "citations": ["https://pubmed.ncbi.nlm.nih.gov/12345/"],
        }
    ]
    assert "House sees a material hERG liability" in payload["consulting_answer"]
    assert payload["citations"] == ["https://pubmed.ncbi.nlm.nih.gov/12345/"]
    assert payload["review_required"] is True

    assert workflow.prediction_client.calls == [
        {
            "smiles": "CCO",
            "target": "KRAS G12C",
            "compound_name": "ABC-101",
        }
    ]
    assert workflow.router_agent.calls[0]["question"] == "이 화합물의 hERG 위험은?"
    assert workflow.router_agent.calls[0]["prediction_bundle"] is prediction_bundle
    assert workflow.evidence_coordinator.collect_calls == [
        {
            "question": "이 화합물의 hERG 위험은?",
            "question_type": "safety_pk",
            "target": "KRAS G12C",
            "compound_name": "ABC-101",
            "retmax": 10,
            "top_k": 5,
        }
    ]
    assert workflow.house_agent.calls[0]["evidence_packet"] == safety_packet
    assert workflow.walter_agent.calls == []
    assert workflow.harvey_agent.calls == []


def test_post_consult_report_rejects_invalid_request_body_before_workflow_runs() -> None:
    workflow = GuardWorkflow()
    client = TestClient(create_app(consult_workflow=workflow))

    response = client.post(
        "/api/reports/consult",
        json={
            "smiles": "CCO",
            "target": "KRAS G12C",
        },
    )

    assert response.status_code == 422
    assert workflow.calls == []


def test_post_consult_report_returns_typed_error_for_invalid_smiles_before_downstream_calls() -> None:
    prediction_bundle = make_prediction_bundle()
    safety_packet = make_safety_packet()
    evidence_bundle = make_bundle(safety_packet)
    workflow = ConsultWorkflow(
        prediction_client=StubTxGemmaClient(prediction_bundle),
        router_agent=StubRouterAgent(
            RoutingDecision(
                question_type="safety_pk",
                selected_agents=["house"],
                routing_reason="The question focuses on hERG safety risk.",
                confidence=0.84,
                fallback_used=False,
            )
        ),
        evidence_coordinator=StubEvidenceCoordinator(
            evidence_bundle,
            {
                "safety": safety_packet,
                "chemistry": EvidencePacket(source="chemistry", query="", items=[]),
                "clinical_regulatory": EvidencePacket(source="clinical_regulatory", query="", items=[]),
            },
        ),
        walter_agent=StubExpertAgent(AgentFinding(agent_id="walter", summary="unused")),
        house_agent=StubExpertAgent(AgentFinding(agent_id="house", summary="unused")),
        harvey_agent=StubExpertAgent(AgentFinding(agent_id="harvey", summary="unused")),
    )
    client = TestClient(create_app(consult_workflow=workflow))

    response = client.post(
        "/api/reports/consult",
        json={
            "smiles": "not-a-smiles",
            "target": "KRAS G12C",
            "question": "이 화합물의 hERG 위험은?",
            "compound_name": "ABC-101",
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "invalid_smiles",
            "message": "Invalid SMILES: not-a-smiles",
        }
    }
    assert workflow.prediction_client.calls == []
    assert workflow.router_agent.calls == []
    assert workflow.evidence_coordinator.collect_calls == []
    assert workflow.house_agent.calls == []
    assert workflow.walter_agent.calls == []
    assert workflow.harvey_agent.calls == []


def test_post_consult_report_rejects_oversized_request_fields_before_workflow_runs() -> None:
    workflow = GuardWorkflow()
    client = TestClient(create_app(consult_workflow=workflow))

    response = client.post(
        "/api/reports/consult",
        json={
            "smiles": "C" * 4097,
            "target": "T" * 513,
            "question": "Q" * 4001,
            "compound_name": "A" * 257,
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "request_validation_error"
    locations = {tuple(item["loc"]) for item in payload["error"]["details"]}
    assert ("body", "smiles") in locations
    assert ("body", "target") in locations
    assert ("body", "question") in locations
    assert ("body", "compound_name") in locations
    assert workflow.calls == []


def test_post_consult_stream_returns_trace_and_result_chunks() -> None:
    workflow = StubStreamingConsultWorkflow(make_consult_report())
    client = TestClient(create_app(consult_workflow=workflow, executive_workflow=object()))

    with client.stream(
        "POST",
        "/api/reports/consult/stream",
        json={
            "smiles": "CCO",
            "target": "KRAS G12C",
            "question": "이 화합물의 hERG 위험은?",
            "compound_name": "ABC-101",
        },
    ) as response:
        chunks = [json.loads(line) for line in response.iter_lines() if line]

    assert response.status_code == 200
    assert [chunk["type"] for chunk in chunks] == ["trace", "trace", "result"]
    assert chunks[0]["trace"]["message"] == "Selected agents: house, harvey"
    assert chunks[1]["trace"]["message"] == "PubMed dry run hits: 12"
    assert chunks[2]["result"]["selected_agents"] == ["house", "harvey"]
    assert workflow.calls[0]["event_sink"] is not None


def test_post_consult_stream_returns_error_chunk_for_application_failures() -> None:
    client = TestClient(
        create_app(
            consult_workflow=RaisingStreamingConsultWorkflow(),
            executive_workflow=object(),
        )
    )

    with client.stream(
        "POST",
        "/api/reports/consult/stream",
        json={
            "smiles": "CCO",
            "target": "KRAS G12C",
            "question": "이 화합물의 hERG 위험은?",
            "compound_name": "ABC-101",
        },
    ) as response:
        chunks = [json.loads(line) for line in response.iter_lines() if line]

    assert response.status_code == 200
    assert chunks == [
        {
            "type": "error",
            "error": {
                "code": "upstream_unavailable",
                "message": "Consult workflow dependency was unavailable.",
                "details": {"dependency": "bedrock"},
            },
        }
    ]
