from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import create_app
from app.api.errors import AppServiceError
from app.domain.models import (
    AgentFinding,
    DecisionDraft,
    EvidenceBundle,
    EvidenceItem,
    EvidencePacket,
    PredictionBundle,
    PredictionSignal,
)
from app.workflows.executive import ExecutiveReport


class StubExecutiveWorkflow:
    def __init__(self, report: ExecutiveReport) -> None:
        self.report = report
        self.calls: list[dict[str, object]] = []

    def run(self, *, smiles: str, target: str, compound_name: str | None = None) -> ExecutiveReport:
        self.calls.append(
            {
                "smiles": smiles,
                "target": target,
                "compound_name": compound_name,
            }
        )
        return self.report


class RaisingExecutiveWorkflow:
    def run(self, *, smiles: str, target: str, compound_name: str | None = None) -> ExecutiveReport:
        raise AppServiceError(
            code="upstream_unavailable",
            message="Executive workflow dependency was unavailable.",
            status_code=503,
            details={"dependency": "txgemma"},
        )


class GuardWorkflow:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run(self, *, smiles: str, target: str, compound_name: str | None = None) -> ExecutiveReport:
        self.calls.append(
            {
                "smiles": smiles,
                "target": target,
                "compound_name": compound_name,
            }
        )
        raise AssertionError("workflow should not be called for invalid requests")


def make_report() -> ExecutiveReport:
    chemistry_packet = EvidencePacket(
        source="chemistry",
        query="chemistry query",
        items=[
            EvidenceItem(
                source="chemistry",
                pmid="chemistry",
                title="Chemistry evidence",
                abstract="Chemistry evidence abstract",
                journal="Chem Journal",
                pub_year=2025,
                url="https://example.org/chemistry",
                score=8.0,
            )
        ],
    )
    prediction_bundle = PredictionBundle(
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
    return ExecutiveReport(
        predictions=prediction_bundle,
        evidence_bundle=EvidenceBundle(
            query="executive assessment query",
            packets={"chemistry": chemistry_packet},
            items=list(chemistry_packet.items),
            source_health="ok",
        ),
        agent_findings=[
            AgentFinding(
                agent_id="walter",
                summary="Walter summary",
                citations=["https://example.org/chemistry"],
            )
        ],
        executive_summary="CEO summary",
        executive_decision=DecisionDraft(
            decision="conditional_go",
            rationale="More evidence is still required.",
            next_steps=["Repeat hERG assay"],
        ),
        citations=["https://example.org/chemistry"],
        review_required=True,
        review_reasons=["missing cross-domain review"],
    )


def test_post_executive_report_returns_machine_readable_response() -> None:
    workflow = StubExecutiveWorkflow(make_report())
    client = TestClient(create_app(consult_workflow=object(), executive_workflow=workflow))

    response = client.post(
        "/api/reports/executive",
        json={
            "smiles": "CCO",
            "target": "KRAS G12C",
            "compound_name": "ABC-101",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predictions"]["canonical_smiles"] == "CCO"
    assert payload["evidence_bundle"]["source_health"] == "ok"
    assert payload["agent_findings"] == [
        {
            "agent_id": "walter",
            "summary": "Walter summary",
            "risks": [],
            "recommendations": [],
            "confidence": 0.0,
            "citations": ["https://example.org/chemistry"],
        }
    ]
    assert payload["executive_summary"] == "CEO summary"
    assert payload["executive_decision"] == {
        "decision": "conditional_go",
        "rationale": "More evidence is still required.",
        "next_steps": ["Repeat hERG assay"],
    }
    assert payload["citations"] == ["https://example.org/chemistry"]
    assert payload["review_required"] is True
    assert payload["review_reasons"] == ["missing cross-domain review"]
    assert workflow.calls == [
        {
            "smiles": "CCO",
            "target": "KRAS G12C",
            "compound_name": "ABC-101",
        }
    ]


def test_post_executive_report_allows_list_metadata_from_live_evidence() -> None:
    clinical_packet = EvidencePacket(
        source="clinicaltrials",
        query="clinicaltrials query",
        items=[
            EvidenceItem(
                source="clinicaltrials",
                pmid="NCT01234567",
                title="Trial evidence",
                abstract="Clinical trial evidence abstract",
                journal="ClinicalTrials.gov",
                pub_year=2025,
                url="https://clinicaltrials.gov/study/NCT01234567",
                score=7.5,
                metadata={
                    "status": "RECRUITING",
                    "conditions": ["KRAS G12C", "Solid Tumor"],
                    "phase": ["PHASE2"],
                },
            )
        ],
    )
    report = ExecutiveReport(
        predictions=PredictionBundle(
            source="txgemma",
            target="KRAS G12C",
            compound_name="ABC-101",
            canonical_smiles="CCO",
            generated_at="2026-03-24T12:00:00+00:00",
            signals=[],
        ),
        evidence_bundle=EvidenceBundle(
            query="executive assessment query",
            packets={"clinicaltrials": clinical_packet},
            items=list(clinical_packet.items),
            source_health="ok",
        ),
        agent_findings=[
            AgentFinding(
                agent_id="walter",
                summary="Walter summary",
                citations=["https://clinicaltrials.gov/study/NCT01234567"],
            )
        ],
        executive_summary="CEO summary",
        executive_decision=DecisionDraft(
            decision="conditional_go",
            rationale="More evidence is still required.",
            next_steps=["Review clinical inclusion criteria"],
        ),
        citations=["https://clinicaltrials.gov/study/NCT01234567"],
        review_required=True,
        review_reasons=["missing cross-domain review"],
    )
    workflow = StubExecutiveWorkflow(report)
    client = TestClient(create_app(consult_workflow=object(), executive_workflow=workflow))

    response = client.post(
        "/api/reports/executive",
        json={
            "smiles": "CCO",
            "target": "KRAS G12C",
            "compound_name": "ABC-101",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    metadata = payload["evidence_bundle"]["items"][0]["metadata"]
    assert metadata == {
        "status": "RECRUITING",
        "conditions": ["KRAS G12C", "Solid Tumor"],
        "phase": ["PHASE2"],
    }


def test_post_executive_report_rejects_invalid_request_body_before_workflow_runs() -> None:
    workflow = GuardWorkflow()
    client = TestClient(create_app(consult_workflow=object(), executive_workflow=workflow))

    response = client.post(
        "/api/reports/executive",
        json={
            "smiles": "CCO",
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "request_validation_error"
    assert workflow.calls == []


def test_post_executive_report_returns_typed_error_response_for_application_failures() -> None:
    client = TestClient(create_app(consult_workflow=object(), executive_workflow=RaisingExecutiveWorkflow()))

    response = client.post(
        "/api/reports/executive",
        json={
            "smiles": "CCO",
            "target": "KRAS G12C",
            "compound_name": "ABC-101",
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "upstream_unavailable",
            "message": "Executive workflow dependency was unavailable.",
            "details": {"dependency": "txgemma"},
        }
    }


def test_post_executive_report_rejects_oversized_request_fields_before_workflow_runs() -> None:
    workflow = GuardWorkflow()
    client = TestClient(create_app(consult_workflow=object(), executive_workflow=workflow))

    response = client.post(
        "/api/reports/executive",
        json={
            "smiles": "C" * 4097,
            "target": "T" * 513,
            "compound_name": "A" * 257,
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "request_validation_error"
    locations = {tuple(item["loc"]) for item in payload["error"]["details"]}
    assert ("body", "smiles") in locations
    assert ("body", "target") in locations
    assert ("body", "compound_name") in locations
    assert workflow.calls == []
