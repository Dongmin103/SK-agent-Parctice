from __future__ import annotations

from app.domain.models import AgentFinding, EvidenceItem, EvidencePacket, PredictionBundle, PredictionSignal
from app.agents.harvey_agent import HarveyAgent


class SequenceRunnable:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        if not self._responses:
            raise RuntimeError("No more stubbed responses")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_evidence_packet() -> EvidencePacket:
    return EvidencePacket(
        source="clinical_regulatory",
        query='"KRAS G12C" AND ("phase 2" OR approval OR FDA)',
        items=[
            EvidenceItem(
                source="clinicaltrials",
                pmid="NCT01234567",
                title="Phase 2 KRAS G12C program shows early activity with execution risk",
                abstract="Ongoing phase 2 enrollment supports continued clinical development, but durability and cohort expansion remain open questions.",
                journal="ClinicalTrials.gov",
                pub_year=2025,
                authors=["ClinicalTrials.gov"],
                url="https://clinicaltrials.gov/study/NCT01234567",
            ),
            EvidenceItem(
                source="openfda",
                pmid="FDA-2025-101",
                title="FDA briefing materials highlight approval hurdles and competitive differentiation needs",
                abstract="The approval path depends on confirmatory evidence while competitive pressure may reduce program priority.",
                journal="openFDA",
                pub_year=2025,
                authors=["FDA"],
                url="https://open.fda.gov/apis/drug/event/example-101",
            ),
        ],
        source_health="ok",
    )


def make_prediction_bundle() -> PredictionBundle:
    return PredictionBundle(
        signals=[
            PredictionSignal(
                name="approval risk",
                value="elevated regulatory uncertainty",
                confidence=0.84,
                risk_level="high",
            ),
            PredictionSignal(
                name="development priority",
                value="moderate competitive pressure",
                confidence=0.76,
                risk_level="elevated",
            ),
        ]
    )


def test_harvey_agent_accepts_structured_output_from_runnable() -> None:
    evidence_packet = make_evidence_packet()
    prediction_bundle = make_prediction_bundle()
    runnable = SequenceRunnable(
        [
            {
                "summary": "Harvey sees a viable but tightly constrained approval path.",
                "risks": ["Confirmatory evidence expectations remain high."],
                "recommendations": ["Clarify the approval strategy before expanding investment."],
                "confidence": 0.79,
                "citations": [evidence_packet.items[0].url, "https://example.com/not-allowed"],
            }
        ]
    )
    agent = HarveyAgent(analyzer_runnable=runnable)

    result = agent.analyze(
        question="승인 가능성과 개발 우선순위는?",
        target="KRAS G12C",
        evidence_packet=evidence_packet,
        prediction_bundle=prediction_bundle,
    )

    assert isinstance(result, AgentFinding)
    assert result.agent_id == "harvey"
    assert result.summary == "Harvey sees a viable but tightly constrained approval path."
    assert result.risks == ["Confirmatory evidence expectations remain high."]
    assert result.recommendations == ["Clarify the approval strategy before expanding investment."]
    assert result.confidence == 0.79
    assert result.citations == [evidence_packet.items[0].url]
    assert runnable.calls[0]["question"] == "승인 가능성과 개발 우선순위는?"
    assert runnable.calls[0]["prediction_bundle"]["signals"][0]["name"] == "approval risk"
    assert runnable.calls[0]["evidence_items"][0]["pmid"] == "NCT01234567"


def test_harvey_agent_falls_back_when_runnable_output_is_invalid() -> None:
    evidence_packet = make_evidence_packet()
    prediction_bundle = make_prediction_bundle()
    runnable = SequenceRunnable(
        [
            {
                "summary": "This should not survive validation.",
                "risks": "not-a-list",
                "recommendations": [],
                "confidence": 0.88,
                "citations": [],
            }
        ]
    )
    agent = HarveyAgent(analyzer_runnable=runnable)

    result = agent.analyze(
        question="승인 가능성과 개발 우선순위는?",
        target="KRAS G12C",
        evidence_packet=evidence_packet,
        prediction_bundle=prediction_bundle,
    )

    assert result.agent_id == "harvey"
    assert result.summary != "This should not survive validation."
    assert any("clinical" in risk.lower() or "development" in risk.lower() for risk in result.risks)
    assert any("approval" in risk.lower() or "regulatory" in risk.lower() for risk in result.risks)
    assert any("strategy" in recommendation.lower() or "different" in recommendation.lower() for recommendation in result.recommendations)
    assert result.citations == [item.url for item in evidence_packet.items]


def test_harvey_agent_returns_low_confidence_fallback_when_inputs_are_sparse() -> None:
    evidence_packet = EvidencePacket(
        source="clinical_regulatory",
        query="",
        items=[],
        source_health="degraded",
        missing_reason="no_clinical_regulatory_hits",
    )
    agent = HarveyAgent()

    result = agent.analyze(
        question="승인 가능성은?",
        evidence_packet=evidence_packet,
        prediction_bundle=None,
    )

    assert result.agent_id == "harvey"
    assert result.summary == "Harvey found limited clinical/regulatory evidence for a stronger conclusion."
    assert result.risks == ["Insufficient clinical, regulatory, or strategy signals collected yet."]
    assert result.recommendations == [
        "Collect additional ClinicalTrials.gov/openFDA evidence and program strategy inputs."
    ]
    assert result.citations == []
    assert result.confidence < 0.5


def test_harvey_agent_fallback_orders_categories_deterministically() -> None:
    evidence_packet = make_evidence_packet()
    prediction_bundle = make_prediction_bundle()
    agent = HarveyAgent()

    result = agent.analyze(
        question="승인 가능성과 개발 우선순위는?",
        target="KRAS G12C",
        evidence_packet=evidence_packet,
        prediction_bundle=prediction_bundle,
    )

    assert result.summary == (
        "Harvey prioritized clinical development, regulatory approval, competitive strategy "
        "signals from evidence and predictions."
    )
    assert result.risks == [
        "Clinical development evidence still leaves execution risk around trial design or durability.",
        "Regulatory approval hurdles remain open and need explicit evidence.",
        "Competitive strategy pressure could weaken program prioritization.",
    ]
    assert result.recommendations == [
        "Review ongoing trial design, endpoints, and enrollment feasibility.",
        "Map the approval pathway with confirmatory evidence requirements.",
        "Clarify differentiation versus competitors before major investment.",
    ]
    assert result.citations == [item.url for item in evidence_packet.items]
