from __future__ import annotations

from app.domain.models import AgentFinding, EvidenceItem, EvidencePacket, PredictionBundle, PredictionSignal
from app.agents.house_agent import HouseAgent


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
        source="pubmed",
        query='"KRAS G12C" AND ("hERG" OR "CYP")',
        items=[
            EvidenceItem(
                source="pubmed",
                pmid="12345",
                title="KRAS G12C inhibitors and cardiotoxicity risk",
                abstract="hERG and QT prolongation signals were reported with CYP-mediated interaction concerns.",
                journal="Drug Safety Journal",
                pub_year=2024,
                authors=["Min Kim"],
                url="https://pubmed.ncbi.nlm.nih.gov/12345/",
            ),
            EvidenceItem(
                source="pubmed",
                pmid="67890",
                title="Kinase inhibitor drug-drug interaction profile",
                abstract="Clearance variability and CYP3A4 inhibition may complicate exposure.",
                journal="Clinical PK",
                pub_year=2023,
                authors=["Jae Park"],
                url="https://pubmed.ncbi.nlm.nih.gov/67890/",
            ),
        ],
        source_health="ok",
    )


def make_sparse_evidence_packet() -> EvidencePacket:
    return EvidencePacket(
        source="pubmed",
        query="generic safety note",
        items=[
            EvidenceItem(
                source="pubmed",
                pmid="24680",
                title="Lead optimization update",
                abstract="Program review noted that additional safety and PK work is pending.",
                journal="Internal Safety Note",
                pub_year=2024,
                authors=["Team Safety"],
                url="https://pubmed.ncbi.nlm.nih.gov/24680/",
            )
        ],
        source_health="ok",
    )


def make_prediction_bundle() -> PredictionBundle:
    return PredictionBundle(
        signals=[
            PredictionSignal(
                name="hERG",
                value="high",
                confidence=0.91,
                risk_level="high",
            ),
            PredictionSignal(
                name="CYP inhibition",
                value="moderate CYP3A4 inhibition",
                confidence=0.78,
                risk_level="elevated",
            ),
        ]
    )


def make_tdc_prediction_bundle() -> PredictionBundle:
    return PredictionBundle(
        signals=[
            PredictionSignal(
                name="hERG",
                value="high hERG inhibition risk",
                confidence=0.91,
                risk_level="high",
            ),
            PredictionSignal(
                name="CYP3A4 inhibition",
                value="high CYP3A4 inhibition risk",
                confidence=0.78,
                risk_level="high",
            ),
            PredictionSignal(
                name="DILI",
                value="high DILI risk",
                confidence=0.74,
                risk_level="high",
            ),
            PredictionSignal(
                name="Clearance hepatocyte",
                value="Clearance hepatocyte 1200.00",
                confidence=0.68,
                risk_level="medium",
            ),
            PredictionSignal(
                name="CYP2D6 substrate",
                value="is a CYP2D6 substrate",
                confidence=0.63,
                risk_level="high",
            ),
        ]
    )


def test_house_agent_accepts_structured_output_from_runnable() -> None:
    evidence_packet = make_evidence_packet()
    prediction_bundle = make_prediction_bundle()
    runnable = SequenceRunnable(
        [
            {
                "summary": "Predicted hERG liability is supported by PubMed evidence.",
                "risks": ["High hERG liability"],
                "recommendations": ["Run a follow-up patch clamp assay."],
                "confidence": 0.82,
                "citations": [evidence_packet.items[0].url, "https://example.com/not-allowed"],
            }
        ]
    )
    agent = HouseAgent(analyzer_runnable=runnable)

    result = agent.analyze(
        question="심장독성/DDI 위험은?",
        target="KRAS G12C",
        evidence_packet=evidence_packet,
        prediction_bundle=prediction_bundle,
    )

    assert isinstance(result, AgentFinding)
    assert result.agent_id == "house"
    assert result.summary == "Predicted hERG liability is supported by PubMed evidence."
    assert result.risks == ["High hERG liability"]
    assert result.recommendations == ["Run a follow-up patch clamp assay."]
    assert result.confidence == 0.82
    assert result.citations == [evidence_packet.items[0].url]
    assert runnable.calls[0]["question"] == "심장독성/DDI 위험은?"
    assert runnable.calls[0]["prediction_bundle"]["signals"][0]["name"] == "hERG"
    assert runnable.calls[0]["evidence_items"][0]["pmid"] == "12345"


def test_house_agent_falls_back_when_runnable_output_is_invalid() -> None:
    evidence_packet = make_evidence_packet()
    prediction_bundle = make_prediction_bundle()
    runnable = SequenceRunnable(
        [
            {
                "summary": "This should not survive validation.",
                "risks": "not-a-list",
                "recommendations": [],
                "confidence": 0.9,
                "citations": [],
            }
        ]
    )
    agent = HouseAgent(analyzer_runnable=runnable)

    result = agent.analyze(
        question="심장독성/DDI 위험은?",
        target="KRAS G12C",
        evidence_packet=evidence_packet,
        prediction_bundle=prediction_bundle,
    )

    assert result.agent_id == "house"
    assert result.summary != "This should not survive validation."
    assert any("hERG" in risk for risk in result.risks)
    assert any("CYP" in risk or "DDI" in risk for risk in result.risks)
    assert any("QT" in recommendation or "CYP" in recommendation for recommendation in result.recommendations)
    assert result.citations == [item.url for item in evidence_packet.items]


def test_house_agent_returns_low_confidence_fallback_when_inputs_are_sparse() -> None:
    evidence_packet = EvidencePacket(
        source="pubmed",
        query="",
        items=[],
        source_health="degraded",
        missing_reason="no_pubmed_hits",
    )
    agent = HouseAgent()

    result = agent.analyze(
        question="독성 위험은?",
        evidence_packet=evidence_packet,
        prediction_bundle=None,
    )

    assert result.agent_id == "house"
    assert result.summary == "House found limited PK/tox evidence for a stronger conclusion."
    assert result.risks == ["Insufficient PK/tox signals collected yet."]
    assert result.recommendations == [
        "Collect additional PubMed PK/tox evidence and TxGemma predictions."
    ]
    assert result.citations == []
    assert result.confidence < 0.5


def test_house_agent_fallback_recognizes_tdc_ddi_hepatic_and_pk_signals() -> None:
    evidence_packet = make_sparse_evidence_packet()
    prediction_bundle = make_tdc_prediction_bundle()
    agent = HouseAgent()

    result = agent.analyze(
        question="심장독성/DDI 위험은?",
        target="KRAS G12C",
        evidence_packet=evidence_packet,
        prediction_bundle=prediction_bundle,
    )

    assert result.summary == (
        "House prioritized cardiotoxicity/hERG, DDI/CYP, hepatotoxicity, PK "
        "signals from predictions and literature."
    )
    assert "Elevated hERG/QT cardiotoxicity signal requires follow-up." in result.risks
    assert "Potential CYP/DDI interaction risk could alter exposure." in result.risks
    assert "Hepatotoxicity signal needs targeted liver safety review." in result.risks
    assert "PK variability around clearance or half-life needs confirmation." in result.risks
