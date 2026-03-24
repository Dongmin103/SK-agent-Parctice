from __future__ import annotations

from app.agents.walter_agent import WalterAgent
from app.domain.models import AgentFinding, EvidenceItem, EvidencePacket, PredictionBundle, PredictionSignal


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
        source="chemistry",
        query='"KRAS G12D" AND (SAR OR analog OR solubility)',
        items=[
            EvidenceItem(
                source="chembl",
                pmid="101",
                title="Scaffold-focused SAR around KRAS G12D inhibitor analogs",
                abstract="Matched analog series improved potency but high lipophilicity reduced developability.",
                journal="Journal of Medicinal Chemistry",
                pub_year=2024,
                authors=["Min Kim"],
                url="https://chembl.example/compound/CHEMBL101",
            ),
            EvidenceItem(
                source="pubchem",
                pmid="202",
                title="Solubility optimization for a covalent KRAS inhibitor series",
                abstract="Aqueous solubility improved after reducing logD while retaining acceptable permeability.",
                journal="ACS Medicinal Chemistry Letters",
                pub_year=2023,
                authors=["Jae Park"],
                url="https://pubchem.example/compound/202",
            ),
        ],
        source_health="ok",
    )


def make_sparse_evidence_packet() -> EvidencePacket:
    return EvidencePacket(
        source="chemistry",
        query="generic chemistry note",
        items=[
            EvidenceItem(
                source="chembl",
                pmid="303",
                title="Lead series update",
                abstract="Lead review is ongoing with no property conclusions yet.",
                journal="Internal Chemistry Memo",
                pub_year=2024,
                authors=["Team Chem"],
                url="https://chembl.example/compound/CHEMBL303",
            )
        ],
        source_health="ok",
    )


def make_prediction_bundle() -> PredictionBundle:
    return PredictionBundle(
        signals=[
            PredictionSignal(
                name="logP",
                value="high",
                confidence=0.89,
                risk_level="high",
            ),
            PredictionSignal(
                name="solubility",
                value="low aqueous solubility",
                confidence=0.85,
                risk_level="elevated",
            ),
            PredictionSignal(
                name="permeability",
                value="acceptable",
                confidence=0.71,
                risk_level="moderate",
            ),
        ]
    )


def make_tdc_prediction_bundle() -> PredictionBundle:
    return PredictionBundle(
        signals=[
            PredictionSignal(
                name="Lipophilicity",
                value="Lipophilicity 5.60",
                confidence=0.89,
                risk_level="high",
            ),
            PredictionSignal(
                name="Solubility",
                value="Solubility -4.20",
                confidence=0.85,
                risk_level="high",
            ),
            PredictionSignal(
                name="Caco2",
                value="Caco2 -0.50",
                confidence=0.71,
                risk_level="medium",
            ),
            PredictionSignal(
                name="Pgp",
                value="high P-glycoprotein efflux liability",
                confidence=0.77,
                risk_level="high",
            ),
            PredictionSignal(
                name="HIA",
                value="high human intestinal absorption",
                confidence=0.66,
                risk_level="low",
            ),
        ]
    )


def test_walter_agent_accepts_structured_output_from_runnable() -> None:
    evidence_packet = make_evidence_packet()
    prediction_bundle = make_prediction_bundle()
    runnable = SequenceRunnable(
        [
            {
                "summary": "Walter sees an SAR-backed chemistry optimization path.",
                "risks": ["Lipophilicity remains higher than desired."],
                "recommendations": ["Trim lipophilicity while preserving the core SAR motif."],
                "confidence": 0.84,
                "citations": [evidence_packet.items[1].url, "https://example.com/not-allowed"],
            }
        ]
    )
    agent = WalterAgent(analyzer_runnable=runnable)

    result = agent.analyze(
        question="구조 개선 방향은?",
        target="KRAS G12D",
        evidence_packet=evidence_packet,
        prediction_bundle=prediction_bundle,
    )

    assert isinstance(result, AgentFinding)
    assert result.agent_id == "walter"
    assert result.summary == "Walter sees an SAR-backed chemistry optimization path."
    assert result.risks == ["Lipophilicity remains higher than desired."]
    assert result.recommendations == ["Trim lipophilicity while preserving the core SAR motif."]
    assert result.confidence == 0.84
    assert result.citations == [evidence_packet.items[1].url]
    assert runnable.calls[0]["question"] == "구조 개선 방향은?"
    assert runnable.calls[0]["prediction_bundle"]["signals"][0]["name"] == "logP"
    assert runnable.calls[0]["evidence_items"][0]["pmid"] == "101"


def test_walter_agent_falls_back_when_runnable_output_is_invalid() -> None:
    evidence_packet = make_evidence_packet()
    prediction_bundle = make_prediction_bundle()
    runnable = SequenceRunnable(
        [
            {
                "summary": "This should not survive validation.",
                "risks": "not-a-list",
                "recommendations": [],
                "confidence": 0.91,
                "citations": [],
            }
        ]
    )
    agent = WalterAgent(analyzer_runnable=runnable)

    result = agent.analyze(
        question="구조 개선 방향은?",
        target="KRAS G12D",
        evidence_packet=evidence_packet,
        prediction_bundle=prediction_bundle,
    )

    assert result.agent_id == "walter"
    assert result.summary != "This should not survive validation."
    assert any("SAR" in risk or "scaffold" in risk for risk in result.risks)
    assert any("solubility" in risk.lower() for risk in result.risks)
    assert any("lipophilicity" in recommendation.lower() for recommendation in result.recommendations)
    assert result.citations == [item.url for item in evidence_packet.items]


def test_walter_agent_returns_low_confidence_fallback_when_inputs_are_sparse() -> None:
    evidence_packet = EvidencePacket(
        source="chemistry",
        query="",
        items=[],
        source_health="degraded",
        missing_reason="no_chemistry_hits",
    )
    agent = WalterAgent()

    result = agent.analyze(
        question="구조 개선 방향은?",
        evidence_packet=evidence_packet,
        prediction_bundle=None,
    )

    assert result.agent_id == "walter"
    assert result.summary == "Walter found limited chemistry evidence for a stronger conclusion."
    assert result.risks == ["Insufficient SAR or structure/property signals collected yet."]
    assert result.recommendations == [
        "Collect additional PubChem/ChEMBL chemistry evidence and property predictions."
    ]
    assert result.citations == []
    assert result.confidence < 0.5


def test_walter_agent_fallback_orders_categories_deterministically() -> None:
    evidence_packet = make_evidence_packet()
    prediction_bundle = make_prediction_bundle()
    agent = WalterAgent()

    result = agent.analyze(
        question="구조 개선 방향은?",
        target="KRAS G12D",
        evidence_packet=evidence_packet,
        prediction_bundle=prediction_bundle,
    )

    assert result.summary == (
        "Walter prioritized SAR/scaffold, solubility, lipophilicity, permeability "
        "signals from chemistry evidence and property predictions."
    )
    assert result.risks == [
        "SAR/scaffold evidence suggests more analog refinement is needed.",
        "Low solubility could limit chemistry progression.",
        "High lipophilicity/logD may reduce developability.",
        "Permeability trade-offs should be checked during optimization.",
    ]
    assert result.recommendations == [
        "Expand analog exploration around the most informative scaffold changes.",
        "Prioritize solubility-focused analog design and confirm experimentally.",
        "Reduce lipophilicity/logD with targeted substituent or scaffold edits.",
        "Balance permeability against polarity during lead optimization.",
    ]
    assert result.citations == [item.url for item in evidence_packet.items]


def test_walter_agent_fallback_recognizes_tdc_absorption_and_lipophilicity_signals() -> None:
    evidence_packet = make_sparse_evidence_packet()
    prediction_bundle = make_tdc_prediction_bundle()
    agent = WalterAgent()

    result = agent.analyze(
        question="구조 개선 방향은?",
        target="KRAS G12D",
        evidence_packet=evidence_packet,
        prediction_bundle=prediction_bundle,
    )

    assert result.summary == (
        "Walter prioritized solubility, lipophilicity, permeability "
        "signals from chemistry evidence and property predictions."
    )
    assert "Low solubility could limit chemistry progression." in result.risks
    assert "High lipophilicity/logD may reduce developability." in result.risks
    assert "Permeability trade-offs should be checked during optimization." in result.risks
