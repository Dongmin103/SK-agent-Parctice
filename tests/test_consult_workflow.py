from __future__ import annotations

from app.agents.router_agent import RoutingDecision
from app.domain.models import (
    AgentFinding,
    EvidenceBundle,
    EvidenceItem,
    EvidencePacket,
    PredictionBundle,
    PredictionSignal,
)
from app.workflows.consult import ConsultWorkflow
from app.workflows.tracing import WorkflowTraceEvent


class StubTxGemmaClient:
    def __init__(self, bundle: PredictionBundle) -> None:
        self.bundle = bundle

    def predict(
        self,
        *,
        smiles: str,
        target: str | None = None,
        compound_name: str | None = None,
    ) -> PredictionBundle:
        return self.bundle


class StubRouterAgent:
    def __init__(self, decision: RoutingDecision) -> None:
        self.decision = decision

    def route(
        self,
        question: str,
        *,
        target: str | None = None,
        compound_name: str | None = None,
        prediction_bundle: PredictionBundle | None = None,
    ) -> RoutingDecision:
        return self.decision


class StubEvidenceCoordinator:
    def __init__(self, bundle: EvidenceBundle, domain_packets: dict[str, EvidencePacket]) -> None:
        self.bundle = bundle
        self.domain_packets = domain_packets

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
        return self.bundle

    def build_domain_packets(self, bundle: EvidenceBundle) -> dict[str, EvidencePacket]:
        return self.domain_packets


class StubExpertAgent:
    def __init__(self, finding: AgentFinding, *, should_fail: bool = False) -> None:
        self.finding = finding
        self.should_fail = should_fail
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
        if self.should_fail:
            raise RuntimeError(f"{self.finding.agent_id} failed")
        return self.finding


def make_prediction_bundle() -> PredictionBundle:
    return PredictionBundle(
        source="txgemma",
        target="KRAS G12C",
        compound_name="ABC-101",
        canonical_smiles="CCO",
        signals=[
            PredictionSignal(
                name="hERG",
                value="elevated",
                confidence=0.82,
                risk_level="high",
            )
        ],
    )


def make_packet(source: str, url_suffix: str) -> EvidencePacket:
    return EvidencePacket(
        source=source,
        query=f"{source} query",
        items=[
            EvidenceItem(
                source=source,
                pmid=url_suffix,
                title=f"{source} evidence",
                abstract=f"{source} abstract",
                journal="Test Journal",
                pub_year=2025,
                url=f"https://example.org/{url_suffix}",
                score=7.5,
            )
        ],
    )


def make_bundle(*packets: EvidencePacket) -> EvidenceBundle:
    items = []
    for packet in packets:
        items.extend(packet.items)
    return EvidenceBundle(
        query="bundle query",
        packets={packet.source: packet for packet in packets},
        items=items,
        source_health="ok",
    )


def test_consult_workflow_preserves_router_selection_order_and_domain_packet_mapping() -> None:
    chemistry_packet = make_packet("chemistry", "chemistry")
    safety_packet = make_packet("safety", "safety")
    clinical_packet = make_packet("clinical_regulatory", "clinical")
    workflow = ConsultWorkflow(
        prediction_client=StubTxGemmaClient(make_prediction_bundle()),
        router_agent=StubRouterAgent(
            RoutingDecision(
                question_type="multi_expert",
                selected_agents=["harvey", "house"],
                routing_reason="Mixed clinical and safety question.",
                confidence=0.81,
                fallback_used=False,
            )
        ),
        evidence_coordinator=StubEvidenceCoordinator(
            make_bundle(chemistry_packet, safety_packet, clinical_packet),
            {
                "chemistry": chemistry_packet,
                "safety": safety_packet,
                "clinical_regulatory": clinical_packet,
            },
        ),
        walter_agent=StubExpertAgent(AgentFinding(agent_id="walter", summary="unused")),
        house_agent=StubExpertAgent(
            AgentFinding(
                agent_id="house",
                summary="House summary",
                citations=[safety_packet.items[0].url],
            )
        ),
        harvey_agent=StubExpertAgent(
            AgentFinding(
                agent_id="harvey",
                summary="Harvey summary",
                citations=[clinical_packet.items[0].url],
            )
        ),
    )

    report = workflow.run(
        smiles="CCO",
        target="KRAS G12C",
        question="안전성과 승인 전략을 같이 봐줘.",
        compound_name="ABC-101",
    )

    assert report.selected_agents == ["harvey", "house"]
    assert [finding.agent_id for finding in report.agent_findings] == ["harvey", "house"]
    assert workflow.harvey_agent.calls[0]["evidence_packet"] == clinical_packet
    assert workflow.house_agent.calls[0]["evidence_packet"] == safety_packet
    assert workflow.walter_agent.calls == []


def test_consult_workflow_keeps_successful_findings_when_one_selected_agent_fails() -> None:
    chemistry_packet = make_packet("chemistry", "chemistry")
    safety_packet = make_packet("safety", "safety")
    workflow = ConsultWorkflow(
        prediction_client=StubTxGemmaClient(make_prediction_bundle()),
        router_agent=StubRouterAgent(
            RoutingDecision(
                question_type="multi_expert",
                selected_agents=["house", "walter"],
                routing_reason="Mixed safety and chemistry question.",
                confidence=0.72,
                fallback_used=True,
            )
        ),
        evidence_coordinator=StubEvidenceCoordinator(
            make_bundle(chemistry_packet, safety_packet),
            {
                "chemistry": chemistry_packet,
                "safety": safety_packet,
                "clinical_regulatory": EvidencePacket(source="clinical_regulatory", query="", items=[]),
            },
        ),
        walter_agent=StubExpertAgent(
            AgentFinding(
                agent_id="walter",
                summary="Walter kept a chemistry finding despite another agent failure.",
                citations=[chemistry_packet.items[0].url],
            )
        ),
        house_agent=StubExpertAgent(
            AgentFinding(agent_id="house", summary="House failed"),
            should_fail=True,
        ),
        harvey_agent=StubExpertAgent(AgentFinding(agent_id="harvey", summary="unused")),
    )

    report = workflow.run(
        smiles="CCO",
        target="KRAS G12C",
        question="구조와 독성 리스크를 같이 봐줘.",
        compound_name="ABC-101",
    )

    assert report.selected_agents == ["house", "walter"]
    assert [finding.agent_id for finding in report.agent_findings] == ["walter"]
    assert "Walter kept a chemistry finding" in report.consulting_answer
    assert report.citations == [chemistry_packet.items[0].url]


def test_consult_workflow_marks_review_required_and_tracks_missing_citations() -> None:
    safety_packet = make_packet("safety", "safety")
    clinical_packet = make_packet("clinical_regulatory", "clinical")
    workflow = ConsultWorkflow(
        prediction_client=StubTxGemmaClient(make_prediction_bundle()),
        router_agent=StubRouterAgent(
            RoutingDecision(
                question_type="multi_expert",
                selected_agents=["house", "harvey"],
                routing_reason="Safety and clinical evidence are both needed.",
                confidence=0.76,
                fallback_used=False,
            )
        ),
        evidence_coordinator=StubEvidenceCoordinator(
            make_bundle(safety_packet, clinical_packet),
            {
                "chemistry": EvidencePacket(source="chemistry", query="", items=[]),
                "safety": safety_packet,
                "clinical_regulatory": clinical_packet,
            },
        ),
        walter_agent=StubExpertAgent(AgentFinding(agent_id="walter", summary="unused")),
        house_agent=StubExpertAgent(
            AgentFinding(
                agent_id="house",
                summary="House summary with evidence.",
                recommendations=["Run a confirmatory patch clamp assay."],
                citations=[safety_packet.items[0].url],
            )
        ),
        harvey_agent=StubExpertAgent(
            AgentFinding(
                agent_id="harvey",
                summary="Harvey summary without citation.",
                recommendations=["Clarify the clinical positioning strategy."],
                citations=[],
            )
        ),
    )

    report = workflow.run(
        smiles="CCO",
        target="KRAS G12C",
        question="안전성과 개발 전략을 같이 봐줘.",
        compound_name="ABC-101",
    )

    assert report.review_required is True
    assert report.citation_validation.complete is False
    assert report.citation_validation.missing_agent_ids == ["harvey"]
    assert report.citation_validation.total_citations == 1
    assert report.citations == [safety_packet.items[0].url]
    assert "House summary with evidence." in report.consulting_answer
    assert "Harvey summary without citation." in report.consulting_answer


def test_consult_workflow_emits_operational_trace_events() -> None:
    safety_domain_packet = make_packet("safety", "safety")
    clinical_domain_packet = make_packet("clinical_regulatory", "clinical")
    evidence_bundle = EvidenceBundle(
        query="trace query",
        packets={
            "pubmed": EvidencePacket(
                source="pubmed",
                query='"ABC-101" AND "cardiotoxicity"',
                items=list(safety_domain_packet.items),
                source_health="ok",
                diagnostics={
                    "query_planner_used": "true",
                    "query_planner_selected_query": '"ABC-101" AND "cardiotoxicity"',
                    "query_planner_best_hit_count": "12",
                },
            ),
            "clinicaltrials": EvidencePacket(
                source="clinicaltrials",
                query="ABC-101 clinical trial",
                items=list(clinical_domain_packet.items),
                source_health="ok",
            ),
            "openfda": EvidencePacket(
                source="openfda",
                query='openfda.generic_name:"ABC-101"',
                items=[],
                source_health="degraded",
                missing_reason="openfda_request_failed",
                diagnostics={"error": "upstream 503"},
            ),
        },
        items=list(safety_domain_packet.items) + list(clinical_domain_packet.items),
        source_health="partial",
        partial_failures=["openfda"],
    )
    workflow = ConsultWorkflow(
        prediction_client=StubTxGemmaClient(make_prediction_bundle()),
        router_agent=StubRouterAgent(
            RoutingDecision(
                question_type="multi_expert",
                selected_agents=["house", "harvey"],
                routing_reason="Safety and clinical evidence are both needed.",
                confidence=0.76,
                fallback_used=False,
            )
        ),
        evidence_coordinator=StubEvidenceCoordinator(
            evidence_bundle,
            {
                "chemistry": EvidencePacket(source="chemistry", query="", items=[]),
                "safety": safety_domain_packet,
                "clinical_regulatory": clinical_domain_packet,
            },
        ),
        walter_agent=StubExpertAgent(AgentFinding(agent_id="walter", summary="unused")),
        house_agent=StubExpertAgent(
            AgentFinding(
                agent_id="house",
                summary="House summary with evidence.",
                citations=[safety_domain_packet.items[0].url],
            )
        ),
        harvey_agent=StubExpertAgent(
            AgentFinding(
                agent_id="harvey",
                summary="Harvey summary with evidence.",
                citations=[clinical_domain_packet.items[0].url],
            )
        ),
    )
    trace_events: list[WorkflowTraceEvent] = []

    workflow.run(
        smiles="CCO",
        target="KRAS G12C",
        question="안전성과 개발 전략을 같이 봐줘.",
        compound_name="ABC-101",
        event_sink=trace_events.append,
    )

    messages = [event.message for event in trace_events]
    assert "Selected agents: house, harvey" in messages
    assert 'PubMed planner selected query: "ABC-101" AND "cardiotoxicity"' in messages
    assert "PubMed dry run hits: 12" in messages
    assert "openFDA degraded: upstream 503" in messages
