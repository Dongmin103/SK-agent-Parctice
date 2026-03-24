from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from app.agents.answer_composer import AnswerComposer
from app.agents.parallel_executor import ParallelExecutionReport, ParallelExecutor
from app.agents.review_policy import ReviewPolicy
from app.agents.registry import AgentRegistry, build_agent_registry
from app.domain.compound import CompoundPreprocessor
from app.domain.models import AgentFinding, CitationValidation, EvidencePacket, PredictionBundle
from app.workflows.tracing import EventSink, emit_evidence_trace_events, emit_trace

LOGGER = logging.getLogger(__name__)

AGENT_TO_DOMAIN_PACKET = {
    "walter": "chemistry",
    "house": "safety",
    "harvey": "clinical_regulatory",
}


@dataclass(slots=True)
class ConsultReport:
    selected_agents: list[str]
    routing_reason: str
    predictions: PredictionBundle
    agent_findings: list[AgentFinding] = field(default_factory=list)
    consulting_answer: str = ""
    citations: list[str] = field(default_factory=list)
    citation_validation: CitationValidation = field(default_factory=CitationValidation)
    review_required: bool = True
    review_reasons: list[str] = field(default_factory=list)


class ConsultWorkflow:
    """Wire prediction, routing, evidence collection, and selected experts."""

    def __init__(
        self,
        *,
        prediction_client: Any,
        router_agent: Any,
        evidence_coordinator: Any,
        walter_agent: Any,
        house_agent: Any,
        harvey_agent: Any,
        agent_registry: AgentRegistry | None = None,
        parallel_executor: ParallelExecutor | None = None,
        answer_composer: AnswerComposer | None = None,
        review_policy: ReviewPolicy | None = None,
        compound_preprocessor: CompoundPreprocessor | None = None,
        retmax: int = 10,
        top_k: int = 5,
    ) -> None:
        self.prediction_client = prediction_client
        self.router_agent = router_agent
        self.evidence_coordinator = evidence_coordinator
        self.walter_agent = walter_agent
        self.house_agent = house_agent
        self.harvey_agent = harvey_agent
        self.agent_registry = agent_registry or build_agent_registry(
            walter=walter_agent,
            house=house_agent,
            harvey=harvey_agent,
        )
        self.parallel_executor = parallel_executor or ParallelExecutor()
        self.answer_composer = answer_composer or AnswerComposer()
        self.review_policy = review_policy or ReviewPolicy()
        self.compound_preprocessor = compound_preprocessor or CompoundPreprocessor()
        self.retmax = retmax
        self.top_k = top_k

    def run(
        self,
        *,
        smiles: str,
        target: str,
        question: str,
        compound_name: str | None = None,
        event_sink: EventSink | None = None,
    ) -> ConsultReport:
        emit_trace(
            event_sink,
            stage="preprocessing",
            message="Preparing compound context.",
        )
        compound_context = self.compound_preprocessor.build_context(
            smiles=smiles,
            target=target,
            compound_name=compound_name,
        )
        predictions = self.prediction_client.predict(
            smiles=compound_context.canonical_smiles or compound_context.smiles,
            target=target,
            compound_name=compound_name,
        )
        predictions.canonical_smiles = compound_context.canonical_smiles
        emit_trace(
            event_sink,
            stage="prediction",
            message=(
                f"TxGemma prediction completed: {len(predictions.signals)} signals, "
                f"{len(predictions.missing_signals)} missing."
            ),
            details={
                "signal_count": str(len(predictions.signals)),
                "missing_signal_count": str(len(predictions.missing_signals)),
            },
        )
        routing = self.router_agent.route(
            question,
            target=target,
            compound_name=compound_name,
            prediction_bundle=predictions,
        )
        emit_trace(
            event_sink,
            stage="routing",
            message=f"Selected agents: {', '.join(routing.selected_agents)}",
            details={
                "selected_agents": ",".join(routing.selected_agents),
                "routing_reason": routing.routing_reason,
            },
        )
        emit_trace(
            event_sink,
            stage="evidence",
            message="Collecting evidence across PubChem, ChEMBL, PubMed, ClinicalTrials.gov, openFDA.",
        )
        evidence_bundle = self.evidence_coordinator.collect_evidence(
            question=question,
            question_type=routing.question_type,
            target=target,
            compound_name=compound_name,
            retmax=self.retmax,
            top_k=self.top_k,
        )
        emit_evidence_trace_events(event_sink, evidence_bundle)
        domain_packets = self.evidence_coordinator.build_domain_packets(evidence_bundle)
        execution_targets = self._build_execution_targets(
            selected_agents=list(routing.selected_agents),
            target=target,
            compound_name=compound_name,
            prediction_bundle=predictions,
            domain_packets=domain_packets,
        )
        execution_report = self.parallel_executor.execute(
            routing.selected_agents,
            question=question,
            agents_by_id=execution_targets,
        )
        findings = self._extract_findings(execution_report)
        emit_trace(
            event_sink,
            stage="expert_analysis",
            message=f"Expert analysis completed: {', '.join(finding.agent_id for finding in findings)}",
            details={"completed_agents": ",".join(finding.agent_id for finding in findings)},
        )
        composed_answer = self.answer_composer.compose(
            selected_agents=list(routing.selected_agents),
            findings=findings,
        )
        review_decision = self.review_policy.evaluate(
            citation_validation=composed_answer.citation_validation,
            findings=findings,
        )
        emit_trace(
            event_sink,
            stage="synthesis",
            message="Consult answer composed.",
            details={"citation_count": str(len(composed_answer.citations))},
        )

        return ConsultReport(
            selected_agents=list(routing.selected_agents),
            routing_reason=routing.routing_reason,
            predictions=predictions,
            agent_findings=findings,
            consulting_answer=composed_answer.answer,
            citations=composed_answer.citations,
            citation_validation=composed_answer.citation_validation,
            review_required=review_decision.review_required,
            review_reasons=review_decision.reasons,
        )

    def _build_execution_targets(
        self,
        *,
        selected_agents: list[str],
        target: str,
        compound_name: str | None,
        prediction_bundle: PredictionBundle,
        domain_packets: dict[str, EvidencePacket],
    ) -> dict[str, Callable[..., AgentFinding]]:
        execution_targets: dict[str, Callable[..., AgentFinding]] = {}
        for agent_id in selected_agents:
            agent = self.agent_registry.resolve(agent_id)
            domain_key = AGENT_TO_DOMAIN_PACKET.get(agent_id)
            evidence_packet = domain_packets.get(domain_key) if domain_key is not None else None
            execution_targets[agent_id] = self._build_bound_agent_callable(
                agent,
                target=target,
                compound_name=compound_name,
                evidence_packet=evidence_packet,
                prediction_bundle=prediction_bundle,
            )
        return execution_targets

    def _build_bound_agent_callable(
        self,
        agent: Any,
        *,
        target: str,
        compound_name: str | None,
        evidence_packet: EvidencePacket | None,
        prediction_bundle: PredictionBundle,
    ) -> Callable[..., AgentFinding]:
        def _invoke(question: str, **_: Any) -> AgentFinding:
            return agent.analyze(
                question,
                target=target,
                compound_name=compound_name,
                evidence_packet=evidence_packet,
                prediction_bundle=prediction_bundle,
            )

        return _invoke

    def _extract_findings(self, execution_report: ParallelExecutionReport) -> list[AgentFinding]:
        findings: list[AgentFinding] = []
        for task in execution_report.tasks:
            if not task.ok:
                LOGGER.warning("Consult agent execution failed for %s: %s", task.agent_id, task.error)
                continue
            if not isinstance(task.output, AgentFinding):
                LOGGER.warning(
                    "Consult agent execution returned unexpected output for %s: %r",
                    task.agent_id,
                    task.output,
                )
                continue
            findings.append(task.output)
        return findings
