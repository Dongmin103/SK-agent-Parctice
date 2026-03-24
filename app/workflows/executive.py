from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from app.agents.ceo_synthesizer import CEOSynthesizer
from app.agents.parallel_executor import ParallelExecutionReport, ParallelExecutor
from app.agents.review_policy import ReviewPolicy
from app.agents.registry import AgentRegistry, build_agent_registry
from app.domain.compound import CompoundPreprocessor
from app.domain.models import (
    AgentFinding,
    CitationValidation,
    DecisionDraft,
    EvidenceBundle,
    EvidencePacket,
    PredictionBundle,
)
from app.workflows.consult import AGENT_TO_DOMAIN_PACKET
from app.workflows.tracing import EventSink, emit_evidence_trace_events, emit_trace

LOGGER = logging.getLogger(__name__)

EXECUTIVE_AGENT_ORDER = ("walter", "house", "harvey")


@dataclass(slots=True)
class ExecutiveReport:
    predictions: PredictionBundle
    evidence_bundle: EvidenceBundle
    agent_findings: list[AgentFinding] = field(default_factory=list)
    executive_summary: str = ""
    executive_decision: DecisionDraft = field(
        default_factory=lambda: DecisionDraft(
            decision="no_go",
            rationale="근거가 부족해 진행 판단을 내릴 수 없습니다.",
            next_steps=["추가 근거를 수집합니다."],
        )
    )
    citations: list[str] = field(default_factory=list)
    citation_validation: CitationValidation = field(default_factory=CitationValidation)
    review_required: bool = True
    review_reasons: list[str] = field(default_factory=list)
    canonical_smiles: str | None = None
    molecule_svg: str | None = None


class ExecutiveWorkflow:
    """Run all expert agents and synthesize an executive decision."""

    def __init__(
        self,
        *,
        prediction_client: Any,
        evidence_coordinator: Any,
        walter_agent: Any,
        house_agent: Any,
        harvey_agent: Any,
        agent_registry: AgentRegistry | None = None,
        parallel_executor: ParallelExecutor | None = None,
        ceo_synthesizer: CEOSynthesizer | None = None,
        review_policy: ReviewPolicy | None = None,
        compound_preprocessor: CompoundPreprocessor | None = None,
        retmax: int = 10,
        top_k: int = 5,
    ) -> None:
        self.prediction_client = prediction_client
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
        self.ceo_synthesizer = ceo_synthesizer or CEOSynthesizer()
        self.review_policy = review_policy or ReviewPolicy()
        self.compound_preprocessor = compound_preprocessor or CompoundPreprocessor()
        self.retmax = retmax
        self.top_k = top_k

    def run(
        self,
        *,
        smiles: str,
        target: str,
        compound_name: str | None = None,
        event_sink: EventSink | None = None,
    ) -> ExecutiveReport:
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
        question = self._build_executive_question(
            smiles=smiles,
            target=target,
            compound_name=compound_name,
        )
        emit_trace(
            event_sink,
            stage="routing",
            message=f"Selected agents: {', '.join(EXECUTIVE_AGENT_ORDER)}",
            details={"selected_agents": ",".join(EXECUTIVE_AGENT_ORDER)},
        )
        emit_trace(
            event_sink,
            stage="evidence",
            message="Collecting evidence across PubChem, ChEMBL, PubMed, ClinicalTrials.gov, openFDA.",
        )
        evidence_bundle = self.evidence_coordinator.collect_evidence(
            question=question,
            question_type="multi_expert",
            target=target,
            compound_name=compound_name,
            smiles=compound_context.canonical_smiles or compound_context.smiles,
            retmax=self.retmax,
            top_k=self.top_k,
        )
        emit_evidence_trace_events(event_sink, evidence_bundle)
        domain_packets = self.evidence_coordinator.build_domain_packets(evidence_bundle)
        execution_targets = self._build_execution_targets(
            target=target,
            compound_name=compound_name,
            prediction_bundle=predictions,
            domain_packets=domain_packets,
        )
        execution_report = self.parallel_executor.execute(
            EXECUTIVE_AGENT_ORDER,
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
        synthesis = self.ceo_synthesizer.synthesize(findings)
        review_decision = self.review_policy.evaluate(
            citation_validation=synthesis.citation_validation,
            findings=findings,
        )
        emit_trace(
            event_sink,
            stage="synthesis",
            message="Executive synthesis completed.",
            details={"citation_count": str(len(synthesis.citations))},
        )

        return ExecutiveReport(
            predictions=predictions,
            evidence_bundle=evidence_bundle,
            agent_findings=findings,
            executive_summary=synthesis.summary,
            executive_decision=synthesis.decision_draft,
            citations=list(synthesis.citations),
            citation_validation=synthesis.citation_validation,
            review_required=review_decision.review_required,
            review_reasons=review_decision.reasons,
            canonical_smiles=compound_context.canonical_smiles,
            molecule_svg=compound_context.molecule_svg,
        )

    def _build_executive_question(
        self,
        *,
        smiles: str,
        target: str,
        compound_name: str | None,
    ) -> str:
        subject = compound_name or smiles
        return f"Executive assessment for {subject} against {target}"

    def _build_execution_targets(
        self,
        *,
        target: str,
        compound_name: str | None,
        prediction_bundle: PredictionBundle,
        domain_packets: dict[str, EvidencePacket],
    ) -> dict[str, Callable[..., AgentFinding]]:
        execution_targets: dict[str, Callable[..., AgentFinding]] = {}
        for agent_id in EXECUTIVE_AGENT_ORDER:
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
                LOGGER.warning("Executive agent execution failed for %s: %s", task.agent_id, task.error)
                continue
            if not isinstance(task.output, AgentFinding):
                LOGGER.warning(
                    "Executive agent execution returned unexpected output for %s: %r",
                    task.agent_id,
                    task.output,
                )
                continue
            findings.append(task.output)
        return findings
