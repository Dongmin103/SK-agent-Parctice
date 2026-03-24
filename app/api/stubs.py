from __future__ import annotations

from app.domain.compound import CompoundPreprocessor
from app.domain.models import (
    AgentFinding,
    DecisionDraft,
    EvidenceBundle,
    EvidenceItem,
    EvidencePacket,
    PredictionBundle,
    PredictionSignal,
)
from app.workflows.consult import ConsultReport
from app.workflows.executive import ExecutiveReport


class StubConsultWorkflow:
    def __init__(self, *, compound_preprocessor: CompoundPreprocessor | None = None) -> None:
        self.compound_preprocessor = compound_preprocessor or CompoundPreprocessor()

    def run(
        self,
        *,
        smiles: str,
        target: str,
        question: str,
        compound_name: str | None = None,
    ) -> ConsultReport:
        del question
        compound_context = self.compound_preprocessor.build_context(
            smiles=smiles,
            target=target,
            compound_name=compound_name,
        )
        prediction_bundle = _build_prediction_bundle(
            smiles=compound_context.canonical_smiles or compound_context.smiles,
            target=target,
            compound_name=compound_name,
        )
        finding = AgentFinding(
            agent_id="house",
            summary="House sees a manageable hERG follow-up need in the stub workflow.",
            risks=["Monitor hERG closely."],
            recommendations=["Repeat hERG assay before advancing."],
            confidence=0.78,
            citations=["https://example.org/stub/house"],
        )
        return ConsultReport(
            selected_agents=["house"],
            routing_reason="Stub workflow mapped the smoke-test question to House.",
            predictions=prediction_bundle,
            agent_findings=[finding],
            consulting_answer=(
                "질문에 대해 House 관점에서 검토했습니다. "
                "House sees a manageable hERG follow-up need in the stub workflow."
            ),
            citations=list(finding.citations),
            review_required=True,
            review_reasons=["연구용 결과이므로 사람 검토가 필요합니다."],
        )


class StubExecutiveWorkflow:
    def __init__(self, *, compound_preprocessor: CompoundPreprocessor | None = None) -> None:
        self.compound_preprocessor = compound_preprocessor or CompoundPreprocessor()

    def run(
        self,
        *,
        smiles: str,
        target: str,
        compound_name: str | None = None,
    ) -> ExecutiveReport:
        compound_context = self.compound_preprocessor.build_context(
            smiles=smiles,
            target=target,
            compound_name=compound_name,
        )
        prediction_bundle = _build_prediction_bundle(
            smiles=compound_context.canonical_smiles or compound_context.smiles,
            target=target,
            compound_name=compound_name,
        )
        evidence_item = EvidenceItem(
            source="stub",
            pmid="stub-executive",
            title="Stub executive evidence",
            abstract="Stub evidence for local API smoke tests.",
            journal="Stub Journal",
            pub_year=2026,
            url="https://example.org/stub/executive",
            score=7.5,
        )
        evidence_bundle = EvidenceBundle(
            query=f"Executive assessment for {compound_name or smiles} against {target}",
            packets={
                "stub": EvidencePacket(
                    source="stub",
                    query=compound_name or smiles,
                    items=[evidence_item],
                )
            },
            items=[evidence_item],
            source_health="ok",
        )
        finding = AgentFinding(
            agent_id="house",
            summary="House sees a manageable safety signal in the stub workflow.",
            risks=["Monitor hERG closely."],
            recommendations=["Repeat hERG assay before IND-enabling studies."],
            confidence=0.8,
            citations=[evidence_item.url],
        )
        return ExecutiveReport(
            predictions=prediction_bundle,
            evidence_bundle=evidence_bundle,
            agent_findings=[finding],
            executive_summary="CEO stub summary",
            executive_decision=DecisionDraft(
                decision="conditional_go",
                rationale="Stub workflow requires one follow-up assay before advancing.",
                next_steps=["Repeat hERG assay before IND-enabling studies."],
            ),
            citations=[evidence_item.url],
            review_required=True,
            review_reasons=["연구용 결과이므로 사람 검토가 필요합니다."],
            canonical_smiles=compound_context.canonical_smiles,
            molecule_svg=compound_context.molecule_svg,
        )


def build_stub_consult_workflow() -> StubConsultWorkflow:
    return StubConsultWorkflow()


def build_stub_executive_workflow() -> StubExecutiveWorkflow:
    return StubExecutiveWorkflow()


def _build_prediction_bundle(
    *,
    smiles: str,
    target: str,
    compound_name: str | None,
) -> PredictionBundle:
    return PredictionBundle(
        source="txgemma_stub",
        target=target,
        compound_name=compound_name,
        canonical_smiles=smiles,
        signals=[
            PredictionSignal(
                name="hERG",
                value="monitor",
                confidence=0.75,
                risk_level="moderate",
            )
        ],
        missing_signals=[],
    )
