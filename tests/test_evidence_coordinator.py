from __future__ import annotations

from app.agents.pubmed_query_agent import PubMedQueryAgentResult
from app.domain.models import EvidenceItem, EvidencePacket
from app.clients.evidence_coordinator import EvidenceCoordinator


class StubPubChemClient:
    def __init__(self, packet: EvidencePacket) -> None:
        self.packet = packet
        self.calls: list[dict[str, object]] = []

    def collect_pubchem_evidence(self, question: str, question_type: str, **kwargs) -> EvidencePacket:
        self.calls.append({"question": question, "question_type": question_type, **kwargs})
        return self.packet


class StubChEMBLClient:
    def __init__(self, packet: EvidencePacket) -> None:
        self.packet = packet

    def collect_chembl_evidence(self, question: str, question_type: str, **kwargs) -> EvidencePacket:
        return self.packet


class StubClinicalTrialsClient:
    def __init__(self, packet: EvidencePacket | None = None, exc: Exception | None = None) -> None:
        self.packet = packet
        self.exc = exc

    def collect_clinicaltrials_evidence(
        self, question: str, question_type: str, **kwargs
    ) -> EvidencePacket:
        if self.exc is not None:
            raise self.exc
        assert self.packet is not None
        return self.packet


class StubOpenFDAClient:
    def __init__(self, packet: EvidencePacket) -> None:
        self.packet = packet

    def collect_openfda_evidence(self, question: str, question_type: str, **kwargs) -> EvidencePacket:
        return self.packet


class StubPubMedClient:
    def __init__(
        self,
        packet: EvidencePacket,
        *,
        packet_from_queries: EvidencePacket | None = None,
    ) -> None:
        self.packet = packet
        self.packet_from_queries = packet_from_queries or packet
        self.collect_calls: list[dict[str, object]] = []
        self.collect_from_queries_calls: list[dict[str, object]] = []

    def collect_pubmed_evidence(self, question: str, question_type: str, **kwargs) -> EvidencePacket:
        self.collect_calls.append({"question": question, "question_type": question_type, **kwargs})
        return self.packet

    def collect_pubmed_evidence_from_queries(
        self,
        question: str,
        queries: list[str],
        **kwargs,
    ) -> EvidencePacket:
        self.collect_from_queries_calls.append(
            {
                "question": question,
                "queries": queries,
                **kwargs,
            }
        )
        return self.packet_from_queries


class StubPubMedQueryPlanner:
    def __init__(
        self,
        result: PubMedQueryAgentResult | None = None,
        *,
        exc: Exception | None = None,
    ) -> None:
        self.result = result
        self.exc = exc
        self.calls: list[dict[str, object]] = []

    def plan(self, request) -> PubMedQueryAgentResult:
        self.calls.append(
            {
                "question": request.question,
                "question_type": request.question_type,
                "target": request.target,
                "compound_name": request.compound_name,
                "prediction_flags": request.prediction_flags,
            }
        )
        if self.exc is not None:
            raise self.exc
        assert self.result is not None
        return self.result


def make_item(source: str, record_id: str, title: str, score: float, url: str) -> EvidenceItem:
    return EvidenceItem(
        source=source,
        pmid=record_id,
        title=title,
        abstract=title,
        journal=source,
        pub_year=2025,
        url=url,
        score=score,
    )


def test_evidence_coordinator_aggregates_packets_and_marks_partial_failures() -> None:
    pubchem_packet = EvidencePacket(
        source="pubchem",
        query="ABC-101",
        items=[
            make_item(
                "pubchem",
                "2244",
                "PubChem summary for ABC-101",
                6.0,
                "https://pubchem.ncbi.nlm.nih.gov/compound/2244",
            )
        ],
        source_health="ok",
    )
    chembl_packet = EvidencePacket(
        source="chembl",
        query="ABC-101",
        items=[],
        source_health="degraded",
        missing_reason="no_chembl_hits",
    )
    pubmed_packet = EvidencePacket(
        source="pubmed",
        query='"ABC-101" AND hERG',
        items=[
            make_item(
                "pubmed",
                "12345",
                "PubMed toxicity evidence for ABC-101",
                8.5,
                "https://pubmed.ncbi.nlm.nih.gov/12345/",
            )
        ],
        source_health="ok",
    )
    openfda_packet = EvidencePacket(
        source="openfda",
        query="ABC-101",
        items=[
            make_item(
                "openfda",
                "FDA-2026-001",
                "openFDA label signal for ABC-101",
                5.0,
                "https://api.fda.gov/drug/label/example",
            )
        ],
        source_health="ok",
    )

    coordinator = EvidenceCoordinator(
        pubmed_client=StubPubMedClient(pubmed_packet),
        pubchem_client=StubPubChemClient(pubchem_packet),
        chembl_client=StubChEMBLClient(chembl_packet),
        clinicaltrials_client=StubClinicalTrialsClient(exc=RuntimeError("upstream 503")),
        openfda_client=StubOpenFDAClient(openfda_packet),
    )

    bundle = coordinator.collect_evidence(
        question="독성, 임상, chemistry evidence를 모아줘",
        question_type="safety",
        target="KRAS G12C",
        compound_name="ABC-101",
    )

    assert bundle.source_health == "partial"
    assert bundle.missing_sources == ["chembl", "clinicaltrials"]
    assert bundle.partial_failures == ["clinicaltrials"]
    assert bundle.items[0].source == "pubmed"
    assert [item.source for item in bundle.items] == ["pubmed", "pubchem", "openfda"]
    assert bundle.packets["clinicaltrials"].source_health == "degraded"
    assert bundle.packets["clinicaltrials"].missing_reason == "clinicaltrials_request_failed"
    assert bundle.packets["clinicaltrials"].diagnostics == {"error": "upstream 503"}


def test_evidence_coordinator_returns_degraded_bundle_when_all_sources_miss() -> None:
    coordinator = EvidenceCoordinator(
        pubmed_client=StubPubMedClient(
            EvidencePacket(
                source="pubmed",
                query="KRAS G12C",
                items=[],
                source_health="degraded",
                missing_reason="no_pubmed_hits",
            )
        ),
        pubchem_client=StubPubChemClient(
            EvidencePacket(
                source="pubchem",
                query="KRAS G12C",
                items=[],
                source_health="degraded",
                missing_reason="no_pubchem_hits",
            )
        ),
        chembl_client=StubChEMBLClient(
            EvidencePacket(
                source="chembl",
                query="KRAS G12C",
                items=[],
                source_health="degraded",
                missing_reason="no_chembl_hits",
            )
        ),
        clinicaltrials_client=StubClinicalTrialsClient(
            packet=EvidencePacket(
                source="clinicaltrials",
                query="KRAS G12C",
                items=[],
                source_health="degraded",
                missing_reason="no_clinicaltrials_hits",
            )
        ),
        openfda_client=StubOpenFDAClient(
            EvidencePacket(
                source="openfda",
                query="KRAS G12C",
                items=[],
                source_health="degraded",
                missing_reason="no_openfda_hits",
            )
        ),
    )

    bundle = coordinator.collect_evidence(
        question="임상 근거가 있나?",
        question_type="regulatory",
        target="KRAS G12C",
    )

    assert bundle.items == []
    assert bundle.source_health == "degraded"
    assert bundle.partial_failures == []
    assert bundle.missing_sources == [
        "pubchem",
        "chembl",
        "pubmed",
        "clinicaltrials",
        "openfda",
    ]


def test_evidence_coordinator_builds_domain_packets_for_expert_workflows() -> None:
    coordinator = EvidenceCoordinator(
        pubmed_client=StubPubMedClient(
            EvidencePacket(
                source="pubmed",
                query="KRAS G12C",
                items=[
                    make_item(
                        "pubmed",
                        "12345",
                        "PubMed toxicity evidence for ABC-101",
                        8.5,
                        "https://pubmed.ncbi.nlm.nih.gov/12345/",
                    )
                ],
                source_health="ok",
            )
        ),
        pubchem_client=StubPubChemClient(
            EvidencePacket(
                source="pubchem",
                query="ABC-101",
                items=[
                    make_item(
                        "pubchem",
                        "2244",
                        "PubChem summary for ABC-101",
                        6.0,
                        "https://pubchem.ncbi.nlm.nih.gov/compound/2244",
                    )
                ],
                source_health="ok",
            )
        ),
        chembl_client=StubChEMBLClient(
            EvidencePacket(
                source="chembl",
                query="ABC-101",
                items=[],
                source_health="degraded",
                missing_reason="no_chembl_hits",
            )
        ),
        clinicaltrials_client=StubClinicalTrialsClient(
            packet=EvidencePacket(
                source="clinicaltrials",
                query="ABC-101",
                items=[
                    make_item(
                        "clinicaltrials",
                        "NCT01234567",
                        "Phase 2 study for ABC-101",
                        7.0,
                        "https://clinicaltrials.gov/study/NCT01234567",
                    )
                ],
                source_health="ok",
            )
        ),
        openfda_client=StubOpenFDAClient(
            EvidencePacket(
                source="openfda",
                query="ABC-101",
                items=[],
                source_health="degraded",
                missing_reason="openfda_request_failed",
                diagnostics={"error": "503"},
            )
        ),
    )

    bundle = coordinator.collect_evidence(
        question="전체 evidence를 모아줘",
        question_type="regulatory",
        target="KRAS G12C",
        compound_name="ABC-101",
    )
    packets = coordinator.build_domain_packets(bundle)

    assert packets["chemistry"].source == "chemistry"
    assert [item.source for item in packets["chemistry"].items] == ["pubchem"]
    assert packets["chemistry"].source_health == "partial"
    assert packets["safety"].source == "safety"
    assert [item.source for item in packets["safety"].items] == ["pubmed"]
    assert packets["safety"].source_health == "ok"
    assert packets["clinical_regulatory"].source == "clinical_regulatory"
    assert [item.source for item in packets["clinical_regulatory"].items] == ["clinicaltrials"]
    assert packets["clinical_regulatory"].source_health == "partial"
    assert packets["clinical_regulatory"].diagnostics == {"sources": "openfda"}


def test_evidence_coordinator_uses_planner_selected_pubmed_queries_when_available() -> None:
    planned_query = '"ABC-101" AND "cardiotoxicity"'
    fallback_query = '"ABC-101"'
    planner = StubPubMedQueryPlanner(
        PubMedQueryAgentResult(
            question_type="safety",
            selected_query=planned_query,
            candidate_queries=[planned_query, fallback_query],
            dry_run_results=[],
            reasoning="Use the most specific toxicity term first.",
            validation_issues=[],
            revision_attempts=0,
            used_llm=True,
            fallback_used=False,
        )
    )
    planned_packet = EvidencePacket(
        source="pubmed",
        query=planned_query,
        items=[
            make_item(
                "pubmed",
                "98765",
                "Planned PubMed evidence for ABC-101",
                9.1,
                "https://pubmed.ncbi.nlm.nih.gov/98765/",
            )
        ],
        source_health="ok",
    )
    pubmed_client = StubPubMedClient(
        EvidencePacket(source="pubmed", query="rule-based", items=[]),
        packet_from_queries=planned_packet,
    )
    coordinator = EvidenceCoordinator(
        pubmed_client=pubmed_client,
        pubmed_query_planner=planner,
        pubchem_client=StubPubChemClient(EvidencePacket(source="pubchem", query="", items=[])),
        chembl_client=StubChEMBLClient(EvidencePacket(source="chembl", query="", items=[])),
        clinicaltrials_client=StubClinicalTrialsClient(
            packet=EvidencePacket(source="clinicaltrials", query="", items=[])
        ),
        openfda_client=StubOpenFDAClient(EvidencePacket(source="openfda", query="", items=[])),
    )

    bundle = coordinator.collect_evidence(
        question="이 화합물의 심장독성 근거를 찾아줘",
        question_type="safety_pk",
        target="KRAS G12C",
        compound_name="ABC-101",
    )

    assert planner.calls == [
        {
            "question": "이 화합물의 심장독성 근거를 찾아줘",
            "question_type": "safety",
            "target": "KRAS G12C",
            "compound_name": "ABC-101",
            "prediction_flags": None,
        }
    ]
    assert pubmed_client.collect_calls == []
    assert pubmed_client.collect_from_queries_calls == [
        {
            "question": "이 화합물의 심장독성 근거를 찾아줘",
            "queries": [planned_query, fallback_query],
            "target": "KRAS G12C",
            "retmax": 10,
            "top_k": 5,
        }
    ]
    assert bundle.packets["pubmed"].query == planned_query
    assert bundle.packets["pubmed"].diagnostics["query_planner_used"] == "true"
    assert bundle.packets["pubmed"].diagnostics["query_planner_question_type"] == "safety"
    assert bundle.packets["pubmed"].diagnostics["query_planner_best_hit_count"] == "0"


def test_evidence_coordinator_falls_back_to_normalized_rule_based_pubmed_query_when_planner_fails() -> None:
    planner = StubPubMedQueryPlanner(exc=RuntimeError("bedrock unavailable"))
    fallback_packet = EvidencePacket(
        source="pubmed",
        query="rule-based regulatory query",
        items=[],
        source_health="degraded",
        missing_reason="no_pubmed_hits",
    )
    pubmed_client = StubPubMedClient(fallback_packet)
    coordinator = EvidenceCoordinator(
        pubmed_client=pubmed_client,
        pubmed_query_planner=planner,
        pubchem_client=StubPubChemClient(EvidencePacket(source="pubchem", query="", items=[])),
        chembl_client=StubChEMBLClient(EvidencePacket(source="chembl", query="", items=[])),
        clinicaltrials_client=StubClinicalTrialsClient(
            packet=EvidencePacket(source="clinicaltrials", query="", items=[])
        ),
        openfda_client=StubOpenFDAClient(EvidencePacket(source="openfda", query="", items=[])),
    )

    bundle = coordinator.collect_evidence(
        question="임상과 허가 관점에서 근거를 정리해줘",
        question_type="clinical_regulatory",
        target="KRAS G12C",
        compound_name="ABC-101",
    )

    assert planner.calls == [
        {
            "question": "임상과 허가 관점에서 근거를 정리해줘",
            "question_type": "regulatory",
            "target": "KRAS G12C",
            "compound_name": "ABC-101",
            "prediction_flags": None,
        }
    ]
    assert pubmed_client.collect_from_queries_calls == []
    assert pubmed_client.collect_calls == [
        {
            "question": "임상과 허가 관점에서 근거를 정리해줘",
            "question_type": "regulatory",
            "target": "KRAS G12C",
            "compound_name": "ABC-101",
            "retmax": 10,
            "top_k": 5,
        }
    ]
    assert bundle.packets["pubmed"].diagnostics["query_planner_error"] == "bedrock unavailable"
