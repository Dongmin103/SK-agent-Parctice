from __future__ import annotations

from dataclasses import replace
from typing import Any

from app.domain.models import EvidenceBundle, EvidenceItem, EvidencePacket, PubMedQueryInput


class EvidenceCoordinator:
    """Collect evidence across all external evidence sources."""

    SOURCE_ORDER = ("pubchem", "chembl", "pubmed", "clinicaltrials", "openfda")
    DOMAIN_SOURCE_MAP = {
        "chemistry": ("pubchem", "chembl"),
        "safety": ("pubmed",),
        "clinical_regulatory": ("clinicaltrials", "openfda"),
    }
    PUBMED_QUESTION_TYPE_MAP = {
        "safety": "safety",
        "pk": "pk",
        "regulatory": "regulatory",
        "complex": "complex",
        "safety_pk": "complex",
        "structure_sar": "complex",
        "clinical_regulatory": "regulatory",
        "multi_expert": "complex",
    }
    PK_HINTS = (
        "pk",
        "pd",
        "ddi",
        "drug-drug",
        "interaction",
        "clearance",
        "half-life",
        "exposure",
        "pharmacokinetic",
        "pharmacokinetics",
        "상호작용",
        "반감기",
        "노출",
    )

    def __init__(
        self,
        *,
        pubmed_client: Any,
        pubmed_query_planner: Any | None = None,
        pubchem_client: Any,
        chembl_client: Any,
        clinicaltrials_client: Any,
        openfda_client: Any,
    ) -> None:
        self.pubmed_client = pubmed_client
        self.pubmed_query_planner = pubmed_query_planner
        self.pubchem_client = pubchem_client
        self.chembl_client = chembl_client
        self.clinicaltrials_client = clinicaltrials_client
        self.openfda_client = openfda_client

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
        packets = {
            "pubchem": self._collect_source(
                source="pubchem",
                client=self.pubchem_client,
                method_name="collect_pubchem_evidence",
                question=question,
                question_type=question_type,
                target=target,
                compound_name=compound_name,
                retmax=retmax,
                top_k=top_k,
            ),
            "chembl": self._collect_source(
                source="chembl",
                client=self.chembl_client,
                method_name="collect_chembl_evidence",
                question=question,
                question_type=question_type,
                target=target,
                compound_name=compound_name,
                retmax=retmax,
                top_k=top_k,
            ),
            "pubmed": self._collect_pubmed_source(
                question=question,
                question_type=question_type,
                target=target,
                compound_name=compound_name,
                retmax=retmax,
                top_k=top_k,
            ),
            "clinicaltrials": self._collect_source(
                source="clinicaltrials",
                client=self.clinicaltrials_client,
                method_name="collect_clinicaltrials_evidence",
                question=question,
                question_type=question_type,
                target=target,
                compound_name=compound_name,
                retmax=retmax,
                top_k=top_k,
            ),
            "openfda": self._collect_source(
                source="openfda",
                client=self.openfda_client,
                method_name="collect_openfda_evidence",
                question=question,
                question_type=question_type,
                target=target,
                compound_name=compound_name,
                retmax=retmax,
                top_k=top_k,
            ),
        }

        items = self._merge_items(packets, top_k=top_k)
        missing_sources = [
            source
            for source in self.SOURCE_ORDER
            if source in packets and not packets[source].items
        ]
        partial_failures = [
            source
            for source in self.SOURCE_ORDER
            if source in packets and packets[source].diagnostics.get("error")
        ]

        return EvidenceBundle(
            query=question,
            packets=packets,
            items=items,
            source_health=self._bundle_health(packets),
            missing_sources=missing_sources,
            partial_failures=partial_failures,
        )

    def build_domain_packets(self, bundle: EvidenceBundle) -> dict[str, EvidencePacket]:
        packets: dict[str, EvidencePacket] = {}
        for domain, sources in self.DOMAIN_SOURCE_MAP.items():
            selected = [bundle.packets[source] for source in sources if source in bundle.packets]
            items: list[EvidenceItem] = []
            failed_sources: list[str] = []
            for packet in selected:
                items.extend(packet.items)
                if packet.diagnostics.get("error"):
                    failed_sources.append(packet.source)

            items.sort(key=lambda item: item.score, reverse=True)
            source_health = self._bundle_health({packet.source: packet for packet in selected})
            packets[domain] = EvidencePacket(
                source=domain,
                query=bundle.query,
                items=items,
                source_health=source_health,
                missing_reason=None if items else f"no_{domain}_hits",
                diagnostics={"sources": ",".join(failed_sources)} if failed_sources else {},
            )
        return packets

    def _collect_source(
        self,
        *,
        source: str,
        client: Any,
        method_name: str,
        question: str,
        question_type: str,
        target: str | None,
        compound_name: str | None,
        retmax: int,
        top_k: int,
    ) -> EvidencePacket:
        if client is None:
            return EvidencePacket(
                source=source,
                query=compound_name or target or question,
                items=[],
                source_health="degraded",
                missing_reason=f"{source}_client_unavailable",
                diagnostics={"error": "client unavailable"},
            )

        method = getattr(client, method_name)
        try:
            return method(
                question,
                question_type,
                target=target,
                compound_name=compound_name,
                retmax=retmax,
                top_k=top_k,
            )
        except Exception as exc:
            return EvidencePacket(
                source=source,
                query=compound_name or target or question,
                items=[],
                source_health="degraded",
                missing_reason=f"{source}_request_failed",
                diagnostics={"error": str(exc)},
            )

    def _collect_pubmed_source(
        self,
        *,
        question: str,
        question_type: str,
        target: str | None,
        compound_name: str | None,
        retmax: int,
        top_k: int,
    ) -> EvidencePacket:
        normalized_question_type = self._normalize_pubmed_question_type(question_type, question)
        if self.pubmed_client is None:
            return EvidencePacket(
                source="pubmed",
                query=compound_name or target or question,
                items=[],
                source_health="degraded",
                missing_reason="pubmed_client_unavailable",
                diagnostics={"error": "client unavailable"},
            )

        if self.pubmed_query_planner is None:
            return self.pubmed_client.collect_pubmed_evidence(
                question,
                normalized_question_type,
                target=target,
                compound_name=compound_name,
                retmax=retmax,
                top_k=top_k,
            )

        request = PubMedQueryInput(
            question=question,
            question_type=normalized_question_type,
            target=target,
            compound_name=compound_name,
        )
        try:
            plan_result = self.pubmed_query_planner.plan(request)
            queries = self._ordered_query_candidates(
                plan_result.selected_query,
                plan_result.candidate_queries,
            )
            if not queries:
                raise ValueError("planner returned no candidate queries")
            packet = self.pubmed_client.collect_pubmed_evidence_from_queries(
                question,
                queries,
                target=target,
                retmax=retmax,
                top_k=top_k,
            )
            return self._with_pubmed_planner_diagnostics(
                packet,
                {
                    "query_planner_used": "true",
                    "query_planner_question_type": normalized_question_type,
                    "query_planner_selected_query": plan_result.selected_query,
                    "query_planner_best_hit_count": str(
                        self._selected_query_hit_count(
                            plan_result.selected_query,
                            plan_result.dry_run_results,
                        )
                    ),
                    "query_planner_fallback_used": str(plan_result.fallback_used).lower(),
                    "query_planner_revision_attempts": str(plan_result.revision_attempts),
                    "query_planner_validation_issues": ",".join(plan_result.validation_issues),
                },
            )
        except Exception as exc:
            packet = self.pubmed_client.collect_pubmed_evidence(
                question,
                normalized_question_type,
                target=target,
                compound_name=compound_name,
                retmax=retmax,
                top_k=top_k,
            )
            return self._with_pubmed_planner_diagnostics(
                packet,
                {
                    "query_planner_used": "false",
                    "query_planner_question_type": normalized_question_type,
                    "query_planner_error": str(exc),
                },
            )

    def _merge_items(
        self,
        packets: dict[str, EvidencePacket],
        *,
        top_k: int,
    ) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        for source in self.SOURCE_ORDER:
            packet = packets.get(source)
            if packet is None:
                continue
            items.extend(packet.items)
        items.sort(key=lambda item: item.score, reverse=True)
        return items[: max(top_k, 0) * len(self.SOURCE_ORDER)] if top_k else items

    def _bundle_health(self, packets: dict[str, EvidencePacket]) -> str:
        if not packets:
            return "degraded"

        health_values = [packet.source_health for packet in packets.values()]
        if all(value == "ok" for value in health_values):
            return "ok"
        if any(value == "ok" for value in health_values):
            return "partial"
        return "degraded"

    def _normalize_pubmed_question_type(self, question_type: str, question: str) -> str:
        mapped = self.PUBMED_QUESTION_TYPE_MAP.get(question_type)
        if mapped == "complex" and question_type == "safety_pk":
            lowered_question = question.lower()
            if any(hint in lowered_question for hint in self.PK_HINTS):
                return "pk"
            return "safety"
        if mapped is not None:
            return mapped
        return "complex"

    def _ordered_query_candidates(
        self,
        selected_query: str,
        candidate_queries: list[str],
    ) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for query in [selected_query, *candidate_queries]:
            clean = query.strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            ordered.append(clean)
        return ordered

    def _with_pubmed_planner_diagnostics(
        self,
        packet: EvidencePacket,
        diagnostics: dict[str, str],
    ) -> EvidencePacket:
        merged = dict(packet.diagnostics)
        merged.update(diagnostics)
        return replace(packet, diagnostics=merged)

    def _selected_query_hit_count(
        self,
        selected_query: str,
        dry_run_results: list[Any],
    ) -> int:
        for result in dry_run_results:
            if getattr(result, "query", "") == selected_query:
                hit_count = getattr(result, "hit_count", 0)
                return hit_count if isinstance(hit_count, int) else 0

        hit_counts = [
            getattr(result, "hit_count", 0)
            for result in dry_run_results
            if isinstance(getattr(result, "hit_count", None), int)
        ]
        return max(hit_counts, default=0)
