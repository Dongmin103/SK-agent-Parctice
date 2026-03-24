from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from app.domain.models import EvidenceBundle


@dataclass(slots=True)
class WorkflowTraceEvent:
    stage: str
    message: str
    level: str = "info"
    details: dict[str, str] = field(default_factory=dict)


EventSink = Callable[[WorkflowTraceEvent], None]

_SOURCE_LABELS = {
    "pubchem": "PubChem",
    "chembl": "ChEMBL",
    "pubmed": "PubMed",
    "clinicaltrials": "ClinicalTrials.gov",
    "openfda": "openFDA",
}


def emit_trace(
    event_sink: EventSink | None,
    *,
    stage: str,
    message: str,
    level: str = "info",
    details: dict[str, str] | None = None,
) -> None:
    if event_sink is None:
        return
    event_sink(
        WorkflowTraceEvent(
            stage=stage,
            message=message,
            level=level,
            details=dict(details or {}),
        )
    )


def emit_evidence_trace_events(
    event_sink: EventSink | None,
    bundle: EvidenceBundle,
) -> None:
    if event_sink is None:
        return

    pubmed_packet = bundle.packets.get("pubmed")
    if pubmed_packet is not None:
        diagnostics = pubmed_packet.diagnostics
        selected_query = diagnostics.get("query_planner_selected_query")
        if diagnostics.get("query_planner_used") == "true" and selected_query:
            emit_trace(
                event_sink,
                stage="evidence",
                message=f"PubMed planner selected query: {selected_query}",
                details={"source": "pubmed", "selected_query": selected_query},
            )
        best_hit_count = diagnostics.get("query_planner_best_hit_count")
        if best_hit_count:
            emit_trace(
                event_sink,
                stage="evidence",
                message=f"PubMed dry run hits: {best_hit_count}",
                details={"source": "pubmed", "dry_run_hits": best_hit_count},
            )
        planner_error = diagnostics.get("query_planner_error")
        if planner_error:
            emit_trace(
                event_sink,
                stage="evidence",
                message=f"PubMed planner fallback: {planner_error}",
                level="warning",
                details={"source": "pubmed", "error": planner_error},
            )

    for source, packet in bundle.packets.items():
        error = packet.diagnostics.get("error")
        if not error:
            continue
        label = _SOURCE_LABELS.get(source, source)
        emit_trace(
            event_sink,
            stage="evidence",
            message=f"{label} degraded: {error}",
            level="warning",
            details={"source": source, "error": error},
        )
