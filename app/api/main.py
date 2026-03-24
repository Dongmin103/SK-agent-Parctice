from __future__ import annotations

import json
import logging
from dataclasses import asdict
from queue import Queue
from threading import Thread
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.api.dependencies import build_consult_workflow, build_executive_workflow
from app.api.errors import AppServiceError, install_exception_handlers
from app.api.schemas import (
    ConsultRequest,
    ConsultResponse,
    ErrorResponse,
    ExecutiveRequest,
    ExecutiveResponse,
)
from app.domain.compound import InvalidSmilesError
from app.api.settings import AppSettings, load_settings
from app.api.stubs import build_stub_consult_workflow, build_stub_executive_workflow
from app.workflows.consult import ConsultReport
from app.workflows.executive import ExecutiveReport
from app.workflows.tracing import WorkflowTraceEvent

LOGGER = logging.getLogger(__name__)
_STREAM_DONE = object()


def create_app(
    *,
    consult_workflow: Any | None = None,
    executive_workflow: Any | None = None,
    settings: AppSettings | None = None,
) -> FastAPI:
    resolved_settings = settings or load_settings()
    app = FastAPI()
    install_exception_handlers(app)

    consult_workflow = consult_workflow or build_consult_workflow(resolved_settings)
    executive_workflow = executive_workflow or build_executive_workflow(resolved_settings)

    @app.post("/api/reports/consult", response_model=ConsultResponse)
    def post_consult(request: ConsultRequest) -> dict[str, object]:
        report = consult_workflow.run(
            smiles=request.smiles,
            target=request.target,
            question=request.question,
            compound_name=request.compound_name,
        )
        return _serialize_consult_report(report)

    @app.post("/api/reports/consult/stream")
    def post_consult_stream(request: ConsultRequest) -> StreamingResponse:
        return StreamingResponse(
            _stream_workflow(
                lambda event_sink: consult_workflow.run(
                    smiles=request.smiles,
                    target=request.target,
                    question=request.question,
                    compound_name=request.compound_name,
                    event_sink=event_sink,
                ),
                _serialize_consult_report,
            ),
            media_type="application/x-ndjson",
        )

    @app.post("/api/reports/executive", response_model=ExecutiveResponse)
    def post_executive(request: ExecutiveRequest) -> dict[str, object]:
        report = executive_workflow.run(
            smiles=request.smiles,
            target=request.target,
            compound_name=request.compound_name,
        )
        return _serialize_executive_report(report)

    @app.post("/api/reports/executive/stream")
    def post_executive_stream(request: ExecutiveRequest) -> StreamingResponse:
        return StreamingResponse(
            _stream_workflow(
                lambda event_sink: executive_workflow.run(
                    smiles=request.smiles,
                    target=request.target,
                    compound_name=request.compound_name,
                    event_sink=event_sink,
                ),
                _serialize_executive_report,
            ),
            media_type="application/x-ndjson",
        )

    return app


def create_runtime_app() -> FastAPI:
    settings = load_settings()
    if settings.use_stub_workflows:
        return create_app(
            consult_workflow=build_stub_consult_workflow(),
            executive_workflow=build_stub_executive_workflow(),
            settings=settings,
        )
    return create_app(settings=settings)


def _serialize_consult_report(report: ConsultReport) -> dict[str, object]:
    return {
        "selected_agents": list(report.selected_agents),
        "routing_reason": report.routing_reason,
        "predictions": _serialize_prediction_bundle(report.predictions),
        "agent_findings": [_serialize_agent_finding(finding) for finding in report.agent_findings],
        "consulting_answer": report.consulting_answer,
        "citations": list(report.citations),
        "review_required": report.review_required,
    }


def _serialize_executive_report(report: ExecutiveReport) -> dict[str, object]:
    return {
        "canonical_smiles": report.canonical_smiles,
        "molecule_svg": report.molecule_svg,
        "predictions": asdict(report.predictions),
        "evidence_bundle": asdict(report.evidence_bundle),
        "agent_findings": [asdict(finding) for finding in report.agent_findings],
        "executive_summary": report.executive_summary,
        "executive_decision": asdict(report.executive_decision),
        "citations": list(report.citations),
        "review_required": report.review_required,
        "review_reasons": list(report.review_reasons),
    }


def _serialize_prediction_bundle(bundle: Any) -> dict[str, object]:
    return asdict(bundle)


def _serialize_agent_finding(finding: Any) -> dict[str, object]:
    return asdict(finding)


def _stream_workflow(
    run_workflow,
    serialize_report,
):
    queue: Queue[object] = Queue()

    def _emit_trace(event: WorkflowTraceEvent) -> None:
        queue.put(
            {
                "type": "trace",
                "trace": _serialize_trace_event(event),
            }
        )

    def _worker() -> None:
        try:
            report = run_workflow(_emit_trace)
        except Exception as exc:
            queue.put(
                {
                    "type": "error",
                    "error": _serialize_stream_error(exc),
                }
            )
        else:
            queue.put(
                {
                    "type": "result",
                    "result": serialize_report(report),
                }
            )
        finally:
            queue.put(_STREAM_DONE)

    Thread(target=_worker, daemon=True).start()

    def _iterator():
        while True:
            item = queue.get()
            if item is _STREAM_DONE:
                break
            yield _encode_stream_chunk(item)

    return _iterator()


def _serialize_trace_event(event: WorkflowTraceEvent) -> dict[str, object]:
    return {
        "stage": event.stage,
        "message": event.message,
        "level": event.level,
        "details": dict(event.details),
    }


def _serialize_stream_error(exc: Exception) -> dict[str, object]:
    if isinstance(exc, InvalidSmilesError):
        payload = ErrorResponse(
            error={
                "code": "invalid_smiles",
                "message": str(exc),
            }
        )
        return payload.model_dump(exclude_none=True)["error"]
    if isinstance(exc, AppServiceError):
        payload = ErrorResponse(
            error={
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            }
        )
        return payload.model_dump(exclude_none=True)["error"]
    if isinstance(exc, ValueError):
        payload = ErrorResponse(
            error={
                "code": "bad_request",
                "message": str(exc),
            }
        )
        return payload.model_dump(exclude_none=True)["error"]

    LOGGER.exception("Unhandled streamed API error", exc_info=exc)
    payload = ErrorResponse(
        error={
            "code": "internal_error",
            "message": "An unexpected error occurred.",
        }
    )
    return payload.model_dump(exclude_none=True)["error"]


def _encode_stream_chunk(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


app = create_runtime_app()
