from __future__ import annotations

import os
from typing import Any
from xml.etree import ElementTree as ET

import streamlit as st
import streamlit.components.v1 as components

from app.ui.client import UiApiClient, UiApiError
from app.ui.presenters import ConsultViewModel, ExecutiveViewModel, build_consult_view_model, build_executive_view_model
from app.ui.theme import build_theme_css

SVG_NAMESPACE = "http://www.w3.org/2000/svg"
XML_NAMESPACE = "http://www.w3.org/XML/1998/namespace"
_ALLOWED_SVG_TAGS = frozenset(
    {
        "svg",
        "g",
        "defs",
        "path",
        "rect",
        "circle",
        "ellipse",
        "line",
        "polyline",
        "polygon",
        "text",
        "tspan",
        "title",
        "desc",
    }
)
_ALLOWED_SVG_ATTRS = frozenset(
    {
        "baseProfile",
        "class",
        "cx",
        "cy",
        "d",
        "fill",
        "fill-opacity",
        "fill-rule",
        "font-family",
        "font-size",
        "font-weight",
        "height",
        "opacity",
        "points",
        "preserveAspectRatio",
        "r",
        "rx",
        "ry",
        "stroke",
        "stroke-linecap",
        "stroke-linejoin",
        "stroke-opacity",
        "stroke-width",
        "style",
        "text-anchor",
        "transform",
        "version",
        "viewBox",
        "width",
        "x",
        "x1",
        "x2",
        "xml:space",
        "y",
        "y1",
        "y2",
    }
)
_ALLOWED_STYLE_PROPERTIES = frozenset(
    {
        "fill",
        "fill-opacity",
        "fill-rule",
        "font-family",
        "font-size",
        "font-weight",
        "opacity",
        "stroke",
        "stroke-linecap",
        "stroke-linejoin",
        "stroke-opacity",
        "stroke-width",
        "text-anchor",
    }
)
_TEXT_SVG_TAGS = frozenset({"title", "desc", "text", "tspan"})

ET.register_namespace("", SVG_NAMESPACE)
ET.register_namespace("xml", XML_NAMESPACE)


# Shared UI metadata and local runtime defaults.
APP_TITLE = "SK Agentic AI Workbench"
DEFAULT_API_BASE_URL = os.environ.get("UI_API_BASE_URL", "http://127.0.0.1:8000")
API_TIMEOUT_SECONDS = 180.0


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="SK",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_theme()
    _initialize_state()

    st.markdown(
        """
        <section class="hero">
          <p class="eyebrow">Streamlit Control Tower</p>
          <h1>SK Agentic AI Workbench</h1>
          <p class="hero-copy">
            Consult와 Executive 흐름을 같은 런타임 위에서 검증하는 로컬 워크벤치입니다.
            입력은 FastAPI로 보내고, 결과는 근거와 의사결정 단위로 다시 정렬해 보여줍니다.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Runtime")
        api_base_url = st.text_input("FastAPI Base URL", value=st.session_state["api_base_url"])
        st.session_state["api_base_url"] = api_base_url
        st.caption("예: http://127.0.0.1:8000")
        st.markdown("---")
        screen = st.radio("Workspace", options=["Consult", "Executive"], horizontal=False)
        st.markdown("---")
        st.markdown(
            """
            **Runbook**

            1. FastAPI 서버를 먼저 실행합니다.
            2. Streamlit에서 요청을 보냅니다.
            3. 결과는 signal / finding / citation 단위로 바로 검토합니다.
            """
        )

    if screen == "Consult":
        _render_consult_screen(api_base_url)
    else:
        _render_executive_screen(api_base_url)


def _initialize_state() -> None:
    st.session_state.setdefault("api_base_url", DEFAULT_API_BASE_URL)
    st.session_state.setdefault("consult_result", None)
    st.session_state.setdefault("consult_error", None)
    st.session_state.setdefault("consult_trace", [])
    st.session_state.setdefault("executive_result", None)
    st.session_state.setdefault("executive_error", None)
    st.session_state.setdefault("executive_trace", [])


def _get_api_client(base_url: str) -> UiApiClient:
    return UiApiClient(base_url=base_url, timeout_seconds=API_TIMEOUT_SECONDS)


def _render_consult_screen(api_base_url: str) -> None:
    st.markdown("## Consult")
    st.caption("질문 기반 라우팅으로 필요한 전문가만 호출합니다.")

    with st.form("consult-form"):
        left, right = st.columns(2)
        with left:
            compound_name = st.text_input("Compound Name", value="ABC-101")
            target = st.text_input("Target", value="KRAS G12C")
        with right:
            smiles = st.text_area("SMILES", value="CCO", height=120)
            question = st.text_area("Question", value="이 화합물의 hERG 위험은?", height=120)
        submitted = st.form_submit_button("Run Consult", width="stretch")

    if submitted:
        trace_entries: list[dict[str, Any]] = []
        try:
            with st.status("Consult is running...", expanded=True) as status:
                for chunk in _get_api_client(api_base_url).stream_consult(
                    smiles=smiles,
                    target=target,
                    question=question,
                    compound_name=compound_name or None,
                ):
                    if chunk.get("type") == "trace":
                        trace = chunk.get("trace")
                        if isinstance(trace, dict):
                            trace_entries.append(trace)
                            status.write(_format_trace_entry(trace))
                            status.update(
                                label=_status_label_from_trace("Consult", trace),
                                state="running",
                            )
                        continue
                    if chunk.get("type") == "result":
                        response = chunk.get("result")
                        if isinstance(response, dict):
                            status.update(label="Consult completed", state="complete")
                            break
                else:
                    response = None
        except Exception as exc:
            api_error = _coerce_ui_api_error(exc)
            if api_error is None:
                raise
            st.session_state["consult_error"] = api_error
            st.session_state["consult_result"] = None
            st.session_state["consult_trace"] = trace_entries
        else:
            st.session_state["consult_error"] = None
            st.session_state["consult_trace"] = trace_entries
            st.session_state["consult_result"] = response

    error = st.session_state.get("consult_error")
    if error is not None:
        _render_error(error)

    trace_entries = st.session_state.get("consult_trace")
    if isinstance(trace_entries, list) and trace_entries:
        _render_trace_entries(trace_entries)

    payload = st.session_state.get("consult_result")
    if not isinstance(payload, dict):
        st.info("Consult 결과는 첫 실행 후 여기에 렌더링됩니다.")
        return

    view_model = build_consult_view_model(payload)
    _render_consult_result(view_model)


def _render_executive_screen(api_base_url: str) -> None:
    st.markdown("## Executive")
    st.caption("세 전문가를 모두 실행한 뒤 CEO synthesis 결과를 검토합니다.")

    with st.form("executive-form"):
        left, right = st.columns(2)
        with left:
            compound_name = st.text_input("Compound Name", value="ABC-101", key="executive-compound-name")
            target = st.text_input("Target", value="KRAS G12C", key="executive-target")
        with right:
            smiles = st.text_area("SMILES", value="N#CC1=CC=CC=C1", height=120, key="executive-smiles")
        submitted = st.form_submit_button("Run Executive", width="stretch")

    if submitted:
        trace_entries: list[dict[str, Any]] = []
        try:
            with st.status("Executive is running...", expanded=True) as status:
                for chunk in _get_api_client(api_base_url).stream_executive(
                    smiles=smiles,
                    target=target,
                    compound_name=compound_name or None,
                ):
                    if chunk.get("type") == "trace":
                        trace = chunk.get("trace")
                        if isinstance(trace, dict):
                            trace_entries.append(trace)
                            status.write(_format_trace_entry(trace))
                            status.update(
                                label=_status_label_from_trace("Executive", trace),
                                state="running",
                            )
                        continue
                    if chunk.get("type") == "result":
                        response = chunk.get("result")
                        if isinstance(response, dict):
                            status.update(label="Executive completed", state="complete")
                            break
                else:
                    response = None
        except Exception as exc:
            api_error = _coerce_ui_api_error(exc)
            if api_error is None:
                raise
            st.session_state["executive_error"] = api_error
            st.session_state["executive_result"] = None
            st.session_state["executive_trace"] = trace_entries
        else:
            st.session_state["executive_error"] = None
            st.session_state["executive_trace"] = trace_entries
            st.session_state["executive_result"] = response

    error = st.session_state.get("executive_error")
    if error is not None:
        _render_error(error)

    trace_entries = st.session_state.get("executive_trace")
    if isinstance(trace_entries, list) and trace_entries:
        _render_trace_entries(trace_entries)

    payload = st.session_state.get("executive_result")
    if not isinstance(payload, dict):
        st.info("Executive 결과는 첫 실행 후 여기에 렌더링됩니다.")
        return

    view_model = build_executive_view_model(payload)
    _render_executive_result(view_model)


def _render_consult_result(view_model: ConsultViewModel) -> None:
    metric_columns = st.columns(4)
    metric_columns[0].metric("Selected Agents", len(view_model.selected_agents))
    metric_columns[1].metric("Signals", len(view_model.prediction_rows))
    metric_columns[2].metric("Citations", view_model.citation_count)
    metric_columns[3].metric("Review", view_model.review_label)

    st.markdown(
        f"""
        <div class="callout-card">
          <p class="callout-label">Routing Reason</p>
          <p class="callout-copy">{view_model.routing_reason}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    answer_col, summary_col = st.columns([1.4, 1.0])
    with answer_col:
        st.markdown("### Consulting Answer")
        st.write(view_model.answer)
    with summary_col:
        st.markdown("### Selected Agents")
        for agent_id in view_model.selected_agents:
            st.markdown(f"- `{agent_id}`")
        if view_model.missing_signals:
            st.markdown("### Missing Signals")
            for signal in view_model.missing_signals:
                st.markdown(f"- `{signal}`")

    tabs = st.tabs(["Signals", "Findings", "Citations"])
    with tabs[0]:
        _render_prediction_table(view_model.prediction_rows)
    with tabs[1]:
        _render_findings(view_model.findings)
    with tabs[2]:
        _render_citations(view_model.citations)


def _render_executive_result(view_model: ExecutiveViewModel) -> None:
    metric_columns = st.columns(4)
    metric_columns[0].metric("Decision", view_model.decision_label)
    metric_columns[1].metric("Signals", len(view_model.prediction_rows))
    metric_columns[2].metric("Evidence Sources", len(view_model.evidence_sources))
    metric_columns[3].metric("Review", "Required" if view_model.review_required else "Cleared")

    lead_col, molecule_col = st.columns([1.3, 0.9])
    with lead_col:
        st.markdown("### Executive Summary")
        st.write(view_model.summary)
        st.markdown("### Decision Rationale")
        st.write(view_model.rationale)
        st.markdown("### Next Steps")
        for step in view_model.next_steps:
            st.markdown(f"- {step}")
        if view_model.canonical_smiles:
            st.caption(f"Canonical SMILES: {view_model.canonical_smiles}")
    with molecule_col:
        st.markdown("### Molecule")
        if view_model.molecule_svg:
            sanitized_svg = _sanitize_svg_markup(view_model.molecule_svg)
            if sanitized_svg is None:
                st.warning("분자 SVG를 안전하게 렌더링할 수 없습니다.")
            else:
                components.html(
                    f"""
                    <div style="background: linear-gradient(180deg, #fff7ea 0%, #f7f4eb 100%);
                                border: 1px solid rgba(34,45,39,0.12);
                                border-radius: 20px;
                                padding: 18px;">
                      {sanitized_svg}
                    </div>
                    """,
                    height=320,
                )
        else:
            st.info("분자 SVG가 아직 없습니다.")

    tabs = st.tabs(["Signals", "Evidence", "Findings", "Citations"])
    with tabs[0]:
        _render_prediction_table(view_model.prediction_rows)
    with tabs[1]:
        _render_evidence_sources(view_model.evidence_sources)
    with tabs[2]:
        _render_findings(view_model.findings)
        if view_model.review_reasons:
            st.markdown("### Review Reasons")
            for reason in view_model.review_reasons:
                st.markdown(f"- {reason}")
    with tabs[3]:
        _render_citations(view_model.citations)


def _render_prediction_table(rows: list[Any]) -> None:
    if not rows:
        st.info("표시할 prediction signal이 없습니다.")
        return
    st.dataframe(
        [
            {
                "Signal": row.label,
                "Value": row.value,
                "Meta": row.meta,
            }
            for row in rows
        ],
        width="stretch",
        hide_index=True,
    )


def _render_findings(findings: list[Any]) -> None:
    if not findings:
        st.info("표시할 agent finding이 없습니다.")
        return

    for finding in findings:
        with st.expander(str(finding.agent_id), expanded=True):
            st.markdown(f"**Confidence:** {finding.confidence:.2f}")
            st.write(finding.summary)
            if finding.risks:
                st.markdown("**Risks**")
                for risk in finding.risks:
                    st.markdown(f"- {risk}")
            if finding.recommendations:
                st.markdown("**Recommendations**")
                for recommendation in finding.recommendations:
                    st.markdown(f"- {recommendation}")
            if finding.citations:
                st.markdown("**Citations**")
                for citation in finding.citations:
                    st.markdown(f"- [{citation}]({citation})")


def _render_evidence_sources(rows: list[Any]) -> None:
    if not rows:
        st.info("표시할 evidence source가 없습니다.")
        return
    st.dataframe(
        [
            {
                "Source": row.source,
                "Health": row.health,
                "Items": row.item_count,
                "Query": row.query,
                "Missing Reason": row.missing_reason or "",
            }
            for row in rows
        ],
        width="stretch",
        hide_index=True,
    )


def _render_citations(citations: list[str]) -> None:
    if not citations:
        st.info("표시할 citation이 없습니다.")
        return
    for citation in citations:
        st.markdown(f"- [{citation}]({citation})")


def _render_trace_entries(entries: list[dict[str, Any]]) -> None:
    with st.expander("Runtime Trace", expanded=True):
        for entry in entries:
            st.markdown(f"- {_format_trace_entry(entry)}")


def _render_error(error: UiApiError) -> None:
    detail_text = ""
    if error.details:
        detail_text = f"\n\nDetails: `{error.details}`"
    st.error(f"[{error.code}] {error.message}{detail_text}")


def _coerce_ui_api_error(exc: Exception) -> UiApiError | None:
    if isinstance(exc, UiApiError):
        return exc
    if exc.__class__.__name__ != "UiApiError":
        return None

    code = getattr(exc, "code", None)
    message = getattr(exc, "message", None)
    if not isinstance(code, str) or not isinstance(message, str):
        return None

    status_code = getattr(exc, "status_code", None)
    if not isinstance(status_code, int):
        status_code = None

    return UiApiError(
        code=code,
        message=message,
        details=getattr(exc, "details", None),
        status_code=status_code,
    )


def _format_trace_entry(entry: dict[str, Any]) -> str:
    stage = str(entry.get("stage", "trace")).strip() or "trace"
    level = str(entry.get("level", "info")).strip() or "info"
    message = str(entry.get("message", "")).strip()
    if not message:
        return f"`{stage}` [{level}]"
    return f"`{stage}` [{level}] {message}"


def _status_label_from_trace(prefix: str, entry: dict[str, Any]) -> str:
    message = str(entry.get("message", "")).strip()
    if not message:
        return f"{prefix} is running..."
    return f"{prefix}: {message}"


def _sanitize_svg_markup(svg_markup: str) -> str | None:
    try:
        root = ET.fromstring(svg_markup)
    except ET.ParseError:
        return None

    if _local_svg_name(root.tag) != "svg":
        return None

    sanitized_root = _sanitize_svg_element(root)
    if sanitized_root is None:
        return None

    return ET.tostring(sanitized_root, encoding="unicode", method="xml")


def _sanitize_svg_element(element: ET.Element) -> ET.Element | None:
    tag_name = _local_svg_name(element.tag)
    if tag_name not in _ALLOWED_SVG_TAGS:
        return None

    sanitized = ET.Element(f"{{{SVG_NAMESPACE}}}{tag_name}")
    for raw_name, raw_value in element.attrib.items():
        attribute_name = _local_svg_attribute_name(raw_name)
        safe_value = _sanitize_svg_attribute(attribute_name, raw_value)
        if safe_value is None:
            continue
        if attribute_name == "xml:space":
            sanitized.set(f"{{{XML_NAMESPACE}}}space", safe_value)
        else:
            sanitized.set(attribute_name, safe_value)

    if tag_name in _TEXT_SVG_TAGS and element.text:
        sanitized.text = element.text

    for child in list(element):
        safe_child = _sanitize_svg_element(child)
        if safe_child is None:
            continue
        sanitized.append(safe_child)
        if child.tail and tag_name in _TEXT_SVG_TAGS:
            safe_child.tail = child.tail

    return sanitized


def _sanitize_svg_attribute(name: str, value: str) -> str | None:
    if not value:
        return None
    if name.startswith("on") or name not in _ALLOWED_SVG_ATTRS:
        return None

    stripped = value.strip()
    if not stripped:
        return None
    if name == "style":
        sanitized_style = _sanitize_svg_style(stripped)
        return sanitized_style or None
    if _contains_unsafe_svg_value(stripped):
        return None
    return stripped


def _sanitize_svg_style(style_value: str) -> str:
    safe_declarations: list[str] = []
    for declaration in style_value.split(";"):
        if ":" not in declaration:
            continue
        property_name, property_value = declaration.split(":", 1)
        property_name = property_name.strip().lower()
        property_value = property_value.strip()
        if property_name not in _ALLOWED_STYLE_PROPERTIES:
            continue
        if not property_value or _contains_unsafe_svg_value(property_value):
            continue
        safe_declarations.append(f"{property_name}:{property_value}")
    return ";".join(safe_declarations)


def _contains_unsafe_svg_value(value: str) -> bool:
    lowered = value.lower()
    return any(token in lowered for token in ("javascript:", "vbscript:", "data:", "expression(", "url("))


def _local_svg_name(name: str) -> str:
    if "}" in name:
        return name.rsplit("}", 1)[1]
    return name


def _local_svg_attribute_name(name: str) -> str:
    if name == f"{{{XML_NAMESPACE}}}space":
        return "xml:space"
    return _local_svg_name(name)


def _apply_theme() -> None:
    st.markdown(
        f"<style>{build_theme_css()}</style>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
