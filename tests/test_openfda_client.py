from __future__ import annotations

from json import JSONDecodeError
from urllib.error import HTTPError, URLError

from app.clients.openfda import OpenFDAClient
from app.domain.models import EvidenceItem


SAMPLE_LABEL_RESPONSE = {
    "meta": {"results": {"skip": 0, "limit": 10, "total": 1}},
    "results": [
        {
            "id": "openfda-imatinib-label",
            "generic_name": ["imatinib mesylate"],
            "brand_name": ["Gleevec"],
            "indications_and_usage": [
                "Imatinib mesylate is indicated for the treatment of chronic myeloid leukemia."
            ],
            "warnings": [
                "Hepatotoxicity and QT prolongation have been reported."
            ],
            "spl_product_data_elements": ["Imatinib mesylate tablets"],
            "effective_time": "20260321",
            "version": "1",
        }
    ],
}


def make_client() -> OpenFDAClient:
    return OpenFDAClient(tool="agentic-ai-poc-tests", email="tests@example.com")


def test_build_openfda_query_for_safety() -> None:
    client = make_client()
    query = client.build_openfda_query(
        "심장독성 위험은?",
        "safety",
        target="imatinib",
    )

    assert 'openfda.generic_name:"imatinib"' in query
    assert "warnings" in query
    assert "hepatotoxicity" in query


def test_search_openfda_parses_results(monkeypatch) -> None:
    client = make_client()

    def fake_request_json(url: str, params: dict[str, str]) -> dict[str, object]:
        assert url.endswith("/drug/label.json")
        assert params["search"] == 'openfda.generic_name:"imatinib"'
        assert params["limit"] == "3"
        return SAMPLE_LABEL_RESPONSE

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    results = client.search_openfda('openfda.generic_name:"imatinib"', limit=3)

    assert results == SAMPLE_LABEL_RESPONSE["results"]


def test_normalize_openfda_label_preserves_source_metadata() -> None:
    client = make_client()
    label = SAMPLE_LABEL_RESPONSE["results"][0]

    item = client.normalize_openfda_label(label)

    assert isinstance(item, EvidenceItem)
    assert item.source == "openfda"
    assert item.pmid == "openfda-imatinib-label"
    assert item.title == "Gleevec / imatinib mesylate label"
    assert "hepatotoxicity" in item.abstract.lower()
    assert item.url == "https://api.fda.gov/drug/label/openfda-imatinib-label"
    assert item.metadata["generic_name"] == ["imatinib mesylate"]
    assert item.metadata["brand_name"] == ["Gleevec"]
    assert item.metadata["effective_time"] == "20260321"


def test_score_openfda_evidence_prefers_keyword_matches() -> None:
    client = make_client()
    strong = EvidenceItem(
        source="openfda",
        pmid="1",
        title="Gleevec label with QT prolongation warning",
        abstract="QT prolongation and hepatotoxicity were reported.",
        journal="openFDA",
        pub_year=None,
        url="https://api.fda.gov/drug/label/1",
    )
    weak = EvidenceItem(
        source="openfda",
        pmid="2",
        title="General product label",
        abstract="General usage information.",
        journal="openFDA",
        pub_year=None,
        url="https://api.fda.gov/drug/label/2",
    )

    strong_score = client.score_openfda_evidence(strong, "심장독성 QT hepatotoxicity", target="imatinib")
    weak_score = client.score_openfda_evidence(weak, "심장독성 QT hepatotoxicity", target="imatinib")

    assert strong_score > weak_score


def test_collect_openfda_evidence_uses_fallback_queries(monkeypatch) -> None:
    client = make_client()
    searched_queries: list[str] = []

    def fake_search(query: str, limit: int = 10) -> list[dict[str, object]]:
        searched_queries.append(query)
        if query == 'openfda.generic_name:"imatinib" AND (warnings OR hepatotoxicity OR QT prolongation)':
            return []
        return SAMPLE_LABEL_RESPONSE["results"]

    monkeypatch.setattr(client, "search_openfda", fake_search)

    packet = client.collect_openfda_evidence(
        question="심장독성 위험은?",
        question_type="safety",
        target="imatinib",
        top_k=1,
    )

    assert packet.source == "openfda"
    assert packet.source_health == "ok"
    assert packet.items[0].pmid == "openfda-imatinib-label"
    assert searched_queries[0] == 'openfda.generic_name:"imatinib" AND (warnings OR hepatotoxicity OR QT prolongation)'


def test_collect_openfda_evidence_returns_missing_reason_when_no_hits(monkeypatch) -> None:
    client = make_client()
    monkeypatch.setattr(client, "search_openfda", lambda query, limit=10: [])

    packet = client.collect_openfda_evidence(
        question="규제 경고가 있나?",
        question_type="regulatory",
        target="imatinib",
    )

    assert packet.items == []
    assert packet.source_health == "degraded"
    assert packet.missing_reason == "no_openfda_hits"


def test_collect_openfda_evidence_returns_degraded_packet_on_request_failure(monkeypatch) -> None:
    client = make_client()

    def raise_error(query: str, limit: int = 10) -> list[dict[str, object]]:
        raise URLError("upstream 503")

    monkeypatch.setattr(client, "search_openfda", raise_error)

    packet = client.collect_openfda_evidence(
        question="규제 경고가 있나?",
        question_type="regulatory",
        target="imatinib",
    )

    assert packet.items == []
    assert packet.source_health == "degraded"
    assert packet.missing_reason == "openfda_request_failed"
    assert packet.diagnostics["error_type"] == "URLError"
    assert "upstream 503" in packet.diagnostics["error"]


def test_collect_openfda_evidence_treats_http_404_as_no_hits_and_tries_fallback_queries(
    monkeypatch,
) -> None:
    client = make_client()
    searched_queries: list[str] = []

    def fake_search(query: str, limit: int = 10) -> list[dict[str, object]]:
        searched_queries.append(query)
        if query == 'openfda.generic_name:"ABC-101" AND (warnings OR hepatotoxicity OR QT prolongation)':
            raise HTTPError(
                url="https://api.fda.gov/drug/label.json",
                code=404,
                msg="Not Found",
                hdrs=None,
                fp=None,
            )
        return []

    monkeypatch.setattr(client, "search_openfda", fake_search)

    packet = client.collect_openfda_evidence(
        question="심장독성 위험은?",
        question_type="safety",
        target="KRAS G12C",
        compound_name="ABC-101",
    )

    assert packet.items == []
    assert packet.missing_reason == "no_openfda_hits"
    assert packet.diagnostics == {}
    assert searched_queries[:2] == [
        'openfda.generic_name:"ABC-101" AND (warnings OR hepatotoxicity OR QT prolongation)',
        'openfda.generic_name:"ABC-101"',
    ]


def test_collect_openfda_evidence_classifies_parse_failures(monkeypatch) -> None:
    client = make_client()

    def raise_error(query: str, limit: int = 10) -> list[dict[str, object]]:
        raise JSONDecodeError("bad json", "{}", 0)

    monkeypatch.setattr(client, "search_openfda", raise_error)

    packet = client.collect_openfda_evidence(
        question="규제 경고가 있나?",
        question_type="regulatory",
        target="imatinib",
    )

    assert packet.items == []
    assert packet.source_health == "degraded"
    assert packet.missing_reason == "openfda_response_parse_failed"
    assert packet.diagnostics["error_type"] == "JSONDecodeError"
