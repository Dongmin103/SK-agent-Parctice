from __future__ import annotations

from json import JSONDecodeError
from urllib.error import URLError

from app.clients.clinicaltrials import ClinicalTrialsClient
from app.domain.models import EvidenceItem


SAMPLE_RESPONSE = {
    "studies": [
        {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT01234567",
                    "briefTitle": "KRAS G12C inhibitor phase 2 study",
                    "officialTitle": "A phase 2 study of KRAS G12C inhibitor ABC-101",
                },
                "statusModule": {
                    "overallStatus": "RECRUITING",
                    "startDateStruct": {"date": "2025-01-01", "type": "ACTUAL"},
                    "primaryCompletionDateStruct": {"date": "2026-06-01", "type": "ESTIMATED"},
                },
                "conditionsModule": {"conditions": ["KRAS G12C", "Solid Tumor"]},
                "designModule": {"phases": ["PHASE2"]},
                "descriptionModule": {
                    "briefSummary": "The study evaluates safety and preliminary efficacy in recruiting patients."
                },
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": "Example Pharma", "class": "INDUSTRY"}
                },
            }
        },
        {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT07654321",
                    "briefTitle": "Unrelated trial",
                },
                "statusModule": {
                    "overallStatus": "COMPLETED",
                },
                "descriptionModule": {
                    "briefSummary": "A completed unrelated study."
                },
            }
        },
    ]
}


def make_client() -> ClinicalTrialsClient:
    return ClinicalTrialsClient(tool="agentic-ai-poc-tests", email="tests@example.com")


def test_search_clinicaltrials_parses_study_list(monkeypatch) -> None:
    client = make_client()

    def fake_request_json(url: str, params: dict[str, str]) -> dict[str, object]:
        assert "clinicaltrials.gov" in url
        assert params["query.term"] == "KRAS G12C"
        assert params["pageSize"] == "5"
        return SAMPLE_RESPONSE

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    studies = client.search_clinicaltrials("KRAS G12C", retmax=5)

    assert len(studies) == 2
    assert studies[0]["protocolSection"]["identificationModule"]["nctId"] == "NCT01234567"


def test_normalize_clinicaltrials_study_preserves_metadata() -> None:
    client = make_client()
    study = SAMPLE_RESPONSE["studies"][0]

    item = client.normalize_clinicaltrials_study(study)

    assert item.source == "clinicaltrials"
    assert item.pmid == "NCT01234567"
    assert item.title == "KRAS G12C inhibitor phase 2 study"
    assert item.abstract.startswith("The study evaluates safety")
    assert item.url == "https://clinicaltrials.gov/study/NCT01234567"
    assert item.metadata["nct_id"] == "NCT01234567"
    assert item.metadata["status"] == "RECRUITING"
    assert item.metadata["conditions"] == ["KRAS G12C", "Solid Tumor"]
    assert item.metadata["phase"] == ["PHASE2"]


def test_score_clinicaltrials_evidence_prefers_relevant_recent_trials() -> None:
    client = make_client()
    strong = EvidenceItem(
        source="clinicaltrials",
        pmid="NCT01234567",
        title="KRAS G12C inhibitor phase 2 study",
        abstract="The study evaluates safety and preliminary efficacy in recruiting patients.",
        journal="ClinicalTrials.gov",
        pub_year=2025,
        url="https://clinicaltrials.gov/study/NCT01234567",
        metadata={"status": "RECRUITING", "conditions": ["KRAS G12C"], "phase": ["PHASE2"]},
    )
    weak = EvidenceItem(
        source="clinicaltrials",
        pmid="NCT07654321",
        title="Unrelated trial",
        abstract="A completed unrelated study.",
        journal="ClinicalTrials.gov",
        pub_year=2018,
        url="https://clinicaltrials.gov/study/NCT07654321",
        metadata={"status": "COMPLETED", "conditions": ["Other"], "phase": []},
    )

    assert client.score_clinicaltrials_evidence(strong, "KRAS G12C 임상 진행 상황은?", target="KRAS G12C") > client.score_clinicaltrials_evidence(
        weak, "KRAS G12C 임상 진행 상황은?", target="KRAS G12C"
    )


def test_collect_clinicaltrials_evidence_uses_fallback_queries(monkeypatch) -> None:
    client = make_client()
    searched_queries: list[str] = []

    def fake_search(query: str, retmax: int = 10):
        searched_queries.append(query)
        if query == '"KRAS G12C" AND (clinical trial OR phase 2 OR phase 3 OR FDA approval)':
            return []
        return [SAMPLE_RESPONSE["studies"][0]]

    monkeypatch.setattr(client, "search_clinicaltrials", fake_search)

    packet = client.collect_clinicaltrials_evidence(
        question="임상 진행 상황은?",
        question_type="regulatory",
        target="KRAS G12C",
        top_k=3,
    )

    assert packet.source == "clinicaltrials"
    assert packet.source_health == "ok"
    assert packet.items
    assert searched_queries[0].startswith('"KRAS G12C" AND')
    assert searched_queries[1] == '"KRAS G12C"'


def test_collect_clinicaltrials_evidence_returns_missing_reason_when_no_hits(monkeypatch) -> None:
    client = make_client()
    monkeypatch.setattr(client, "search_clinicaltrials", lambda query, retmax=10: [])

    packet = client.collect_clinicaltrials_evidence(
        question="임상 근거가 있나?",
        question_type="regulatory",
        target="KRAS G12C",
    )

    assert packet.items == []
    assert packet.source_health == "degraded"
    assert packet.missing_reason == "no_clinicaltrials_hits"


def test_collect_clinicaltrials_evidence_returns_degraded_packet_on_request_failure(monkeypatch) -> None:
    client = make_client()

    def boom(url: str, params: dict[str, str]) -> dict[str, object]:
        raise URLError("upstream 503")

    monkeypatch.setattr(client, "_request_json", boom)

    packet = client.collect_clinicaltrials_evidence(
        question="임상 근거가 있나?",
        question_type="regulatory",
        target="KRAS G12C",
    )

    assert packet.items == []
    assert packet.source_health == "degraded"
    assert packet.missing_reason == "clinicaltrials_request_failed"
    assert packet.diagnostics["error_type"] == "URLError"
    assert "upstream 503" in packet.diagnostics["error"]


def test_collect_clinicaltrials_evidence_classifies_parse_failures(monkeypatch) -> None:
    client = make_client()

    def boom(url: str, params: dict[str, str]) -> dict[str, object]:
        raise JSONDecodeError("bad json", "{}", 0)

    monkeypatch.setattr(client, "_request_json", boom)

    packet = client.collect_clinicaltrials_evidence(
        question="임상 근거가 있나?",
        question_type="regulatory",
        target="KRAS G12C",
    )

    assert packet.items == []
    assert packet.source_health == "degraded"
    assert packet.missing_reason == "clinicaltrials_response_parse_failed"
    assert packet.diagnostics["error_type"] == "JSONDecodeError"
