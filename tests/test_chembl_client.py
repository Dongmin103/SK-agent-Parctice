from __future__ import annotations

from json import JSONDecodeError
from urllib.error import URLError

from app.clients.chembl import ChEMBLClient
from app.domain.models import EvidenceItem, EvidencePacket


SAMPLE_SEARCH_JSON = {
    "molecules": [
        {
            "molecule_chembl_id": "CHEMBL941",
            "pref_name": "Imatinib",
            "molecule_properties": {
                "alogp": "4.59",
                "full_mwt": "493.62",
                "psa": "86.28",
            },
            "molecule_structures": {
                "canonical_smiles": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
            },
            "max_phase": "4.0",
            "first_approval": 2001,
        }
    ]
}

SAMPLE_FETCH_JSON = {
    "molecule_chembl_id": "CHEMBL941",
    "pref_name": "Imatinib",
    "synonyms": ["STI571", "Gleevec"],
    "molecule_properties": {
        "alogp": "4.59",
        "full_mwt": "493.62",
        "psa": "86.28",
    },
    "molecule_structures": {
        "canonical_smiles": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
    },
    "max_phase": "4.0",
    "first_approval": 2001,
    "atc_classifications": ["L01EA01"],
}


def make_client() -> ChEMBLClient:
    return ChEMBLClient(tool="agentic-ai-poc-tests", email="tests@example.com")


def test_search_chembl_parses_molecule_ids(monkeypatch) -> None:
    client = make_client()

    def fake_request_json(url: str, params: dict[str, str]) -> dict[str, object]:
        assert "molecule/search.json" in url
        assert params["q"] == "Imatinib"
        return SAMPLE_SEARCH_JSON

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    assert client.search_chembl("Imatinib", retmax=3) == ["CHEMBL941"]


def test_fetch_chembl_molecules_parses_detail_payload(monkeypatch) -> None:
    client = make_client()

    def fake_request_json(url: str, params: dict[str, str]) -> dict[str, object]:
        assert url.endswith("/CHEMBL941.json")
        return SAMPLE_FETCH_JSON

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    molecules = client.fetch_chembl_molecules(["CHEMBL941"])

    assert len(molecules) == 1
    assert molecules[0].chembl_id == "CHEMBL941"
    assert molecules[0].pref_name == "Imatinib"
    assert molecules[0].canonical_smiles.endswith("n1")
    assert molecules[0].synonyms == ["STI571", "Gleevec"]


def test_normalize_chembl_molecule_preserves_metadata() -> None:
    client = make_client()
    raw = client._parse_chembl_molecule(SAMPLE_FETCH_JSON)

    item = client.normalize_chembl_molecule(raw)

    assert isinstance(item, EvidenceItem)
    assert item.source == "chembl"
    assert item.pmid == "CHEMBL941"
    assert item.url == "https://www.ebi.ac.uk/chembl/compound_report_card/CHEMBL941/"
    assert item.metadata["chembl_id"] == "CHEMBL941"
    assert item.metadata["pref_name"] == "Imatinib"
    assert item.metadata["canonical_smiles"].startswith("Cc1ccc(")
    assert item.metadata["synonyms"] == "STI571, Gleevec"


def test_score_chembl_evidence_prefers_keyword_hits() -> None:
    client = make_client()
    strong = EvidenceItem(
        source="chembl",
        pmid="CHEMBL941",
        title="Imatinib",
        abstract="Approved kinase inhibitor with solid clinical phase history.",
        journal="ChEMBL",
        pub_year=2001,
        url="https://www.ebi.ac.uk/chembl/compound_report_card/CHEMBL941/",
    )
    weak = EvidenceItem(
        source="chembl",
        pmid="CHEMBL000",
        title="Unrelated scaffold",
        abstract="",
        journal="ChEMBL",
        pub_year=None,
        url="https://www.ebi.ac.uk/chembl/compound_report_card/CHEMBL000/",
    )

    strong_score = client.score_chembl_evidence(strong, "approval phase kinase", target="BCR-ABL")
    weak_score = client.score_chembl_evidence(weak, "approval phase kinase", target="BCR-ABL")

    assert strong_score > weak_score


def test_collect_chembl_evidence_uses_fallback_queries(monkeypatch) -> None:
    client = make_client()
    searched_queries: list[str] = []

    def fake_search(query: str, retmax: int = 10) -> list[str]:
        searched_queries.append(query)
        if query == '"Imatinib" BCR-ABL':
            return []
        return ["CHEMBL941"]

    monkeypatch.setattr(client, "search_chembl", fake_search)
    monkeypatch.setattr(
        client,
        "fetch_chembl_molecules",
        lambda chembl_ids: [client._parse_chembl_molecule(SAMPLE_FETCH_JSON)],
    )

    packet = client.collect_chembl_evidence(
        question="이 화합물의 승인 및 키나아제 evidence는?",
        question_type="regulatory",
        target="BCR-ABL",
        compound_name="Imatinib",
        top_k=3,
    )

    assert packet.source == "chembl"
    assert packet.source_health == "ok"
    assert packet.items
    assert searched_queries[0] == '"Imatinib" BCR-ABL'
    assert packet.items[0].pmid == "CHEMBL941"


def test_collect_chembl_evidence_returns_missing_reason_when_no_hits(monkeypatch) -> None:
    client = make_client()
    monkeypatch.setattr(client, "search_chembl", lambda query, retmax=10: [])

    packet = client.collect_chembl_evidence(
        question="이 화합물의 독성 evidence는?",
        question_type="safety",
        target="BCR-ABL",
        compound_name="Imatinib",
    )

    assert packet.items == []
    assert packet.missing_reason == "no_chembl_hits"
    assert packet.source_health == "degraded"


def test_collect_chembl_evidence_marks_request_failure(monkeypatch) -> None:
    client = make_client()

    def boom(*args, **kwargs):
        raise URLError("upstream unavailable")

    monkeypatch.setattr(client, "search_chembl", boom)

    packet = client.collect_chembl_evidence(
        question="이 화합물의 독성 evidence는?",
        question_type="safety",
        target="BCR-ABL",
        compound_name="Imatinib",
    )

    assert packet.items == []
    assert packet.source_health == "degraded"
    assert packet.missing_reason == "chembl_request_failed"
    assert packet.diagnostics["error_type"] == "URLError"
    assert "upstream unavailable" in packet.diagnostics["error"]


def test_collect_chembl_evidence_classifies_parse_failures(monkeypatch) -> None:
    client = make_client()

    def boom(*args, **kwargs):
        raise JSONDecodeError("bad json", "{}", 0)

    monkeypatch.setattr(client, "search_chembl", boom)

    packet = client.collect_chembl_evidence(
        question="이 화합물의 독성 evidence는?",
        question_type="safety",
        target="BCR-ABL",
        compound_name="Imatinib",
    )

    assert packet.items == []
    assert packet.source_health == "degraded"
    assert packet.missing_reason == "chembl_response_parse_failed"
    assert packet.diagnostics["error_type"] == "JSONDecodeError"
