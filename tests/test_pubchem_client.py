from __future__ import annotations

from json import JSONDecodeError
from urllib.error import HTTPError, URLError

from app.clients.pubchem import PubChemClient
from app.domain.models import EvidenceItem


SAMPLE_CID_RESPONSE = {
    "IdentifierList": {
        "CID": [2244],
    }
}

SAMPLE_PROPERTY_RESPONSE = {
    "PropertyTable": {
        "Properties": [
            {
                "CID": 2244,
                "Title": "Aspirin",
                "MolecularFormula": "C9H8O4",
                "XLogP": 1.2,
                "TPSA": 63.6,
                "ConnectivitySMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
            }
        ]
    }
}


def make_client() -> PubChemClient:
    return PubChemClient(tool="agentic-ai-poc-tests", email="tests@example.com")


def test_search_pubchem_parses_cids(monkeypatch) -> None:
    client = make_client()

    def fake_request_json(url: str, params: dict[str, str]) -> dict[str, object]:
        assert "compound/name/Aspirin/cids/JSON" in url
        return SAMPLE_CID_RESPONSE

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    assert client.search_pubchem("Aspirin", retmax=3) == ["2244"]


def test_fetch_pubchem_compounds_parses_properties(monkeypatch) -> None:
    client = make_client()

    def fake_request_json(url: str, params: dict[str, str]) -> dict[str, object]:
        assert "compound/cid/2244/property/" in url
        return SAMPLE_PROPERTY_RESPONSE

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    compounds = client.fetch_pubchem_compounds(["2244"])

    assert len(compounds) == 1
    assert compounds[0]["cid"] == "2244"
    assert compounds[0]["title"] == "Aspirin"
    assert compounds[0]["formula"] == "C9H8O4"
    assert compounds[0]["smiles"] == "CC(=O)OC1=CC=CC=C1C(=O)O"


def test_normalize_pubchem_compound_preserves_metadata() -> None:
    client = make_client()
    raw = {
        "cid": "2244",
        "title": "Aspirin",
        "formula": "C9H8O4",
        "xlogp": 1.2,
        "tpsa": 63.6,
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
    }

    item = client.normalize_pubchem_compound(raw)

    assert item.source == "pubchem"
    assert item.pmid == "2244"
    assert item.title == "Aspirin"
    assert "C9H8O4" in item.abstract
    assert item.url == "https://pubchem.ncbi.nlm.nih.gov/compound/2244"
    assert item.metadata["cid"] == "2244"
    assert item.metadata["formula"] == "C9H8O4"
    assert item.metadata["xlogp"] == 1.2


def test_score_pubchem_evidence_prefers_keyword_hits() -> None:
    client = make_client()
    strong = EvidenceItem(
        source="pubchem",
        pmid="2244",
        title="Aspirin",
        abstract="Kinase scaffold property summary with XLogP and TPSA signals.",
        journal="PubChem",
        pub_year=None,
        url="https://pubchem.ncbi.nlm.nih.gov/compound/2244",
    )
    weak = EvidenceItem(
        source="pubchem",
        pmid="1",
        title="Unrelated molecule",
        abstract="General information only.",
        journal="PubChem",
        pub_year=None,
        url="https://pubchem.ncbi.nlm.nih.gov/compound/1",
    )

    strong_score = client.score_pubchem_evidence(strong, "kinase property XLogP TPSA", target="Aspirin")
    weak_score = client.score_pubchem_evidence(weak, "kinase property XLogP TPSA", target="Aspirin")

    assert strong_score > weak_score


def test_collect_pubchem_evidence_uses_fallback_queries(monkeypatch) -> None:
    client = make_client()
    searched_queries: list[str] = []

    def fake_search(query: str, retmax: int = 10) -> list[str]:
        searched_queries.append(query)
        if query == "ABC-101":
            return []
        return ["2244"]

    monkeypatch.setattr(client, "search_pubchem", fake_search)
    monkeypatch.setattr(
        client,
        "fetch_pubchem_compounds",
        lambda cids: [
            {
                "cid": "2244",
                "title": "Aspirin",
                "formula": "C9H8O4",
                "xlogp": 1.2,
                "tpsa": 63.6,
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
            }
        ],
    )

    packet = client.collect_pubchem_evidence(
        question="이 화합물의 chemistry evidence는?",
        question_type="chemistry",
        target="KRAS G12C",
        compound_name="ABC-101",
        top_k=2,
    )

    assert packet.source == "pubchem"
    assert packet.source_health == "ok"
    assert packet.items[0].pmid == "2244"
    assert searched_queries[:2] == ["ABC-101", "KRAS G12C"]


def test_collect_pubchem_evidence_returns_missing_reason_when_no_hits(monkeypatch) -> None:
    client = make_client()
    monkeypatch.setattr(client, "search_pubchem", lambda query, retmax=10: [])

    packet = client.collect_pubchem_evidence(
        question="이 화합물의 chemistry evidence는?",
        question_type="chemistry",
        target="KRAS G12C",
        compound_name="ABC-101",
    )

    assert packet.items == []
    assert packet.source_health == "degraded"
    assert packet.missing_reason == "no_pubchem_hits"


def test_collect_pubchem_evidence_returns_degraded_packet_on_request_failure(monkeypatch) -> None:
    client = make_client()

    def boom(query: str, retmax: int = 10) -> list[str]:
        raise URLError("upstream timeout")

    monkeypatch.setattr(client, "search_pubchem", boom)

    packet = client.collect_pubchem_evidence(
        question="이 화합물의 chemistry evidence는?",
        question_type="chemistry",
        target="KRAS G12C",
        compound_name="ABC-101",
    )

    assert packet.items == []
    assert packet.source_health == "degraded"
    assert packet.missing_reason == "pubchem_request_failed"
    assert packet.diagnostics["error_type"] == "URLError"
    assert "upstream timeout" in packet.diagnostics["error"]


def test_collect_pubchem_evidence_treats_http_404_as_no_hits_and_tries_fallback_queries(
    monkeypatch,
) -> None:
    client = make_client()
    searched_queries: list[str] = []

    def fake_search(query: str, retmax: int = 10) -> list[str]:
        searched_queries.append(query)
        if query == "ABC-101":
            raise HTTPError(
                url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/ABC-101/cids/JSON",
                code=404,
                msg="PUGREST.NotFound",
                hdrs=None,
                fp=None,
            )
        return []

    monkeypatch.setattr(client, "search_pubchem", fake_search)

    packet = client.collect_pubchem_evidence(
        question="이 화합물의 chemistry evidence는?",
        question_type="chemistry",
        target="KRAS G12C",
        compound_name="ABC-101",
    )

    assert packet.items == []
    assert packet.missing_reason == "no_pubchem_hits"
    assert packet.diagnostics == {}
    assert searched_queries[:2] == ["ABC-101", "KRAS G12C"]


def test_collect_pubchem_evidence_classifies_parse_failures(monkeypatch) -> None:
    client = make_client()

    def boom(query: str, retmax: int = 10) -> list[str]:
        raise JSONDecodeError("bad json", "{}", 0)

    monkeypatch.setattr(client, "search_pubchem", boom)

    packet = client.collect_pubchem_evidence(
        question="이 화합물의 chemistry evidence는?",
        question_type="chemistry",
        target="KRAS G12C",
        compound_name="ABC-101",
    )

    assert packet.items == []
    assert packet.source_health == "degraded"
    assert packet.missing_reason == "pubchem_response_parse_failed"
    assert packet.diagnostics["error_type"] == "JSONDecodeError"


def test_collect_pubchem_evidence_rescores_existing_items(monkeypatch) -> None:
    client = make_client()
    original_item = EvidenceItem(
        source="pubchem",
        pmid="2244",
        title="Aspirin",
        abstract="Original summary",
        journal="PubChem",
        pub_year=None,
        url="https://pubchem.ncbi.nlm.nih.gov/compound/2244",
        score=99.0,
        missing_reason="original-missing-reason",
        metadata={"cid": "2244"},
    )

    monkeypatch.setattr(client, "search_pubchem", lambda query, retmax=10: ["2244"])
    monkeypatch.setattr(client, "fetch_pubchem_compounds", lambda cids: [{"cid": "2244"}])
    monkeypatch.setattr(client, "normalize_pubchem_compound", lambda compound: original_item)
    monkeypatch.setattr(client, "score_pubchem_evidence", lambda item, question, target=None: 4.5)

    packet = client.collect_pubchem_evidence(
        question="chemistry evidence?",
        question_type="chemistry",
        target="KRAS G12C",
        compound_name="ABC-101",
    )

    assert packet.items[0] is not original_item
    assert packet.items[0].score == 4.5
    assert packet.items[0].missing_reason == "original-missing-reason"
    assert packet.items[0].metadata == {"cid": "2244"}
