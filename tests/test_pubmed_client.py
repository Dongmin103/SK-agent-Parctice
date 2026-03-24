from __future__ import annotations

from app.clients.pubmed import PubMedClient
from app.domain.models import EvidenceItem, PubMedArticleRaw


SAMPLE_XML = """\
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345</PMID>
      <Article>
        <ArticleTitle>KRAS G12C inhibitors and cardiotoxicity risk</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">KRAS G12C inhibitors may affect hERG and QT prolongation.</AbstractText>
          <AbstractText Label="RESULTS">Cardiotoxicity findings were limited but measurable.</AbstractText>
        </Abstract>
        <Journal>
          <JournalIssue>
            <PubDate>
              <Year>2024</Year>
            </PubDate>
          </JournalIssue>
          <Title>Drug Safety Journal</Title>
        </Journal>
        <AuthorList>
          <Author>
            <LastName>Kim</LastName>
            <ForeName>Min</ForeName>
          </Author>
        </AuthorList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>67890</PMID>
      <Article>
        <ArticleTitle>Older PK evidence for kinase inhibitors</ArticleTitle>
        <Journal>
          <JournalIssue>
            <PubDate>
              <MedlineDate>2018 Jan-Feb</MedlineDate>
            </PubDate>
          </JournalIssue>
          <Title>Clinical PK</Title>
        </Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


def make_client() -> PubMedClient:
    return PubMedClient(tool="agentic-ai-poc-tests", email="tests@example.com")


def test_build_pubmed_query_for_safety() -> None:
    client = make_client()
    query = client.build_pubmed_query(
        "이 화합물의 심장독성 위험은?",
        "safety",
        target="KRAS G12C",
    )
    assert '"KRAS G12C"' in query
    assert "cardiotoxicity" in query
    assert "hERG" in query


def test_search_pubmed_parses_idlist(monkeypatch) -> None:
    client = make_client()

    def fake_request_json(url: str, params: dict[str, str]) -> dict[str, object]:
        assert params["db"] == "pubmed"
        assert params["term"] == "KRAS"
        return {"esearchresult": {"idlist": ["1", "2", "3"]}}

    monkeypatch.setattr(client, "_request_json", fake_request_json)
    assert client.search_pubmed("KRAS", retmax=3) == ["1", "2", "3"]


def test_fetch_pubmed_articles_parses_xml(monkeypatch) -> None:
    client = make_client()
    monkeypatch.setattr(client, "_request_text", lambda url, params: SAMPLE_XML)

    articles = client.fetch_pubmed_articles(["12345", "67890"])

    assert len(articles) == 2
    assert articles[0].pmid == "12345"
    assert "BACKGROUND:" in articles[0].abstract
    assert articles[1].pub_year == 2018
    assert articles[1].abstract == ""


def test_normalize_pubmed_article_marks_missing_abstract() -> None:
    client = make_client()
    raw = PubMedArticleRaw(
        pmid="67890",
        title="Older PK evidence for kinase inhibitors",
        abstract="",
        journal="Clinical PK",
        pub_year=2018,
        authors=[],
    )
    item = client.normalize_pubmed_article(raw)
    assert item.url.endswith("/67890/")
    assert item.missing_reason == "abstract_unavailable"


def test_score_pubmed_evidence_prefers_keyword_hits() -> None:
    client = make_client()
    strong = EvidenceItem(
        source="pubmed",
        pmid="12345",
        title="KRAS G12C inhibitors and cardiotoxicity risk",
        abstract="QT prolongation and hERG signal observed.",
        journal="Drug Safety Journal",
        pub_year=2024,
        authors=["Min Kim"],
    )
    weak = EvidenceItem(
        source="pubmed",
        pmid="67890",
        title="General kinase inhibitor observations",
        abstract="",
        journal="Clinical PK",
        pub_year=2018,
        authors=[],
        missing_reason="abstract_unavailable",
    )
    strong_score = client.score_pubmed_evidence(strong, "심장독성 hERG 위험", target="KRAS G12C")
    weak_score = client.score_pubmed_evidence(weak, "심장독성 hERG 위험", target="KRAS G12C")
    assert strong_score > weak_score


def test_collect_pubmed_evidence_uses_fallback_queries(monkeypatch) -> None:
    client = make_client()
    searched_queries: list[str] = []

    def fake_search(query: str, retmax: int = 10) -> list[str]:
        searched_queries.append(query)
        if query == '"KRAS G12C" AND (cardiotoxicity OR hERG OR QT prolongation OR hepatotoxicity OR CYP)':
            return []
        return ["12345"]

    monkeypatch.setattr(client, "search_pubmed", fake_search)
    monkeypatch.setattr(
        client,
        "fetch_pubmed_articles",
        lambda pmids: [
            PubMedArticleRaw(
                pmid="12345",
                title="KRAS G12C inhibitors and cardiotoxicity risk",
                abstract="Cardiotoxicity and hERG findings were reported.",
                journal="Drug Safety Journal",
                pub_year=2024,
                authors=["Min Kim"],
            )
        ],
    )

    packet = client.collect_pubmed_evidence(
        question="이 화합물의 심장독성 위험은?",
        question_type="safety",
        target="KRAS G12C",
        top_k=3,
    )

    assert packet.source == "pubmed"
    assert packet.source_health == "ok"
    assert packet.items
    assert searched_queries[0].startswith('"KRAS G12C" AND')


def test_collect_pubmed_evidence_returns_missing_reason_when_no_hits(monkeypatch) -> None:
    client = make_client()
    monkeypatch.setattr(client, "search_pubmed", lambda query, retmax=10: [])

    packet = client.collect_pubmed_evidence(
        question="FDA 승인 가능성은?",
        question_type="regulatory",
        target="KRAS G12C",
    )

    assert packet.items == []
    assert packet.missing_reason == "no_pubmed_hits"
    assert packet.source_health == "degraded"
