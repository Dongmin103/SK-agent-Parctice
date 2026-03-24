from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import replace
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from app.domain.models import EvidenceItem, EvidencePacket, PubMedArticleRaw, PubMedQueryInput

LOGGER = logging.getLogger(__name__)

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{PUBMED_BASE}/esearch.fcgi"
EFETCH_URL = f"{PUBMED_BASE}/efetch.fcgi"

QUESTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "safety": ("cardiotoxicity", "hERG", "QT prolongation", "hepatotoxicity", "CYP"),
    "pk": ("pharmacokinetics", "clearance", "half-life", "drug-drug interaction"),
    "regulatory": ("clinical trial", "FDA approval", "phase 2", "phase 3"),
}

QUESTION_ALIASES: dict[str, str] = {
    "toxicity": "safety",
    "ddi": "pk",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]{1,}")


class TTLCache:
    """Tiny in-memory TTL cache for local development and tests."""

    def __init__(self, ttl_seconds: int = 86_400):
        self.ttl_seconds = ttl_seconds
        self._data: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        record = self._data.get(key)
        if record is None:
            return None
        expires_at, value = record
        if expires_at < time.time():
            self._data.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._data[key] = (time.time() + self.ttl_seconds, value)


class PubMedClient:
    """Search and normalize PubMed metadata using NCBI E-utilities."""

    def __init__(
        self,
        *,
        tool: str,
        email: str,
        api_key: str | None = None,
        timeout: float = 10.0,
        cache_ttl_seconds: int = 86_400,
    ) -> None:
        self.tool = tool
        self.email = email
        self.api_key = api_key
        self.timeout = timeout
        self.cache = TTLCache(ttl_seconds=cache_ttl_seconds)

    def build_pubmed_query(
        self,
        question: str,
        question_type: str,
        target: str | None = None,
        compound_name: str | None = None,
    ) -> str:
        category = QUESTION_ALIASES.get(question_type, question_type)
        keywords = QUESTION_KEYWORDS.get(category) or QUESTION_KEYWORDS["safety"]
        keyword_clause = " OR ".join(keywords)

        if compound_name and target:
            base = f'("{compound_name}" OR "{target}")'
        elif compound_name:
            base = f'"{compound_name}"'
        elif target:
            base = f'"{target}"'
        else:
            base = self._query_from_question(question)

        return f"{base} AND ({keyword_clause})"

    def build_query_candidates(
        self,
        question: str,
        question_type: str,
        target: str | None = None,
        compound_name: str | None = None,
    ) -> list[str]:
        queries: list[str] = []

        if compound_name and target:
            queries.append(self.build_pubmed_query(question, question_type, target, compound_name))
        if target:
            queries.append(self.build_pubmed_query(question, question_type, target, None))
            queries.append(f'"{target}"')
        if compound_name:
            queries.append(self.build_pubmed_query(question, question_type, None, compound_name))
            queries.append(f'"{compound_name}"')
        if not queries:
            queries.append(self._query_from_question(question))

        # Preserve order while removing duplicates.
        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            if query not in seen:
                seen.add(query)
                deduped.append(query)
        return deduped

    def search_pubmed(self, query: str, retmax: int = 10) -> list[str]:
        cache_key = f"esearch:{query}:{retmax}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return list(cached)

        params = self._common_params()
        params.update(
            {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": str(retmax),
                "sort": "relevance",
            }
        )

        payload = self._request_json(ESEARCH_URL, params)
        pmids = payload.get("esearchresult", {}).get("idlist", [])
        if not isinstance(pmids, list):
            pmids = []
        pmids = [str(pmid) for pmid in pmids]
        self.cache.set(cache_key, pmids)
        return pmids

    def fetch_pubmed_articles(self, pmids: list[str]) -> list[PubMedArticleRaw]:
        if not pmids:
            return []

        cache_key = f"efetch:{','.join(pmids)}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return list(cached)

        params = self._common_params()
        params.update(
            {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
            }
        )

        xml_text = self._request_text(EFETCH_URL, params)
        articles = self._parse_pubmed_xml(xml_text)
        self.cache.set(cache_key, articles)
        return articles

    def normalize_pubmed_article(self, article: PubMedArticleRaw) -> EvidenceItem:
        missing_reason = "abstract_unavailable" if not article.abstract else None
        return EvidenceItem(
            source="pubmed",
            pmid=article.pmid,
            title=article.title,
            abstract=article.abstract,
            journal=article.journal,
            pub_year=article.pub_year,
            authors=article.authors,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{article.pmid}/",
            score=0.0,
            missing_reason=missing_reason,
        )

    def score_pubmed_evidence(
        self,
        item: EvidenceItem,
        question: str,
        target: str | None = None,
    ) -> float:
        question_tokens = self._tokenize(question)
        title_tokens = self._tokenize(item.title)
        abstract_tokens = self._tokenize(item.abstract)
        target_tokens = self._tokenize(target or "")

        title_hits = len(question_tokens & title_tokens)
        abstract_hits = len(question_tokens & abstract_tokens)
        target_hits = len(target_tokens & (title_tokens | abstract_tokens))

        score = 0.0
        score += title_hits * 3.0
        score += abstract_hits * 1.5
        score += target_hits * 4.0

        if item.pub_year is not None:
            current_year = time.gmtime().tm_year
            if item.pub_year >= current_year - 5:
                score += 1.0

        if item.missing_reason == "abstract_unavailable":
            score -= 2.0
        return score

    def collect_pubmed_evidence(
        self,
        question: str,
        question_type: str,
        target: str | None = None,
        compound_name: str | None = None,
        retmax: int = 10,
        top_k: int = 5,
    ) -> EvidencePacket:
        queries = self.build_query_candidates(
            question,
            question_type,
            target=target,
            compound_name=compound_name,
        )

        for query in queries:
            pmids = self.search_pubmed(query, retmax=retmax)
            if not pmids:
                continue

            raw_articles = self.fetch_pubmed_articles(pmids)
            items = [self.normalize_pubmed_article(article) for article in raw_articles]
            scored = [
                replace(item, score=self.score_pubmed_evidence(item, question, target=target))
                for item in items
            ]
            scored.sort(key=lambda item: item.score, reverse=True)

            return EvidencePacket(
                source="pubmed",
                query=query,
                items=scored[:top_k],
                source_health="ok",
            )

        return EvidencePacket(
            source="pubmed",
            query=queries[-1] if queries else "",
            items=[],
            source_health="degraded",
            missing_reason="no_pubmed_hits",
        )

    def collect_pubmed_evidence_from_queries(
        self,
        question: str,
        queries: list[str],
        *,
        target: str | None = None,
        retmax: int = 10,
        top_k: int = 5,
    ) -> EvidencePacket:
        for query in queries:
            pmids = self.search_pubmed(query, retmax=retmax)
            if not pmids:
                continue

            raw_articles = self.fetch_pubmed_articles(pmids)
            items = [self.normalize_pubmed_article(article) for article in raw_articles]
            scored = [
                replace(item, score=self.score_pubmed_evidence(item, question, target=target))
                for item in items
            ]
            scored.sort(key=lambda item: item.score, reverse=True)

            return EvidencePacket(
                source="pubmed",
                query=query,
                items=scored[:top_k],
                source_health="ok",
            )

        return EvidencePacket(
            source="pubmed",
            query=queries[-1] if queries else "",
            items=[],
            source_health="degraded",
            missing_reason="no_pubmed_hits",
        )

    def _common_params(self) -> dict[str, str]:
        params = {"tool": self.tool, "email": self.email}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _request_json(self, url: str, params: dict[str, str]) -> dict[str, Any]:
        with self._open_url(url, params) as response:
            return json.load(response)

    def _request_text(self, url: str, params: dict[str, str]) -> str:
        with self._open_url(url, params) as response:
            return response.read().decode("utf-8")

    def _open_url(self, url: str, params: dict[str, str]):
        full_url = f"{url}?{urlencode(params)}"
        request = Request(full_url, headers={"User-Agent": self.tool})
        LOGGER.debug("Calling PubMed endpoint: %s", full_url)
        return urlopen(request, timeout=self.timeout)

    def _parse_pubmed_xml(self, xml_text: str) -> list[PubMedArticleRaw]:
        root = ElementTree.fromstring(xml_text)
        articles: list[PubMedArticleRaw] = []
        for article_node in root.findall(".//PubmedArticle"):
            pmid = self._element_text(article_node.find(".//MedlineCitation/PMID"))
            title = self._element_text(article_node.find(".//Article/ArticleTitle"))

            abstract_parts: list[str] = []
            for abstract_node in article_node.findall(".//Article/Abstract/AbstractText"):
                text = self._element_text(abstract_node)
                label = abstract_node.attrib.get("Label")
                if text and label:
                    abstract_parts.append(f"{label}: {text}")
                elif text:
                    abstract_parts.append(text)

            journal = self._element_text(article_node.find(".//Article/Journal/Title"))
            pub_year = self._extract_pub_year(article_node)
            authors = self._extract_authors(article_node)

            articles.append(
                PubMedArticleRaw(
                    pmid=pmid,
                    title=title,
                    abstract=" ".join(part for part in abstract_parts if part).strip(),
                    journal=journal,
                    pub_year=pub_year,
                    authors=authors,
                )
            )
        return articles

    def _extract_pub_year(self, article_node: ElementTree.Element) -> int | None:
        year_text = self._element_text(article_node.find(".//Article/Journal/JournalIssue/PubDate/Year"))
        if year_text.isdigit():
            return int(year_text)

        medline_date = self._element_text(article_node.find(".//Article/Journal/JournalIssue/PubDate/MedlineDate"))
        match = re.search(r"(19|20)\d{2}", medline_date)
        if match:
            return int(match.group(0))
        return None

    def _extract_authors(self, article_node: ElementTree.Element) -> list[str]:
        authors: list[str] = []
        for author_node in article_node.findall(".//Article/AuthorList/Author"):
            last_name = self._element_text(author_node.find("LastName"))
            fore_name = self._element_text(author_node.find("ForeName"))
            collective_name = self._element_text(author_node.find("CollectiveName"))
            if collective_name:
                authors.append(collective_name)
            elif last_name and fore_name:
                authors.append(f"{fore_name} {last_name}")
            elif last_name:
                authors.append(last_name)
        return authors

    def _element_text(self, node: ElementTree.Element | None) -> str:
        if node is None:
            return ""
        return " ".join(text.strip() for text in node.itertext() if text and text.strip())

    def _query_from_question(self, question: str) -> str:
        tokens = sorted(self._tokenize(question))
        return " ".join(tokens)

    def _tokenize(self, text: str) -> set[str]:
        return {token.lower() for token in TOKEN_RE.findall(text or "")}
