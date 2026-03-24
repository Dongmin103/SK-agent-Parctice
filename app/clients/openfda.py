from __future__ import annotations

import json
import re
from dataclasses import replace
from json import JSONDecodeError
from typing import Any
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.clients.pubmed import TTLCache
from app.domain.models import EvidenceItem, EvidencePacket

OPENFDA_URL = "https://api.fda.gov/drug/label.json"
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]{1,}")
QUESTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "safety": ("warnings", "hepatotoxicity", "QT prolongation"),
    "regulatory": ("boxed warning", "approval", "warnings"),
    "pk": ("drug interaction", "dosage", "exposure"),
}
QUESTION_ALIASES = {
    "toxicity": "safety",
    "ddi": "pk",
}


class OpenFDAClient:
    def __init__(
        self,
        *,
        tool: str,
        email: str,
        timeout: float = 10.0,
        cache_ttl_seconds: int = 86_400,
    ) -> None:
        self.tool = tool
        self.email = email
        self.timeout = timeout
        self.cache = TTLCache(ttl_seconds=cache_ttl_seconds)

    def build_openfda_query(
        self,
        question: str,
        question_type: str,
        target: str | None = None,
        compound_name: str | None = None,
    ) -> str:
        category = QUESTION_ALIASES.get(question_type, question_type)
        keywords = QUESTION_KEYWORDS.get(category) or QUESTION_KEYWORDS["safety"]
        keyword_clause = " OR ".join(keywords)
        base_name = compound_name or target or self._query_from_question(question)
        return f'openfda.generic_name:"{base_name}" AND ({keyword_clause})'

    def build_query_candidates(
        self,
        question: str,
        question_type: str,
        target: str | None = None,
        compound_name: str | None = None,
    ) -> list[str]:
        queries: list[str] = []
        if compound_name or target:
            queries.append(
                self.build_openfda_query(
                    question,
                    question_type,
                    target=target,
                    compound_name=compound_name,
                )
            )
        if compound_name:
            queries.append(f'openfda.generic_name:"{compound_name}"')
        if target:
            queries.append(f'openfda.generic_name:"{target}"')

        fallback = self._query_from_question(question)
        if fallback:
            queries.append(fallback)

        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            if query and query not in seen:
                seen.add(query)
                deduped.append(query)
        return deduped or ['openfda.generic_name:"drug"']

    def search_openfda(self, query: str, limit: int = 10) -> list[dict[str, object]]:
        cache_key = f"openfda:search:{query}:{limit}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return list(cached)

        payload = self._request_json(OPENFDA_URL, {"search": query, "limit": str(limit)})
        results = payload.get("results", [])
        if not isinstance(results, list):
            results = []
        self.cache.set(cache_key, results)
        return results

    def normalize_openfda_label(self, label: dict[str, object]) -> EvidenceItem:
        label_id = str(label.get("id", ""))
        generic_names = self._string_list(
            label.get("generic_name")
            or (label.get("openfda") or {}).get("generic_name")
            or []
        )
        brand_names = self._string_list(
            label.get("brand_name")
            or (label.get("openfda") or {}).get("brand_name")
            or []
        )
        indications = self._string_list(label.get("indications_and_usage") or [])
        warnings = self._string_list(label.get("warnings") or label.get("boxed_warning") or [])

        title_left = brand_names[0] if brand_names else label_id
        title_right = generic_names[0] if generic_names else "drug"
        abstract_parts = []
        if indications:
            abstract_parts.append(indications[0])
        if warnings:
            abstract_parts.append(warnings[0])

        return EvidenceItem(
            source="openfda",
            pmid=label_id,
            title=f"{title_left} / {title_right} label",
            abstract=" ".join(abstract_parts).strip(),
            journal="openFDA",
            pub_year=self._extract_year(str(label.get("effective_time", ""))),
            url=f"https://api.fda.gov/drug/label/{label_id}",
            metadata={
                "generic_name": generic_names,
                "brand_name": brand_names,
                "effective_time": str(label.get("effective_time", "")),
                "version": str(label.get("version", "")),
            },
        )

    def score_openfda_evidence(
        self,
        item: EvidenceItem,
        question: str,
        target: str | None = None,
    ) -> float:
        question_tokens = self._tokenize(question)
        title_tokens = self._tokenize(item.title)
        abstract_tokens = self._tokenize(item.abstract)
        target_tokens = self._tokenize(target or "")
        metadata_tokens = self._tokenize(
            " ".join(str(value) for value in item.metadata.values() if value not in (None, ""))
        )

        score = 0.0
        score += len(question_tokens & title_tokens) * 3.0
        score += len(question_tokens & abstract_tokens) * 1.5
        score += len(question_tokens & metadata_tokens) * 1.0
        score += len(target_tokens & (title_tokens | abstract_tokens | metadata_tokens)) * 4.0
        return score

    def collect_openfda_evidence(
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
        active_query = queries[0] if queries else ""
        try:
            for query in queries:
                active_query = query
                try:
                    labels = self.search_openfda(query, limit=retmax)
                except HTTPError as exc:
                    if self._is_not_found_error(exc):
                        continue
                    return self._failure_packet(
                        query=active_query,
                        missing_reason="openfda_request_failed",
                        exc=exc,
                    )
                if not labels:
                    continue

                items = [self.normalize_openfda_label(label) for label in labels]
                scored = [
                    replace(item, score=self.score_openfda_evidence(item, question, target=target))
                    for item in items
                ]
                scored.sort(key=lambda item: item.score, reverse=True)

                return EvidencePacket(
                    source="openfda",
                    query=query,
                    items=scored[:top_k],
                    source_health="ok",
                )
        except (HTTPError, URLError) as exc:
            return self._failure_packet(
                query=active_query,
                missing_reason="openfda_request_failed",
                exc=exc,
            )
        except JSONDecodeError as exc:
            return self._failure_packet(
                query=active_query,
                missing_reason="openfda_response_parse_failed",
                exc=exc,
            )

        return EvidencePacket(
            source="openfda",
            query=queries[-1] if queries else "",
            items=[],
            source_health="degraded",
            missing_reason="no_openfda_hits",
        )

    def _request_json(self, url: str, params: dict[str, str]) -> dict[str, Any]:
        full_url = f"{url}?{urlencode(params)}"
        request = Request(full_url, headers={"User-Agent": self.tool})
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def _failure_packet(
        self,
        *,
        query: str,
        missing_reason: str,
        exc: Exception,
    ) -> EvidencePacket:
        return EvidencePacket(
            source="openfda",
            query=query,
            items=[],
            source_health="degraded",
            missing_reason=missing_reason,
            diagnostics={"error": str(exc), "error_type": type(exc).__name__},
        )

    def _is_not_found_error(self, exc: HTTPError) -> bool:
        return exc.code == 404

    def _extract_year(self, value: str) -> int | None:
        match = re.search(r"(19|20)\d{2}", value or "")
        if match:
            return int(match.group(0))
        return None

    def _query_from_question(self, question: str) -> str:
        return " ".join(sorted(self._tokenize(question)))

    def _string_list(self, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if str(item).strip()]

    def _tokenize(self, text: str) -> set[str]:
        return {token.lower() for token in TOKEN_RE.findall(text or "")}
