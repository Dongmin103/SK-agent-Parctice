from __future__ import annotations

import json
import re
from dataclasses import replace
from json import JSONDecodeError
from typing import Any
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.clients.pubmed import QUESTION_ALIASES, TTLCache
from app.domain.models import EvidenceItem, EvidencePacket

CLINICALTRIALS_URL = "https://clinicaltrials.gov/api/v2/studies"
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]{1,}")
QUESTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "safety": ("clinical trial", "safety", "adverse event", "dose limiting toxicity"),
    "pk": ("clinical trial", "pharmacokinetics", "drug interaction", "exposure"),
    "regulatory": ("clinical trial", "phase 2", "phase 3", "FDA approval"),
}


class ClinicalTrialsClient:
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

    def build_clinicaltrials_query(
        self,
        question: str,
        question_type: str,
        target: str | None = None,
        compound_name: str | None = None,
    ) -> str:
        category = QUESTION_ALIASES.get(question_type, question_type)
        keywords = QUESTION_KEYWORDS.get(category) or QUESTION_KEYWORDS["regulatory"]
        keyword_clause = " OR ".join(keywords)
        base = ""
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
            queries.append(self.build_clinicaltrials_query(question, question_type, target, compound_name))
        if target:
            queries.append(self.build_clinicaltrials_query(question, question_type, target, None))
            queries.append(f'"{target}"')
        if compound_name:
            queries.append(self.build_clinicaltrials_query(question, question_type, None, compound_name))
            queries.append(f'"{compound_name}"')
        fallback = self._query_from_question(question)
        if fallback:
            queries.append(fallback)

        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            if query and query not in seen:
                seen.add(query)
                deduped.append(query)
        return deduped or ["clinical trial"]

    def search_clinicaltrials(self, query: str, retmax: int = 10) -> list[dict[str, object]]:
        cache_key = f"clinicaltrials:search:{query}:{retmax}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return list(cached)

        payload = self._request_json(
            CLINICALTRIALS_URL,
            {"query.term": query, "pageSize": str(retmax)},
        )
        studies = payload.get("studies", [])
        if not isinstance(studies, list):
            studies = []
        self.cache.set(cache_key, studies)
        return studies

    def normalize_clinicaltrials_study(self, study: dict[str, object]) -> EvidenceItem:
        protocol = study.get("protocolSection") or {}
        identification = protocol.get("identificationModule") or {}
        status = protocol.get("statusModule") or {}
        conditions = protocol.get("conditionsModule") or {}
        design = protocol.get("designModule") or {}
        description = protocol.get("descriptionModule") or {}
        sponsor = protocol.get("sponsorCollaboratorsModule") or {}

        nct_id = str(identification.get("nctId", ""))
        title = str(
            identification.get("briefTitle", "")
            or identification.get("officialTitle", "")
            or nct_id
        )
        abstract = str(description.get("briefSummary", "")).strip()
        start_date = (status.get("startDateStruct") or {}).get("date", "")
        lead_sponsor = (sponsor.get("leadSponsor") or {}).get("name", "")
        conditions_list = conditions.get("conditions") or []
        phase_list = design.get("phases") or []

        return EvidenceItem(
            source="clinicaltrials",
            pmid=nct_id,
            title=title,
            abstract=abstract,
            journal="ClinicalTrials.gov",
            pub_year=self._extract_year(start_date),
            authors=[str(lead_sponsor)] if lead_sponsor else [],
            url=f"https://clinicaltrials.gov/study/{nct_id}",
            metadata={
                "nct_id": nct_id,
                "status": str(status.get("overallStatus", "")),
                "conditions": list(conditions_list) if isinstance(conditions_list, list) else [],
                "phase": list(phase_list) if isinstance(phase_list, list) else [],
            },
        )

    def score_clinicaltrials_evidence(
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

        status = str(item.metadata.get("status", "")).upper()
        if status in {"RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION"}:
            score += 1.5
        if item.pub_year is not None and item.pub_year >= 2020:
            score += 1.0
        return score

    def collect_clinicaltrials_evidence(
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
                studies = self.search_clinicaltrials(query, retmax=retmax)
                if not studies:
                    continue

                items = [self.normalize_clinicaltrials_study(study) for study in studies]
                scored = [
                    replace(
                        item,
                        score=self.score_clinicaltrials_evidence(item, question, target=target),
                    )
                    for item in items
                ]
                scored.sort(key=lambda item: item.score, reverse=True)

                return EvidencePacket(
                    source="clinicaltrials",
                    query=query,
                    items=scored[:top_k],
                    source_health="ok",
                )
        except (HTTPError, URLError) as exc:
            return self._failure_packet(
                query=active_query,
                missing_reason="clinicaltrials_request_failed",
                exc=exc,
            )
        except JSONDecodeError as exc:
            return self._failure_packet(
                query=active_query,
                missing_reason="clinicaltrials_response_parse_failed",
                exc=exc,
            )

        return EvidencePacket(
            source="clinicaltrials",
            query=queries[-1] if queries else "",
            items=[],
            source_health="degraded",
            missing_reason="no_clinicaltrials_hits",
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
            source="clinicaltrials",
            query=query,
            items=[],
            source_health="degraded",
            missing_reason=missing_reason,
            diagnostics={"error": str(exc), "error_type": type(exc).__name__},
        )

    def _extract_year(self, value: str) -> int | None:
        match = re.search(r"(19|20)\d{2}", value or "")
        if match:
            return int(match.group(0))
        return None

    def _query_from_question(self, question: str) -> str:
        return " ".join(sorted(self._tokenize(question)))

    def _tokenize(self, text: str) -> set[str]:
        return {token.lower() for token in TOKEN_RE.findall(text or "")}
