from __future__ import annotations

import json
import re
from dataclasses import replace
from json import JSONDecodeError
from typing import Any
from urllib.parse import quote
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.clients.pubmed import TTLCache
from app.domain.models import EvidenceItem, EvidencePacket

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]{1,}")


class PubChemClient:
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

    def build_query_candidates(
        self,
        question: str,
        question_type: str,
        target: str | None = None,
        compound_name: str | None = None,
    ) -> list[str]:
        queries: list[str] = []
        if compound_name:
            queries.append(compound_name)
        if target:
            queries.append(target)
        fallback = self._query_from_question(question)
        if fallback:
            queries.append(fallback)

        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            if query and query not in seen:
                seen.add(query)
                deduped.append(query)
        return deduped or ["compound"]

    def search_pubchem(self, query: str, retmax: int = 10) -> list[str]:
        cache_key = f"pubchem:search:{query}:{retmax}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return list(cached)

        encoded_query = quote(query, safe="")
        payload = self._request_json(f"{PUBCHEM_BASE}/name/{encoded_query}/cids/JSON", {})
        cids = payload.get("IdentifierList", {}).get("CID", [])
        if not isinstance(cids, list):
            cids = []

        results = [str(cid) for cid in cids[:retmax]]
        self.cache.set(cache_key, results)
        return results

    def fetch_pubchem_compounds(self, cids: list[str]) -> list[dict[str, object]]:
        if not cids:
            return []

        cache_key = f"pubchem:fetch:{','.join(cids)}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return list(cached)

        properties = "Title,MolecularFormula,XLogP,TPSA,ConnectivitySMILES"
        payload = self._request_json(
            f"{PUBCHEM_BASE}/cid/{','.join(cids)}/property/{properties}/JSON",
            {},
        )
        rows = payload.get("PropertyTable", {}).get("Properties", [])
        if not isinstance(rows, list):
            rows = []

        parsed = [
            {
                "cid": str(row.get("CID", "")),
                "title": str(row.get("Title", "") or row.get("IUPACName", "") or row.get("CID", "")),
                "formula": str(row.get("MolecularFormula", "")),
                "xlogp": row.get("XLogP"),
                "tpsa": row.get("TPSA"),
                "smiles": str(
                    row.get("ConnectivitySMILES", "")
                    or row.get("CanonicalSMILES", "")
                    or row.get("IsomericSMILES", "")
                ),
            }
            for row in rows
        ]
        self.cache.set(cache_key, parsed)
        return parsed

    def normalize_pubchem_compound(self, compound: dict[str, object]) -> EvidenceItem:
        cid = str(compound.get("cid", ""))
        title = str(compound.get("title", "") or cid)
        formula = str(compound.get("formula", ""))
        xlogp = compound.get("xlogp")
        tpsa = compound.get("tpsa")
        smiles = str(compound.get("smiles", ""))
        summary_parts = []
        if formula:
            summary_parts.append(f"Formula {formula}")
        if xlogp not in (None, ""):
            summary_parts.append(f"XLogP {xlogp}")
        if tpsa not in (None, ""):
            summary_parts.append(f"TPSA {tpsa}")
        if smiles:
            summary_parts.append(f"SMILES {smiles}")

        return EvidenceItem(
            source="pubchem",
            pmid=cid,
            title=title,
            abstract="; ".join(summary_parts),
            journal="PubChem",
            pub_year=None,
            url=f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
            metadata={
                "cid": cid,
                "formula": formula,
                "xlogp": xlogp,
                "tpsa": tpsa,
                "smiles": smiles,
            },
        )

    def score_pubchem_evidence(
        self,
        item: EvidenceItem,
        question: str,
        target: str | None = None,
    ) -> float:
        question_tokens = self._tokenize(question)
        title_tokens = self._tokenize(item.title)
        abstract_tokens = self._tokenize(item.abstract)
        target_tokens = self._tokenize(target or "")

        score = 0.0
        score += len(question_tokens & title_tokens) * 3.0
        score += len(question_tokens & abstract_tokens) * 1.5
        score += len(target_tokens & (title_tokens | abstract_tokens)) * 4.0

        metadata_text = " ".join(str(value) for value in item.metadata.values() if value not in (None, ""))
        score += len(question_tokens & self._tokenize(metadata_text)) * 1.0
        return score

    def collect_pubchem_evidence(
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
                    cids = self.search_pubchem(query, retmax=retmax)
                except HTTPError as exc:
                    if self._is_not_found_error(exc):
                        continue
                    return self._failure_packet(
                        query=active_query,
                        missing_reason="pubchem_request_failed",
                        exc=exc,
                    )
                if not cids:
                    continue

                compounds = self.fetch_pubchem_compounds(cids)
                items = [self.normalize_pubchem_compound(compound) for compound in compounds]
                scored = [
                    replace(item, score=self.score_pubchem_evidence(item, question, target=target))
                    for item in items
                ]
                scored.sort(key=lambda item: item.score, reverse=True)

                return EvidencePacket(
                    source="pubchem",
                    query=query,
                    items=scored[:top_k],
                    source_health="ok",
                )
        except (HTTPError, URLError) as exc:
            return self._failure_packet(
                query=active_query,
                missing_reason="pubchem_request_failed",
                exc=exc,
            )
        except JSONDecodeError as exc:
            return self._failure_packet(
                query=active_query,
                missing_reason="pubchem_response_parse_failed",
                exc=exc,
            )

        return EvidencePacket(
            source="pubchem",
            query=queries[-1] if queries else "",
            items=[],
            source_health="degraded",
            missing_reason="no_pubchem_hits",
        )

    def _request_json(self, url: str, params: dict[str, str]) -> dict[str, Any]:
        request = Request(url, headers={"User-Agent": self.tool})
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
            source="pubchem",
            query=query,
            items=[],
            source_health="degraded",
            missing_reason=missing_reason,
            diagnostics={"error": str(exc), "error_type": type(exc).__name__},
        )

    def _is_not_found_error(self, exc: HTTPError) -> bool:
        return exc.code == 404

    def _query_from_question(self, question: str) -> str:
        return " ".join(sorted(self._tokenize(question)))

    def _tokenize(self, text: str) -> set[str]:
        return {token.lower() for token in TOKEN_RE.findall(text or "")}
