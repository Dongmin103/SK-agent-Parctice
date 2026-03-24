from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from json import JSONDecodeError
from typing import Any
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.clients.pubmed import TTLCache
from app.domain.models import EvidenceItem, EvidencePacket

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]{1,}")


@dataclass(slots=True)
class ChEMBLMoleculeRaw:
    chembl_id: str
    pref_name: str
    canonical_smiles: str
    synonyms: list[str] = field(default_factory=list)
    alogp: str = ""
    full_mwt: str = ""
    psa: str = ""
    max_phase: str = ""
    first_approval: int | None = None


class ChEMBLClient:
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
        if compound_name and target:
            queries.append(f'"{compound_name}" {target}')
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

    def search_chembl(self, query: str, retmax: int = 10) -> list[str]:
        cache_key = f"chembl:search:{query}:{retmax}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return list(cached)

        payload = self._request_json(
            f"{CHEMBL_BASE}/molecule/search.json",
            {"q": query, "limit": str(retmax)},
        )
        molecules = payload.get("molecules", [])
        if not isinstance(molecules, list):
            molecules = []
        chembl_ids = [str(molecule.get("molecule_chembl_id", "")) for molecule in molecules]
        chembl_ids = [chembl_id for chembl_id in chembl_ids if chembl_id]

        self.cache.set(cache_key, chembl_ids)
        return chembl_ids

    def fetch_chembl_molecules(self, chembl_ids: list[str]) -> list[ChEMBLMoleculeRaw]:
        if not chembl_ids:
            return []

        cache_key = f"chembl:fetch:{','.join(chembl_ids)}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return list(cached)

        molecules = [
            self._parse_chembl_molecule(
                self._request_json(f"{CHEMBL_BASE}/molecule/{chembl_id}.json", {})
            )
            for chembl_id in chembl_ids
        ]
        self.cache.set(cache_key, molecules)
        return molecules

    def normalize_chembl_molecule(self, molecule: ChEMBLMoleculeRaw) -> EvidenceItem:
        summary_parts = []
        if molecule.alogp:
            summary_parts.append(f"AlogP {molecule.alogp}")
        if molecule.full_mwt:
            summary_parts.append(f"MW {molecule.full_mwt}")
        if molecule.psa:
            summary_parts.append(f"PSA {molecule.psa}")
        if molecule.max_phase:
            summary_parts.append(f"Phase {molecule.max_phase}")
        if molecule.first_approval:
            summary_parts.append(f"First approval {molecule.first_approval}")
        if molecule.synonyms:
            summary_parts.append(f"Synonyms {', '.join(molecule.synonyms)}")

        return EvidenceItem(
            source="chembl",
            pmid=molecule.chembl_id,
            title=molecule.pref_name or molecule.chembl_id,
            abstract="; ".join(summary_parts),
            journal="ChEMBL",
            pub_year=molecule.first_approval,
            url=f"https://www.ebi.ac.uk/chembl/compound_report_card/{molecule.chembl_id}/",
            metadata={
                "chembl_id": molecule.chembl_id,
                "pref_name": molecule.pref_name,
                "canonical_smiles": molecule.canonical_smiles,
                "synonyms": ", ".join(molecule.synonyms),
                "alogp": molecule.alogp,
                "full_mwt": molecule.full_mwt,
                "psa": molecule.psa,
                "max_phase": molecule.max_phase,
            },
        )

    def score_chembl_evidence(
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

        if item.pub_year is not None and item.pub_year >= 2000:
            score += 1.0
        return score

    def collect_chembl_evidence(
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
                chembl_ids = self.search_chembl(query, retmax=retmax)
                if not chembl_ids:
                    continue

                molecules = self.fetch_chembl_molecules(chembl_ids)
                items = [self.normalize_chembl_molecule(molecule) for molecule in molecules]
                scored = [
                    replace(item, score=self.score_chembl_evidence(item, question, target=target))
                    for item in items
                ]
                scored.sort(key=lambda item: item.score, reverse=True)

                return EvidencePacket(
                    source="chembl",
                    query=query,
                    items=scored[:top_k],
                    source_health="ok",
                )
        except (HTTPError, URLError) as exc:
            return self._failure_packet(
                query=active_query,
                missing_reason="chembl_request_failed",
                exc=exc,
            )
        except JSONDecodeError as exc:
            return self._failure_packet(
                query=active_query,
                missing_reason="chembl_response_parse_failed",
                exc=exc,
            )

        return EvidencePacket(
            source="chembl",
            query=queries[-1] if queries else "",
            items=[],
            source_health="degraded",
            missing_reason="no_chembl_hits",
        )

    def _parse_chembl_molecule(self, payload: dict[str, object]) -> ChEMBLMoleculeRaw:
        properties = payload.get("molecule_properties") or {}
        structures = payload.get("molecule_structures") or {}
        synonyms = payload.get("synonyms") or payload.get("molecule_synonyms") or []
        if not isinstance(synonyms, list):
            synonyms = []
        synonym_values: list[str] = []
        for synonym in synonyms:
            if isinstance(synonym, dict):
                value = synonym.get("molecule_synonym", "")
            else:
                value = synonym
            if value:
                synonym_values.append(str(value))

        first_approval = payload.get("first_approval")
        return ChEMBLMoleculeRaw(
            chembl_id=str(payload.get("molecule_chembl_id", "")),
            pref_name=str(payload.get("pref_name", "")),
            canonical_smiles=str(structures.get("canonical_smiles", "")),
            synonyms=synonym_values,
            alogp=str(properties.get("alogp", "")),
            full_mwt=str(properties.get("full_mwt", "")),
            psa=str(properties.get("psa", "")),
            max_phase=str(payload.get("max_phase", "")),
            first_approval=int(first_approval) if isinstance(first_approval, int) else None,
        )

    def _failure_packet(
        self,
        *,
        query: str,
        missing_reason: str,
        exc: Exception,
    ) -> EvidencePacket:
        return EvidencePacket(
            source="chembl",
            query=query,
            items=[],
            source_health="degraded",
            missing_reason=missing_reason,
            diagnostics={"error": str(exc), "error_type": type(exc).__name__},
        )

    def _request_json(self, url: str, params: dict[str, str]) -> dict[str, Any]:
        full_url = f"{url}?{urlencode(params)}" if params else url
        request = Request(full_url, headers={"User-Agent": self.tool})
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def _query_from_question(self, question: str) -> str:
        return " ".join(sorted(self._tokenize(question)))

    def _tokenize(self, text: str) -> set[str]:
        return {token.lower() for token in TOKEN_RE.findall(text or "")}
