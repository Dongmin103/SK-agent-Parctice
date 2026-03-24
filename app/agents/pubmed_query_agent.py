from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from typing import Literal, Protocol, TypedDict

from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator

from app.clients.pubmed import PubMedClient
from app.domain.models import PubMedQueryInput

LOGGER = logging.getLogger(__name__)

TERM_SANITIZE_RE = re.compile(r"[^A-Za-z0-9\s\-\./+]")
SMILESISH_RE = re.compile(r"[=#@\[\]\\()/]")
ALLOWED_QUESTION_TYPES = {"safety", "pk", "regulatory", "complex"}
BEDROCK_REGION_ENV_VARS = ("BEDROCK_AWS_REGION", "AWS_REGION", "AWS_DEFAULT_REGION")


class PubMedQueryPlan(BaseModel):
    question_type: Literal["safety", "pk", "regulatory", "complex"]
    primary_terms: list[str] = Field(default_factory=list, max_length=8)
    optional_terms: list[str] = Field(default_factory=list, max_length=8)
    excluded_terms: list[str] = Field(default_factory=list, max_length=8)
    reasoning: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("primary_terms", "optional_terms", "excluded_terms")
    @classmethod
    def normalize_terms(cls, value: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for term in value:
            clean = " ".join(term.strip().split())
            if not clean:
                continue
            if clean.lower() in seen:
                continue
            seen.add(clean.lower())
            deduped.append(clean)
        return deduped


class QueryValidationResult(BaseModel):
    valid: bool
    issues: list[str] = Field(default_factory=list)
    sanitized_primary_terms: list[str] = Field(default_factory=list)
    sanitized_optional_terms: list[str] = Field(default_factory=list)
    sanitized_excluded_terms: list[str] = Field(default_factory=list)
    candidate_queries: list[str] = Field(default_factory=list)


class QueryDryRunResult(BaseModel):
    query: str
    hit_count: int
    accepted: bool


class PubMedQueryAgentResult(BaseModel):
    question_type: str
    selected_query: str
    candidate_queries: list[str]
    dry_run_results: list[QueryDryRunResult]
    reasoning: str
    validation_issues: list[str]
    revision_attempts: int
    used_llm: bool
    fallback_used: bool


class PubMedQueryAgentState(TypedDict, total=False):
    request: PubMedQueryInput
    baseline_queries: list[str]
    plan: PubMedQueryPlan | None
    validation: QueryValidationResult
    dry_run_results: list[QueryDryRunResult]
    revision_attempts: int
    final_result: PubMedQueryAgentResult
    llm_reasoning: str


class QueryPlanRunnable(Protocol):
    def invoke(self, payload: Mapping[str, object]) -> PubMedQueryPlan | Mapping[str, object]:
        ...


PLANNER_SYSTEM_PROMPT = """You design PubMed search strategies for drug discovery questions.

Rules:
- Return only structured fields requested by the schema.
- Use PubMed-friendly biomedical phrases, not raw SMILES.
- Prefer target, compound name, toxicity/PK/regulatory terms.
- Keep primary_terms concise and highly discriminative.
- Use optional_terms for broader synonyms.
- For complex questions, keep question_type='complex'.
- Never invent proprietary internal data sources.
"""

PLANNER_USER_TEMPLATE = """Question: {question}
Question type hint: {question_type}
Target: {target}
Compound name: {compound_name}
Prediction flags: {prediction_flags}

Create a PubMed query plan with primary terms, optional terms, and any terms to exclude."""

REVISER_SYSTEM_PROMPT = """You improve PubMed search strategies after validation failures or zero-hit dry runs.

Rules:
- Broaden the query carefully when there are zero hits.
- Keep target or compound terms when they are informative.
- Remove malformed, overly specific, or SMILES-like terms.
- Return only the structured schema fields.
"""

REVISER_USER_TEMPLATE = """Original question: {question}
Original question type: {question_type}
Target: {target}
Compound name: {compound_name}

Previous plan: {previous_plan}
Validation issues: {validation_issues}
Dry run results: {dry_run_results}

Revise the PubMed query plan to improve search quality."""


class PubMedQueryPlannerAgent:
    """LangGraph-powered query planner with validation and dry-run search loops."""

    def __init__(
        self,
        pubmed_client: PubMedClient,
        *,
        planner_runnable: QueryPlanRunnable | None = None,
        reviser_runnable: QueryPlanRunnable | None = None,
        max_revision_attempts: int = 2,
        retmax: int = 10,
    ) -> None:
        self.pubmed_client = pubmed_client
        self.planner_runnable = planner_runnable
        self.reviser_runnable = reviser_runnable or planner_runnable
        self.max_revision_attempts = max_revision_attempts
        self.retmax = retmax
        self.graph = self._build_graph()

    @classmethod
    def from_bedrock(
        cls,
        pubmed_client: PubMedClient,
        *,
        model_id: str,
        region_name: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 700,
        max_revision_attempts: int = 2,
        retmax: int = 10,
    ) -> "PubMedQueryPlannerAgent":
        planner_llm = ChatBedrockConverse(
            model=model_id,
            region_name=region_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        planner_chain = (
            ChatPromptTemplate.from_messages(
                [("system", PLANNER_SYSTEM_PROMPT), ("user", PLANNER_USER_TEMPLATE)]
            )
            | planner_llm.with_structured_output(PubMedQueryPlan)
        )
        reviser_chain = (
            ChatPromptTemplate.from_messages(
                [("system", REVISER_SYSTEM_PROMPT), ("user", REVISER_USER_TEMPLATE)]
            )
            | planner_llm.with_structured_output(PubMedQueryPlan)
        )
        return cls(
            pubmed_client,
            planner_runnable=planner_chain,
            reviser_runnable=reviser_chain,
            max_revision_attempts=max_revision_attempts,
            retmax=retmax,
        )

    def plan(self, request: PubMedQueryInput) -> PubMedQueryAgentResult:
        state: PubMedQueryAgentState = {
            "request": request,
            "revision_attempts": 0,
        }
        result = self.graph.invoke(state)
        return result["final_result"]

    def _build_graph(self):
        graph = StateGraph(PubMedQueryAgentState)
        graph.add_node("bootstrap", self._bootstrap_node)
        graph.add_node("plan", self._plan_node)
        graph.add_node("validate", self._validate_node)
        graph.add_node("dry_run", self._dry_run_node)
        graph.add_node("revise", self._revise_node)
        graph.add_node("finalize", self._finalize_node)

        graph.add_edge(START, "bootstrap")
        graph.add_edge("bootstrap", "plan")
        graph.add_edge("plan", "validate")
        graph.add_edge("validate", "dry_run")
        graph.add_conditional_edges(
            "dry_run",
            self._route_after_dry_run,
            {"revise": "revise", "finalize": "finalize"},
        )
        graph.add_edge("revise", "validate")
        graph.add_edge("finalize", END)
        return graph.compile()

    def _bootstrap_node(self, state: PubMedQueryAgentState) -> PubMedQueryAgentState:
        request = state["request"]
        baseline_queries = self.pubmed_client.build_query_candidates(
            request.question,
            request.question_type,
            target=request.target,
            compound_name=request.compound_name,
        )
        return {"baseline_queries": baseline_queries}

    def _plan_node(self, state: PubMedQueryAgentState) -> PubMedQueryAgentState:
        if self.planner_runnable is None:
            return {"plan": None, "llm_reasoning": "LLM planner unavailable; using deterministic fallback."}

        request = state["request"]
        payload = {
            "question": request.question,
            "question_type": request.question_type,
            "target": request.target or "",
            "compound_name": request.compound_name or "",
            "prediction_flags": request.prediction_flags or {},
        }
        try:
            plan = self.planner_runnable.invoke(payload)
        except Exception as exc:
            LOGGER.warning("Planner runnable failed, falling back to deterministic queries: %s", exc)
            return {"plan": None, "llm_reasoning": f"Planner failure: {exc}"}

        if isinstance(plan, PubMedQueryPlan):
            return {"plan": plan, "llm_reasoning": plan.reasoning}

        coerced = PubMedQueryPlan.model_validate(plan)
        return {"plan": coerced, "llm_reasoning": coerced.reasoning}

    def _validate_node(self, state: PubMedQueryAgentState) -> PubMedQueryAgentState:
        request = state["request"]
        baseline_queries = state["baseline_queries"]
        plan = state.get("plan")

        if plan is None:
            return {
                "validation": QueryValidationResult(
                    valid=False,
                    issues=["planner_unavailable"],
                    candidate_queries=baseline_queries,
                )
            }

        validation = self._validate_and_compile_plan(plan, request, baseline_queries)
        return {"validation": validation}

    def _dry_run_node(self, state: PubMedQueryAgentState) -> PubMedQueryAgentState:
        validation = state["validation"]
        dry_run_results: list[QueryDryRunResult] = []
        for query in validation.candidate_queries:
            hit_count = len(self.pubmed_client.search_pubmed(query, retmax=self.retmax))
            dry_run_results.append(
                QueryDryRunResult(query=query, hit_count=hit_count, accepted=hit_count > 0)
            )
        return {"dry_run_results": dry_run_results}

    def _route_after_dry_run(self, state: PubMedQueryAgentState) -> str:
        results = state["dry_run_results"]
        attempts = state["revision_attempts"]
        if any(result.hit_count > 0 for result in results):
            return "finalize"
        if self.reviser_runnable is None or attempts >= self.max_revision_attempts:
            return "finalize"
        return "revise"

    def _revise_node(self, state: PubMedQueryAgentState) -> PubMedQueryAgentState:
        if self.reviser_runnable is None:
            return {}

        request = state["request"]
        plan = state.get("plan")
        validation = state["validation"]
        dry_run_results = state["dry_run_results"]

        payload = {
            "question": request.question,
            "question_type": request.question_type,
            "target": request.target or "",
            "compound_name": request.compound_name or "",
            "previous_plan": plan.model_dump() if plan else {},
            "validation_issues": validation.issues,
            "dry_run_results": [item.model_dump() for item in dry_run_results],
        }

        try:
            revised_plan = self.reviser_runnable.invoke(payload)
            if not isinstance(revised_plan, PubMedQueryPlan):
                revised_plan = PubMedQueryPlan.model_validate(revised_plan)
        except Exception as exc:
            LOGGER.warning("Reviser runnable failed; finalizing with fallback queries: %s", exc)
            return {"revision_attempts": state["revision_attempts"] + 1}

        return {
            "plan": revised_plan,
            "llm_reasoning": revised_plan.reasoning,
            "revision_attempts": state["revision_attempts"] + 1,
        }

    def _finalize_node(self, state: PubMedQueryAgentState) -> PubMedQueryAgentState:
        validation = state["validation"]
        dry_run_results = state["dry_run_results"]
        baseline_queries = state["baseline_queries"]
        request = state["request"]

        best = max(dry_run_results, key=lambda item: item.hit_count, default=None)
        fallback_used = not validation.valid or best is None or best.hit_count == 0
        selected_query = (
            best.query
            if best is not None and best.hit_count > 0
            else (baseline_queries[0] if baseline_queries else "")
        )
        used_llm = state.get("plan") is not None

        final_result = PubMedQueryAgentResult(
            question_type=request.question_type,
            selected_query=selected_query,
            candidate_queries=validation.candidate_queries or baseline_queries,
            dry_run_results=dry_run_results,
            reasoning=state.get("llm_reasoning", ""),
            validation_issues=validation.issues,
            revision_attempts=state["revision_attempts"],
            used_llm=used_llm,
            fallback_used=fallback_used,
        )
        return {"final_result": final_result}

    def _validate_and_compile_plan(
        self,
        plan: PubMedQueryPlan,
        request: PubMedQueryInput,
        baseline_queries: list[str],
    ) -> QueryValidationResult:
        issues: list[str] = []

        if plan.question_type not in ALLOWED_QUESTION_TYPES:
            issues.append("invalid_question_type")

        primary_terms = self._sanitize_terms(plan.primary_terms, issues, prefix="primary")
        optional_terms = self._sanitize_terms(plan.optional_terms, issues, prefix="optional")
        excluded_terms = self._sanitize_terms(plan.excluded_terms, issues, prefix="excluded")

        if not primary_terms:
            issues.append("missing_primary_terms")

        candidate_queries = self._compile_queries(
            request=request,
            primary_terms=primary_terms,
            optional_terms=optional_terms,
            excluded_terms=excluded_terms,
        )

        if not candidate_queries:
            candidate_queries = baseline_queries
            issues.append("no_compiled_queries")

        return QueryValidationResult(
            valid=not issues,
            issues=issues,
            sanitized_primary_terms=primary_terms,
            sanitized_optional_terms=optional_terms,
            sanitized_excluded_terms=excluded_terms,
            candidate_queries=candidate_queries,
        )

    def _sanitize_terms(self, terms: list[str], issues: list[str], *, prefix: str) -> list[str]:
        sanitized: list[str] = []
        seen: set[str] = set()
        for term in terms:
            if SMILESISH_RE.search(term):
                issues.append(f"{prefix}_contains_smiles_like_term")
                continue
            clean = TERM_SANITIZE_RE.sub(" ", term).strip()
            clean = " ".join(clean.split())
            if not clean:
                continue
            if len(clean) > 80:
                issues.append(f"{prefix}_term_too_long")
                continue
            lowered = clean.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            sanitized.append(clean)
        return sanitized

    def _compile_queries(
        self,
        *,
        request: PubMedQueryInput,
        primary_terms: list[str],
        optional_terms: list[str],
        excluded_terms: list[str],
    ) -> list[str]:
        base_terms = [term for term in (request.compound_name, request.target) if term and term.strip()]
        base_expr = ""
        if base_terms:
            quoted = [self._quote(term) for term in base_terms]
            base_expr = f"({' OR '.join(quoted)})" if len(quoted) > 1 else quoted[0]

        primary_expr = self._terms_to_or_clause(primary_terms)
        combined_expr = self._terms_to_or_clause(primary_terms + optional_terms)
        excluded_expr = self._terms_to_or_clause(excluded_terms)

        queries: list[str] = []
        if base_expr and primary_expr:
            queries.append(f"{base_expr} AND {primary_expr}")
        if base_expr and combined_expr and combined_expr != primary_expr:
            queries.append(f"{base_expr} AND {combined_expr}")
        if primary_expr and not base_expr:
            queries.append(primary_expr)
        if base_expr:
            queries.append(base_expr)

        if excluded_expr:
            queries = [f"{query} NOT {excluded_expr}" for query in queries]

        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            if query not in seen:
                seen.add(query)
                deduped.append(query)
        return deduped

    def _terms_to_or_clause(self, terms: list[str]) -> str:
        if not terms:
            return ""
        quoted = [self._quote(term) for term in terms]
        if len(quoted) == 1:
            return quoted[0]
        return f"({' OR '.join(quoted)})"

    def _quote(self, term: str) -> str:
        escaped = " ".join(term.split()).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'


def build_bedrock_pubmed_query_agent(
    pubmed_client: PubMedClient,
    *,
    model_id: str | None = None,
    region_name: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 700,
    max_revision_attempts: int = 2,
    retmax: int = 10,
) -> PubMedQueryPlannerAgent:
    resolved_model_id = model_id or os.environ.get("BEDROCK_PUBMED_QUERY_MODEL_ID")
    if not resolved_model_id:
        raise ValueError(
            "A Bedrock model id is required. Set BEDROCK_PUBMED_QUERY_MODEL_ID or pass model_id."
        )
    resolved_region_name = region_name or next(
        (os.environ.get(name) for name in BEDROCK_REGION_ENV_VARS if os.environ.get(name)),
        None,
    )
    return PubMedQueryPlannerAgent.from_bedrock(
        pubmed_client,
        model_id=resolved_model_id,
        region_name=resolved_region_name,
        temperature=temperature,
        max_tokens=max_tokens,
        max_revision_attempts=max_revision_attempts,
        retmax=retmax,
    )
