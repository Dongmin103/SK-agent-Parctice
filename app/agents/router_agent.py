from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Literal, Protocol

from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError, field_validator

from app.domain.models import PredictionBundle

LOGGER = logging.getLogger(__name__)

ALLOWED_AGENTS = ("walter", "house", "harvey")
QUESTION_TYPE_TO_AGENTS: dict[str, list[str]] = {
    "structure_sar": ["walter"],
    "safety_pk": ["house"],
    "clinical_regulatory": ["harvey"],
    "multi_expert": list(ALLOWED_AGENTS),
}
CHEMISTRY_KEYWORDS = (
    "structure",
    "sar",
    "scaffold",
    "chemistry",
    "compound design",
    "solubility",
    "permeability",
    "lipophilicity",
    "logp",
    "logd",
    "구조",
    "물성",
    "최적화",
)
SAFETY_KEYWORDS = (
    "tox",
    "toxicity",
    "safety",
    "safety profile",
    "pk",
    "pd",
    "ddi",
    "cyp",
    "herg",
    "qt",
    "clearance",
    "half-life",
    "drug-drug",
    "독성",
    "안전",
    "안전성",
    "안전 프로파일",
    "상호작용",
    "반감기",
    "노출",
)
CLINICAL_KEYWORDS = (
    "clinical",
    "regulatory",
    "approval",
    "fda",
    "phase",
    "trial",
    "patient",
    "development strategy",
    "허가",
    "승인",
    "규제",
    "임상",
    "전략",
    "개발 전략",
    "개발전략",
)

ROUTER_SYSTEM_PROMPT = """You are the consult Router for a drug discovery assistant.

Rules:
- Return only the structured schema fields.
- Select from walter, house, harvey only.
- Use structure/SAR/property questions for Walter.
- Use safety/PK/PD/DDI/toxicity questions for House.
- Use clinical/regulatory/approval strategy questions for Harvey.
- For mixed or ambiguous questions, return question_type='multi_expert' and multiple agents.
"""

ROUTER_USER_TEMPLATE = """Question: {question}
Target: {target}
Compound name: {compound_name}
Prediction bundle: {prediction_bundle}

Classify the consult question and select the experts to call."""


class RouterDecisionOutput(BaseModel):
    question_type: Literal["structure_sar", "safety_pk", "clinical_regulatory", "multi_expert"]
    selected_agents: list[str] = Field(min_length=1, max_length=3)
    routing_reason: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("selected_agents")
    @classmethod
    def normalize_selected_agents(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            item = str(value).strip().lower()
            if not item or item in seen:
                continue
            if item not in ALLOWED_AGENTS:
                raise ValueError(f"unsupported agent: {item}")
            seen.add(item)
            cleaned.append(item)
        if not cleaned:
            raise ValueError("selected_agents must not be empty")
        return cleaned

    @field_validator("routing_reason")
    @classmethod
    def validate_routing_reason(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("routing_reason must not be blank")
        return cleaned


class RoutingDecision(BaseModel):
    question_type: Literal["structure_sar", "safety_pk", "clinical_regulatory", "multi_expert"]
    selected_agents: list[str]
    routing_reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    fallback_used: bool = False


class RouterRunnable(Protocol):
    def invoke(self, payload: Mapping[str, object]) -> RouterDecisionOutput | Mapping[str, object]:
        ...


class ConsultRouterAgent:
    """Route consult questions to the right expert agents."""

    def __init__(
        self,
        *,
        router_runnable: RouterRunnable | None = None,
        low_confidence_threshold: float = 0.6,
    ) -> None:
        self.router_runnable = router_runnable
        self.low_confidence_threshold = low_confidence_threshold

    @classmethod
    def from_bedrock(
        cls,
        *,
        model_id: str,
        region_name: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        low_confidence_threshold: float = 0.6,
    ) -> "ConsultRouterAgent":
        llm = ChatBedrockConverse(
            model=model_id,
            region_name=region_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        chain = (
            ChatPromptTemplate.from_messages(
                [("system", ROUTER_SYSTEM_PROMPT), ("user", ROUTER_USER_TEMPLATE)]
            )
            | llm.with_structured_output(RouterDecisionOutput)
        )
        return cls(
            router_runnable=chain,
            low_confidence_threshold=low_confidence_threshold,
        )

    def route(
        self,
        question: str,
        *,
        target: str | None = None,
        compound_name: str | None = None,
        prediction_bundle: PredictionBundle | None = None,
    ) -> RoutingDecision:
        payload = {
            "question": question,
            "target": target or "",
            "compound_name": compound_name or "",
            "prediction_bundle": self._serialize_prediction_bundle(prediction_bundle),
        }

        if self.router_runnable is None:
            return self._keyword_route(question)

        try:
            output = self.router_runnable.invoke(payload)
            if not isinstance(output, RouterDecisionOutput):
                output = RouterDecisionOutput.model_validate(output)
        except (ValidationError, ValueError, TypeError) as exc:
            LOGGER.warning("Router output validation failed; using all-expert fallback: %s", exc)
            return self._all_expert_fallback("Router fallback triggered because the structured output was invalid.")
        except Exception as exc:
            LOGGER.warning("Router runnable failed; using all-expert fallback: %s", exc)
            return self._all_expert_fallback("Router fallback triggered because the model call failed.")

        if output.confidence < self.low_confidence_threshold:
            return self._all_expert_fallback(
                "Router fallback triggered because the model confidence was below threshold."
            )

        selected_agents = self._selected_agents_for_question_type(output.question_type)
        return RoutingDecision(
            question_type=output.question_type,
            selected_agents=selected_agents,
            routing_reason=output.routing_reason,
            confidence=output.confidence,
            fallback_used=False,
        )

    def _keyword_route(self, question: str) -> RoutingDecision:
        lowered = question.lower()
        selected_agents: list[str] = []

        if any(keyword in lowered for keyword in CHEMISTRY_KEYWORDS):
            selected_agents.append("walter")
        if any(keyword in lowered for keyword in SAFETY_KEYWORDS):
            selected_agents.append("house")
        if any(keyword in lowered for keyword in CLINICAL_KEYWORDS):
            selected_agents.append("harvey")

        if not selected_agents:
            return self._all_expert_fallback(
                "Keyword fallback could not confidently classify the question, so all experts were selected."
            )

        if len(selected_agents) == 1:
            question_type = self._question_type_for_agent(selected_agents[0])
            return RoutingDecision(
                question_type=question_type,
                selected_agents=selected_agents,
                routing_reason="Keyword fallback matched the question to a single expert domain.",
                confidence=0.68,
                fallback_used=True,
            )

        return RoutingDecision(
            question_type="multi_expert",
            selected_agents=selected_agents,
            routing_reason="Keyword fallback matched multiple expert domains.",
            confidence=0.55,
            fallback_used=True,
        )

    def _question_type_for_agent(self, agent_id: str) -> str:
        for question_type, agent_ids in QUESTION_TYPE_TO_AGENTS.items():
            if agent_ids == [agent_id]:
                return question_type
        return "multi_expert"

    def _all_expert_fallback(self, reason: str) -> RoutingDecision:
        return RoutingDecision(
            question_type="multi_expert",
            selected_agents=list(ALLOWED_AGENTS),
            routing_reason=reason,
            confidence=0.35,
            fallback_used=True,
        )

    def _selected_agents_for_question_type(self, question_type: str) -> list[str]:
        selected_agents = QUESTION_TYPE_TO_AGENTS.get(question_type)
        if selected_agents is None:
            return list(ALLOWED_AGENTS)
        return list(selected_agents)

    def _serialize_prediction_bundle(
        self, prediction_bundle: PredictionBundle | None
    ) -> dict[str, object]:
        if prediction_bundle is None:
            return {"source": "", "signals": [], "generated_at": ""}
        return {
            "source": prediction_bundle.source,
            "generated_at": prediction_bundle.generated_at,
            "signals": [
                {
                    "name": signal.name,
                    "value": signal.value,
                    "unit": signal.unit or "",
                    "confidence": signal.confidence,
                    "risk_level": signal.risk_level or "",
                }
                for signal in prediction_bundle.signals
            ],
        }
