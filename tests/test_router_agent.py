from __future__ import annotations

from app.domain.models import PredictionBundle, PredictionSignal
from app.agents.router_agent import ConsultRouterAgent, RoutingDecision


class SequenceRunnable:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        if not self._responses:
            raise RuntimeError("No more stubbed responses")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_prediction_bundle() -> PredictionBundle:
    return PredictionBundle(
        signals=[
            PredictionSignal(
                name="hERG",
                value="elevated",
                confidence=0.82,
                risk_level="high",
            )
        ]
    )


def test_router_agent_accepts_structured_output_from_runnable() -> None:
    prediction_bundle = make_prediction_bundle()
    runnable = SequenceRunnable(
        [
            {
                "question_type": "safety_pk",
                "selected_agents": ["house"],
                "routing_reason": "The question focuses on hERG and DDI risk.",
                "confidence": 0.84,
            }
        ]
    )
    agent = ConsultRouterAgent(router_runnable=runnable)

    result = agent.route(
        question="이 화합물의 hERG/DDI 위험은?",
        target="KRAS G12C",
        compound_name="ABC-101",
        prediction_bundle=prediction_bundle,
    )

    assert isinstance(result, RoutingDecision)
    assert result.question_type == "safety_pk"
    assert result.selected_agents == ["house"]
    assert result.routing_reason == "The question focuses on hERG and DDI risk."
    assert result.confidence == 0.84
    assert result.fallback_used is False
    assert runnable.calls[0]["question"] == "이 화합물의 hERG/DDI 위험은?"
    assert runnable.calls[0]["prediction_bundle"]["signals"][0]["name"] == "hERG"


def test_router_agent_falls_back_to_all_experts_when_output_is_invalid() -> None:
    runnable = SequenceRunnable(
        [
            {
                "question_type": "unsupported",
                "selected_agents": "house",
                "routing_reason": "",
                "confidence": 0.9,
            }
        ]
    )
    agent = ConsultRouterAgent(router_runnable=runnable)

    result = agent.route(question="이 후보의 개발 리스크를 전체적으로 봐줘.")

    assert result.question_type == "multi_expert"
    assert result.selected_agents == ["walter", "house", "harvey"]
    assert result.fallback_used is True
    assert "fallback" in result.routing_reason.lower()
    assert result.confidence < 0.6


def test_router_agent_uses_keyword_rules_when_llm_is_unavailable() -> None:
    agent = ConsultRouterAgent()

    result = agent.route(question="구조 최적화와 SAR 관점에서 어떤 리스크가 있나?")

    assert result.question_type == "structure_sar"
    assert result.selected_agents == ["walter"]
    assert result.fallback_used is True
    assert "keyword" in result.routing_reason.lower()
    assert result.confidence > 0.5


def test_router_agent_uses_multiple_experts_for_mixed_korean_review_question() -> None:
    agent = ConsultRouterAgent()

    result = agent.route(
        question="lead compound 진입 전 체크포인트 - 화학적 물성, 안전 프로파일, 개발 전략을 한 번에 검토해주세요."
    )

    assert result.question_type == "multi_expert"
    assert result.selected_agents == ["walter", "house", "harvey"]
    assert result.fallback_used is True
    assert "multiple expert domains" in result.routing_reason.lower()


def test_router_agent_routes_selected_agents_from_question_type() -> None:
    runnable = SequenceRunnable(
        [
            {
                "question_type": "clinical_regulatory",
                "selected_agents": ["walter", "harvey", "harvey"],
                "routing_reason": "The question is about approval strategy.",
                "confidence": 0.91,
            }
        ]
    )
    agent = ConsultRouterAgent(router_runnable=runnable)

    result = agent.route(question="FDA 승인 전략은?", target="KRAS G12C")

    assert result.question_type == "clinical_regulatory"
    assert result.selected_agents == ["harvey"]
    assert result.fallback_used is False


def test_router_agent_uses_all_experts_when_confidence_is_too_low() -> None:
    runnable = SequenceRunnable(
        [
            {
                "question_type": "structure_sar",
                "selected_agents": ["walter"],
                "routing_reason": "The question is about chemistry optimization.",
                "confidence": 0.12,
            }
        ]
    )
    agent = ConsultRouterAgent(router_runnable=runnable)

    result = agent.route(question="구조 최적화 관점에서 리스크는?")

    assert result.question_type == "multi_expert"
    assert result.selected_agents == ["walter", "house", "harvey"]
    assert result.fallback_used is True
    assert "confidence" in result.routing_reason.lower()
