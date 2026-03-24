from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.agents.parallel_executor import ParallelExecutor


@dataclass(slots=True)
class FakeFinding:
    agent_id: str
    summary: str


class StubAgent:
    def __init__(self, agent_id: str, *, should_fail: bool = False) -> None:
        self.agent_id = agent_id
        self.should_fail = should_fail
        self.calls: list[dict[str, object]] = []

    def analyze(
        self,
        question: str,
        *,
        target: str | None = None,
        compound_name: str | None = None,
        evidence_packet=None,
        prediction_bundle=None,
    ) -> FakeFinding:
        payload = {
            "question": question,
            "target": target,
            "compound_name": compound_name,
            "evidence_packet": evidence_packet,
            "prediction_bundle": prediction_bundle,
        }
        self.calls.append(payload)
        if self.should_fail:
            raise RuntimeError(f"{self.agent_id} failed")
        return FakeFinding(agent_id=self.agent_id, summary=f"finding:{self.agent_id}")


def test_parallel_executor_preserves_selection_order_and_forwards_shared_kwargs() -> None:
    walter = StubAgent("walter")
    house = StubAgent("house")
    executor = ParallelExecutor({"walter": walter, "house": house})

    result = executor.execute(
        ["house", "walter"],
        question="심장독성 위험은?",
        target="KRAS G12C",
        compound_name="ABC-101",
        evidence_packet={"source": "pubmed"},
        prediction_bundle={"signals": []},
    )

    assert [task.agent_id for task in result.tasks] == ["house", "walter"]
    assert [task.ok for task in result.tasks] == [True, True]
    assert [task.output.summary for task in result.tasks] == [
        "finding:house",
        "finding:walter",
    ]
    assert house.calls[0]["question"] == "심장독성 위험은?"
    assert house.calls[0]["target"] == "KRAS G12C"
    assert house.calls[0]["compound_name"] == "ABC-101"
    assert walter.calls[0]["evidence_packet"] == {"source": "pubmed"}
    assert walter.calls[0]["prediction_bundle"] == {"signals": []}


def test_parallel_executor_returns_structured_failure_for_missing_or_failing_agents() -> None:
    house = StubAgent("house", should_fail=True)
    executor = ParallelExecutor({"house": house})

    result = executor.execute(
        ["walter", "house"],
        question="구조와 독성 리스크를 같이 봐줘.",
        target="KRAS G12C",
    )

    assert [task.agent_id for task in result.tasks] == ["walter", "house"]
    assert result.tasks[0].ok is False
    assert result.tasks[0].error == "Unsupported agent id: walter"
    assert result.tasks[1].ok is False
    assert result.tasks[1].error == "house failed"
    assert result.outputs == []
    assert result.failures == result.tasks


def test_parallel_executor_rejects_blank_selection() -> None:
    executor = ParallelExecutor({})

    with pytest.raises(ValueError, match="selected_agents must not be empty"):
        executor.execute([], question="임상 전략은?")
