from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


@dataclass(slots=True)
class ParallelTaskResult:
    agent_id: str
    ok: bool
    output: Any = None
    error: str | None = None


@dataclass(slots=True)
class ParallelExecutionReport:
    tasks: list[ParallelTaskResult] = field(default_factory=list)

    @property
    def outputs(self) -> list[Any]:
        return [task.output for task in self.tasks if task.ok]

    @property
    def failures(self) -> list[ParallelTaskResult]:
        return [task for task in self.tasks if not task.ok]


class ParallelExecutor:
    """Run selected expert agents against shared kwargs while preserving order."""

    def __init__(
        self,
        agents_by_id: Mapping[str, Any] | None = None,
        *,
        resolver: Callable[[str], Any] | None = None,
        max_workers: int | None = None,
    ) -> None:
        self.agents_by_id = dict(agents_by_id or {})
        self.resolver = resolver
        self.max_workers = max_workers

    def execute(
        self,
        selected_agents: Sequence[str],
        *,
        question: str,
        agents_by_id: Mapping[str, Any] | None = None,
        **shared_kwargs: Any,
    ) -> ParallelExecutionReport:
        if not selected_agents:
            raise ValueError("selected_agents must not be empty")

        if len(selected_agents) == 1:
            return ParallelExecutionReport(
                tasks=[
                    self._run_task(
                        selected_agents[0],
                        question=question,
                        agents_by_id=agents_by_id,
                        shared_kwargs=shared_kwargs,
                    )
                ]
            )

        max_workers = self.max_workers or len(selected_agents)
        results: list[ParallelTaskResult | None] = [None] * len(selected_agents)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    self._run_task,
                    agent_id,
                    question=question,
                    agents_by_id=agents_by_id,
                    shared_kwargs=shared_kwargs,
                ): index
                for index, agent_id in enumerate(selected_agents)
            }

            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception as exc:  # pragma: no cover - defensive guard
                    agent_id = selected_agents[index]
                    results[index] = ParallelTaskResult(
                        agent_id=agent_id,
                        ok=False,
                        error=str(exc),
                    )

        return ParallelExecutionReport(tasks=[task for task in results if task is not None])

    def _run_task(
        self,
        agent_id: str,
        *,
        question: str,
        agents_by_id: Mapping[str, Any] | None,
        shared_kwargs: Mapping[str, Any],
    ) -> ParallelTaskResult:
        try:
            agent = self._resolve_agent(agent_id, agents_by_id=agents_by_id)
        except Exception as exc:
            return ParallelTaskResult(agent_id=agent_id, ok=False, error=str(exc))

        try:
            output = self._invoke_agent(
                agent,
                question=question,
                shared_kwargs=shared_kwargs,
            )
        except Exception as exc:
            return ParallelTaskResult(agent_id=agent_id, ok=False, error=str(exc))

        return ParallelTaskResult(agent_id=agent_id, ok=True, output=output)

    def _resolve_agent(self, agent_id: str, *, agents_by_id: Mapping[str, Any] | None = None) -> Any:
        if agents_by_id is not None and agent_id in agents_by_id:
            return agents_by_id[agent_id]
        if agent_id in self.agents_by_id:
            return self.agents_by_id[agent_id]
        if self.resolver is not None:
            return self.resolver(agent_id)
        raise ValueError(f"Unsupported agent id: {agent_id}")

    def _invoke_agent(
        self,
        agent: Any,
        *,
        question: str,
        shared_kwargs: Mapping[str, Any],
    ) -> Any:
        if hasattr(agent, "analyze") and callable(agent.analyze):
            return agent.analyze(question, **dict(shared_kwargs))
        if callable(agent):
            return agent(question, **dict(shared_kwargs))
        raise TypeError("Agent must expose an analyze(...) method or be callable")
