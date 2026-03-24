from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SUPPORTED_AGENT_IDS = ("walter", "house", "harvey")


def _normalize_agent_id(agent_id: str) -> str:
    return agent_id.strip().lower()


@dataclass(slots=True)
class AgentRegistry:
    """Stable lookup layer for consult/executive expert agents."""

    walter: Any
    house: Any
    harvey: Any

    @property
    def available_agent_ids(self) -> list[str]:
        return list(SUPPORTED_AGENT_IDS)

    def resolve(self, agent_id: str) -> Any:
        normalized = _normalize_agent_id(agent_id)
        if normalized == "walter":
            return self.walter
        if normalized == "house":
            return self.house
        if normalized == "harvey":
            return self.harvey
        raise ValueError(
            f"Unsupported agent id: {agent_id!r}. Supported agent ids: {', '.join(SUPPORTED_AGENT_IDS)}"
        )

    def resolve_many(self, agent_ids: list[str]) -> list[Any]:
        return [self.resolve(agent_id) for agent_id in agent_ids]

    def as_dict(self) -> dict[str, Any]:
        return {
            "walter": self.walter,
            "house": self.house,
            "harvey": self.harvey,
        }


def build_agent_registry(*, walter: Any, house: Any, harvey: Any) -> AgentRegistry:
    """Build a registry with the canonical expert agent ids."""
    return AgentRegistry(walter=walter, house=house, harvey=harvey)
