from __future__ import annotations

import pytest

from app.agents.registry import AgentRegistry, build_agent_registry


class DummyAgent:
    def __init__(self, name: str) -> None:
        self.name = name


def test_agent_registry_resolves_supported_agents_and_preserves_order() -> None:
    registry = AgentRegistry(
        walter=DummyAgent("walter"),
        house=DummyAgent("house"),
        harvey=DummyAgent("harvey"),
    )

    assert registry.resolve("walter").name == "walter"
    assert registry.resolve("house").name == "house"
    assert registry.resolve("harvey").name == "harvey"
    assert registry.available_agent_ids == ["walter", "house", "harvey"]
    assert [agent.name for agent in registry.resolve_many(["harvey", "walter"])] == [
        "harvey",
        "walter",
    ]


def test_agent_registry_accepts_whitespace_and_case_insensitive_ids() -> None:
    registry = AgentRegistry(
        walter=DummyAgent("walter"),
        house=DummyAgent("house"),
        harvey=DummyAgent("harvey"),
    )

    assert registry.resolve(" Walter ").name == "walter"
    assert registry.resolve("HOUSE").name == "house"


def test_agent_registry_raises_helpful_error_for_unknown_agent() -> None:
    registry = AgentRegistry(
        walter=DummyAgent("walter"),
        house=DummyAgent("house"),
        harvey=DummyAgent("harvey"),
    )

    with pytest.raises(ValueError, match="Supported agent ids"):
        registry.resolve("router")


def test_build_agent_registry_requires_all_supported_agents() -> None:
    with pytest.raises(TypeError):
        build_agent_registry(walter=DummyAgent("walter"), house=DummyAgent("house"))
