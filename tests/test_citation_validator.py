from __future__ import annotations

from app.agents.citation_validator import CitationValidator
from app.domain.models import AgentFinding


def test_validate_reports_complete_when_every_finding_has_citations() -> None:
    validator = CitationValidator()
    findings = [
        AgentFinding(
            agent_id="walter",
            summary="Walter summary",
            citations=["https://example.org/a", "https://example.org/b"],
        ),
        AgentFinding(
            agent_id="house",
            summary="House summary",
            citations=["https://example.org/b", "https://example.org/c"],
        ),
    ]

    result = validator.validate(findings)

    assert result.complete is True
    assert result.missing_agent_ids == []
    assert result.total_citations == 3


def test_validate_preserves_order_for_missing_citations_and_counts_unique_urls() -> None:
    validator = CitationValidator()
    findings = [
        AgentFinding(agent_id="walter", summary="Walter summary", citations=[]),
        AgentFinding(
            agent_id="house",
            summary="House summary",
            citations=["https://example.org/a", "https://example.org/a"],
        ),
        AgentFinding(agent_id="harvey", summary="Harvey summary", citations=[]),
    ]

    result = validator.validate(findings)

    assert result.complete is False
    assert result.missing_agent_ids == ["walter", "harvey"]
    assert result.total_citations == 1


def test_validate_treats_empty_findings_as_incomplete_with_no_citations() -> None:
    validator = CitationValidator()

    result = validator.validate([])

    assert result.complete is False
    assert result.missing_agent_ids == []
    assert result.total_citations == 0


def test_validate_treats_blank_citation_entries_as_missing() -> None:
    validator = CitationValidator()
    findings = [
        AgentFinding(agent_id="walter", summary="Walter summary", citations=["   "]),
        AgentFinding(
            agent_id="house",
            summary="House summary",
            citations=["https://example.org/a"],
        ),
    ]

    result = validator.validate(findings)

    assert result.complete is False
    assert result.missing_agent_ids == ["walter"]
    assert result.total_citations == 1
