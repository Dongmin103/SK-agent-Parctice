# Module Context

- This directory owns regression coverage for deterministic clients, agent graphs, and shared models.
- Current coverage exists for the PubMed client and PubMed query planner agent.
- Test work follows one-feature-per-session discipline and must map to the active item in [`../architecture_progress_checklist_ko.md`](../architecture_progress_checklist_ko.md).

# Tech Stack & Constraints

- Use `pytest`.
- Tests must run from the project-local `.venv`.
- Default test mode is offline and deterministic.
- Network and Bedrock integration tests must be opt-in, not part of the default suite.
- User workflow shorthand may say `npm test`, but this repository's canonical equivalent is `./.venv/bin/pytest -q` until a wrapper exists.

# Implementation Patterns

- Mirror production module boundaries in test files.
- Name tests after behaviors, not implementation trivia.
- Use stub runnables for LLM behavior and monkeypatch for client IO.
- Prefer small sample payloads over large fixture dumps.
- If a fallback path exists in production, it needs a direct test.

# Testing Strategy

- Run all tests: `./.venv/bin/pytest -q`
- Run one module: `./.venv/bin/pytest tests/test_pubmed_query_agent.py -q`
- Run one behavior cluster: `./.venv/bin/pytest tests/test_pubmed_client.py -k fallback -q`
- Required implementation order for new functionality:
  - write the test first
  - run the targeted test and confirm failure
  - implement the smallest change
  - rerun the targeted test
  - rerun the relevant regression slice
- Required coverage categories for new work:
  - happy path
  - malformed input
  - no-result fallback
  - deterministic ranking or selection behavior
  - structured output validation

# Local Golden Rules

## Do's
- Always test what happens when the LLM output is invalid or incomplete.
- Always test fallback and degraded states, not only success cases.
- Always keep tests readable enough to explain the intended architecture.
- Always update [`../architecture_progress_checklist_ko.md`](../architecture_progress_checklist_ko.md) when a feature slice changes test status from not started to in progress or done.

## Don'ts
- Do not require live AWS credentials in the default test suite.
- Do not couple tests to ephemeral timestamps unless asserted loosely.
- Do not create fixtures that hide the actual behavior under test.
- Do not implement production code before a failing test exists for the active change.
