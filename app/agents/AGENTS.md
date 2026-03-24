# Module Context

- This directory owns agent logic, orchestration, planner chains, graph state, and Bedrock-driven structured reasoning.
- Agents here decide how to plan, route, revise, and synthesize. They do not own raw external API access.
- Current implemented file: `pubmed_query_agent.py`.

# Tech Stack & Constraints

- Use LangChain, LangGraph, Pydantic, and Bedrock integrations here.
- Use `ChatBedrockConverse` for Bedrock chat models unless there is a concrete reason not to.
- Keep model IDs configurable through environment variables or explicit constructor arguments.
- Prefer Claude Sonnet 4.5 for planning and structured reasoning tasks unless cost or availability requires a different Bedrock model.
- Never hardcode AWS credentials.

# Implementation Patterns

- Keep a clean separation between:
  - prompt construction
  - structured output schema
  - deterministic validation
  - graph orchestration
- Use Pydantic models for all LLM outputs.
- Use graph nodes for meaningful state transitions only. Good nodes:
  - bootstrap
  - plan
  - validate
  - dry_run
  - revise
  - finalize
- Keep fallbacks explicit. Graphs must be inspectable and testable without real Bedrock access.
- If an agent output influences downstream API calls, compile and sanitize the output before use.
- Agent modules may depend on `app.clients` and `app.domain`. They must not own XML parsing, HTTP client code, or raw search result normalization.

# Testing Strategy

- Unit-test planner and reviser behavior with stub runnables.
- Cover:
  - structured output acceptance
  - validation failures
  - fallback behavior
  - revision loops
  - query selection logic
- Do not require real Bedrock access for unit tests.
- Add a separate optional integration test only when credentials and region are explicitly available.

# Local Golden Rules

## Do's
- Always keep agent outputs narrow and machine-readable.
- Always include deterministic fallback paths.
- Always record enough metadata to explain why a query or route was chosen.
- Always keep graph state minimal and serializable.

## Don'ts
- Do not let agents call PubMed or other evidence APIs directly.
- Do not mix prompt authoring and parsing logic into one monolithic function.
- Do not return free-form prose when downstream code expects structured decisions.
- Do not add new graph nodes without a concrete state transition reason.
