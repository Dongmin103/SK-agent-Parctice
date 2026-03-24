# Project Context & Operations

## Business Context
- This repository is an Agentic AI drug discovery PoC.
- The current implemented slice is PubMed evidence retrieval plus a LangChain/LangGraph query planning agent.
- The target architecture is a central orchestration system that combines TxGemma prediction, evidence collection, routing, expert agents, and response synthesis.
- The current architecture source of truth is [`./architecture_progress_checklist_ko.md`](./architecture_progress_checklist_ko.md). Treat it as the project TODO ledger and update work against that checklist, not memory.
- The target implementation spec is documented in [`./skbiopharm_agentic_ai_poc_implementation_plan_ko.md`](./skbiopharm_agentic_ai_poc_implementation_plan_ko.md). Use it when building out unimplemented layers.
- The target system visualization is documented in [`./architecture_overview_ko.md`](./architecture_overview_ko.md). Use it to keep runtime boundaries and module boundaries consistent.
- The agent topology and role interaction map are documented in [`./agent_structure_overview_ko.md`](./agent_structure_overview_ko.md). Use it when changing agent responsibilities or consult/executive call paths.

## Tech Stack
- Python application code.
- LangChain, LangGraph, Bedrock integration for agent logic.
- PubMed access through NCBI E-utilities.
- Local `.venv` for project dependencies.
- Current implemented modules live under `app/`.

## Operational Commands
- Create local environment: `python3 -m venv .venv`
- Install project dependencies: `./.venv/bin/pip install -e .`
- Run all tests: `./.venv/bin/pytest -q`
- Run one test file: `./.venv/bin/pytest tests/test_pubmed_client.py -q`
- Run one test case: `./.venv/bin/pytest tests/test_pubmed_query_agent.py -k revise -q`
- Show current file tree: `find . -maxdepth 3 -type f | sort`
- Run ad hoc Python against repo code: `PYTHONPATH=. ./.venv/bin/python - <<'PY' ... PY`
- User-level TDD shorthand says `npm test`. This repository does not currently have a Node test runner, so the equivalent command is `./.venv/bin/pytest -q`. If a wrapper is added later, keep it behaviorally aligned with the pytest suite.

## Session Workflow
- One session equals one feature. Do not implement multiple unrelated features in a single session.
- Start each work session by reading [`./architecture_progress_checklist_ko.md`](./architecture_progress_checklist_ko.md) and selecting the first active unchecked item for that feature slice.
- Keep the active scope small enough to finish with a red-green-refactor cycle in one session whenever possible.
- Subagents may be used to parallelize bounded tasks inside the same feature only. Do not use subagents to start unrelated checklist items.
- End each session by updating [`./architecture_progress_checklist_ko.md`](./architecture_progress_checklist_ko.md) so progress is explicit.

## MVP Execution Mode
- Default execution mode is MVP-first.
- The immediate goal is to prove the smallest useful local flow, not to complete every target-state optimization from the full implementation plan.
- Before adding cache layers, CloudWatch wiring, Secrets Manager integration, or Streamlit UI, prove that the local FastAPI server can run the intended workflow end to end.
- If the full implementation plan and the active MVP checklist ordering differ, follow [`./architecture_progress_checklist_ko.md`](./architecture_progress_checklist_ko.md) for sequencing.
- Treat cache, infra hardening, and nonessential operational polish as post-MVP unless they are strictly required to make the local flow run.

## Central Control & Delegation
- The root session is the control tower. It owns architecture, task decomposition, integration decisions, and final acceptance.
- Subagents are used for bounded work units only. A subagent must have a narrow responsibility, a clear write scope, and a concrete deliverable.
- The root session may dispatch coding and verification work to subagents, but only after defining the active feature boundary and the owning checklist item.
- Recommended work-unit split:
  - client integrations in `app/clients`
  - orchestration and Bedrock/LangGraph logic in `app/agents`
  - shared data contracts in `app/domain`
  - validation and regression coverage in `tests`
- The control tower must reconcile subagent outputs before code lands. Do not let subagents define cross-cutting interfaces independently.
- Every unit of work should map back to a checklist item in [`./architecture_progress_checklist_ko.md`](./architecture_progress_checklist_ko.md).

# Golden Rules

## Immutable
- Never hardcode secrets, API keys, tokens, or private endpoints in tracked files.
- Never bypass structured validation for LLM outputs that affect search planning, routing, or agent selection.
- Never let LLM code directly become the source of truth for external API calls. Compile and validate agent output in deterministic Python first.
- Never make a subagent responsible for both cross-module interface design and implementation in multiple modules at once.
- Never remove or overwrite user-authored work without explicit instruction.
- Never work on more than one feature track per session.
- Never skip the failing-test step for new functionality.

## Do's
- Always treat [`./architecture_progress_checklist_ko.md`](./architecture_progress_checklist_ko.md) as the execution ledger for architecture progress.
- Always treat [`./memory.md`](./memory.md) as supporting implementation memory only, not as the task ledger.
- Always start from the first active unchecked checklist item for the current feature.
- Always prefer the smallest runnable vertical slice over broader architectural completeness.
- Always verify the local server path before adding optimizations such as integrated caches or cloud-only operational features.
- Always consult [`./skbiopharm_agentic_ai_poc_implementation_plan_ko.md`](./skbiopharm_agentic_ai_poc_implementation_plan_ko.md) before introducing a new module, API, workflow, or agent that belongs to the target architecture.
- Always consult [`./architecture_overview_ko.md`](./architecture_overview_ko.md) before changing orchestration boundaries, data flow, or module ownership.
- Always consult [`./agent_structure_overview_ko.md`](./agent_structure_overview_ko.md) before changing Router, Walter, House, Harvey, Answer Composer, or CEO Synthesizer responsibilities.
- Always consult [`./memory.md`](./memory.md) first when blocked on a previously explored integration edge case.
- Always keep Bedrock prompts, structured output schemas, and deterministic validation separated.
- Always prefer deterministic code for API clients, parsing, normalization, scoring, and caching.
- Always add or update tests when changing `app/clients`, `app/agents`, or shared models.
- Always keep agent outputs machine-readable first and human-readable second.
- Always design new work so that consult and executive workflows can reuse the same core data structures.
- Always keep modules small and purpose-specific.
- Always implement features with TDD: write the test, run it to confirm failure, implement the smallest change, then run the suite again to confirm success.
- Always break work into the smallest testable vertical slice that can be completed in one session.

## Don'ts
- Do not call external APIs from expert agents if that logic belongs in clients.
- Do not place business logic in tests, notebooks, or scratch scripts.
- Do not introduce new framework dependencies without adding them to `pyproject.toml` and documenting why.
- Do not make query planning depend on UI concerns.
- Do not use vector RAG for PubMed until the deterministic E-utilities pipeline is proven insufficient.
- Do not let fallback paths be implicit. Fallback logic must be visible in code and testable.
- Do not chain multiple checklist items into one long-lived session unless they are part of the same feature and the same red-green-refactor loop.
- Do not implement cache or operations work ahead of the minimum local runnable workflow unless the checklist explicitly calls for it.

# Standards & References

## Code Standards
- Use ASCII by default.
- Prefer small pure functions with explicit inputs and outputs.
- Keep external IO isolated in clients.
- Keep orchestration logic in agents or workflows, not in domain models.
- Shared models must stay stable and minimal.

## Git Strategy
- Use focused branches when git is initialized. If a branch is created by the agent, use the `codex/` prefix.
- Keep commits small and scoped to one logical unit.
- Preferred commit format: `type(scope): summary`
- Examples:
  - `feat(pubmed): add query planner agent`
  - `test(agents): cover planner fallback`
  - `refactor(domain): split evidence models`

## References
- Architecture status: [`./architecture_progress_checklist_ko.md`](./architecture_progress_checklist_ko.md)
- Supporting implementation memory: [`./memory.md`](./memory.md)
- Target implementation plan: [`./skbiopharm_agentic_ai_poc_implementation_plan_ko.md`](./skbiopharm_agentic_ai_poc_implementation_plan_ko.md)
- Architecture visualization: [`./architecture_overview_ko.md`](./architecture_overview_ko.md)
- Agent topology visualization: [`./agent_structure_overview_ko.md`](./agent_structure_overview_ko.md)
- Project report and architectural rationale: [`./agentic_ai_poc_report_ko.md`](./agentic_ai_poc_report_ko.md)
- Package metadata: [`./pyproject.toml`](./pyproject.toml)

## Maintenance Policy
- If code and AGENTS rules drift, update the AGENTS files or explicitly flag the mismatch.
- If the checklist and repository state diverge, update the checklist in the same workstream.
- If the implementation plan or architecture visualization becomes stale, update those documents in the same workstream as the architectural change.
- If agent responsibilities or call relationships change, update `agent_structure_overview_ko.md` in the same workstream.
- If a new high-context zone appears, add a nested `AGENTS.md` for it rather than bloating the root file.
- If the user changes team workflow rules, reflect them in AGENTS before starting the next feature slice.

# Context Map (Action-Based Routing)

- **[Agent orchestration, Bedrock integration, LangChain/LangGraph logic](./app/agents/AGENTS.md)** — Query planners, routers, expert agents, graph orchestration, structured LLM outputs.
- **[External evidence and model clients](./app/clients/AGENTS.md)** — PubMed, PubChem, ChEMBL, ClinicalTrials.gov, openFDA, TxGemma, deterministic API access and normalization.
- **[Shared models and data contracts](./app/domain/AGENTS.md)** — Dataclasses, schemas, evidence packets, prediction bundles, cross-module contracts.
- **[Automated validation and regression coverage](./tests/AGENTS.md)** — Unit tests, mock patterns, fixture strategy, offline test rules.
