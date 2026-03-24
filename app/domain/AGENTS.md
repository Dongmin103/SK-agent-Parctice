# Module Context

- This directory owns shared data contracts across clients, agents, workflows, and APIs.
- Current implemented models include PubMed query input, raw article records, evidence items, and evidence packets.
- This module should stay framework-light and free of external IO.

# Tech Stack & Constraints

- Use dataclasses or Pydantic models when validation is needed.
- Keep imports lightweight. Domain models must not depend on LangChain, LangGraph, HTTP clients, or boto3.
- Shared models should be stable enough to support both consult and executive workflows.

# Implementation Patterns

- Put cross-module contracts here first before wiring orchestration.
- Add new fields only when there is a clear downstream need.
- Prefer additive evolution over breaking renames.
- Group related contracts by workflow concern:
  - evidence
  - prediction
  - routing
  - agent findings
  - decision drafts

# Testing Strategy

- Validate default values, serialization assumptions, and backward-compatible construction.
- If moving from dataclasses to Pydantic, add tests that prove existing callers still work.
- Use domain-level tests to lock down schema expectations before expanding API layers.

# Local Golden Rules

## Do's
- Always keep model names explicit and domain-specific.
- Always make optionality intentional.
- Always prefer shared contracts over ad hoc dictionaries passed across modules.

## Don'ts
- Do not add network, filesystem, or AWS logic here.
- Do not mix domain objects with prompt text or UI formatting.
- Do not make a domain model own orchestration behavior.
