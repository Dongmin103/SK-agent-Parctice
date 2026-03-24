# Module Context

- This directory owns deterministic integrations with external systems.
- Clients here fetch, parse, normalize, score, and cache data from external services.
- Current implemented client: `pubmed.py`.

# Tech Stack & Constraints

- Prefer Python standard library or lightweight clients for deterministic external API work.
- Use `boto3` for AWS service access when implementing model or infrastructure clients such as TxGemma on SageMaker.
- Keep Bedrock model invocation out of clients unless the client is explicitly an AWS model adapter.
- PubMed integration must use NCBI E-utilities, not scraping.
- Treat each client as an adapter with stable output contracts defined in `app/domain`.

# Implementation Patterns

- One client per external system.
- Each client should expose a small public API:
  - build or accept request inputs
  - perform search or fetch
  - normalize into shared models
  - expose a single top-level collection method where useful
- Keep raw parsing helpers private.
- Use explicit cache keys and TTLs.
- Return normalized domain objects, not raw JSON or XML, from public methods.
- Model or evidence clients may expose dry-run and low-level helper methods, but the top-level collector should stay simple.

# Testing Strategy

- Unit-test parsing, normalization, ranking, caching, and fallback search behavior.
- Mock network calls at the client boundary.
- Test both hit and no-hit cases.
- Test malformed and missing-field responses where realistic.
- Keep tests offline by default.

# Local Golden Rules

## Do's
- Always normalize external responses into domain models before returning.
- Always make partial failure observable through return values or metadata.
- Always preserve enough source metadata for citation and debugging.
- Always prefer deterministic ranking before adding an LLM reranker.

## Don'ts
- Do not return provider-specific payloads to agent modules.
- Do not bury cache behavior in unrelated helper functions.
- Do not let client code depend on UI, FastAPI, or prompt templates.
- Do not add new evidence sources without tests for parsing and normalization.
