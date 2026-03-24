# SK Agent Practice

Agentic AI drug discovery PoC for local validation of `consult` and `executive` workflows.

This repository combines:

- TxGemma-based ADMET prediction
- multi-source evidence collection across PubMed, PubChem, ChEMBL, ClinicalTrials.gov, and openFDA
- Bedrock-based routing and expert agents
- consult/executive orchestration on FastAPI
- Streamlit UI for local runtime validation

## Current Scope

Implemented local vertical slice:

- compound preprocessing from SMILES
- TDC ADMET 22 prediction contract
- PubMed query planner agent with runtime fallback
- consult router plus Walter / House / Harvey expert agents
- executive synthesis flow
- streamed runtime trace/status in UI

## Architecture Documents

- `architecture_progress_checklist_ko.md`: execution ledger and current status
- `architecture_overview_ko.md`: system boundaries and module flow
- `agent_structure_overview_ko.md`: agent topology and responsibilities
- `skbiopharm_agentic_ai_poc_implementation_plan_ko.md`: target implementation plan

## Project Structure

```text
app/
  agents/      Bedrock agents, routing, synthesis
  api/         FastAPI app, settings, dependency wiring
  clients/     External evidence and model clients
  domain/      Shared models and prediction registry
  ui/          Streamlit local workbench
  workflows/   Consult and executive orchestration
tests/         Offline regression suite
```

## Local Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -e .
```

## Environment

The app reads both `.env` and `.env.local`. For local development, use `.env.local`.

Typical variables:

```bash
# TxGemma
TXGEMMA_SAGEMAKER_ENDPOINT_NAME=...
TXGEMMA_AWS_REGION=ap-southeast-2

# Bedrock
BEDROCK_AWS_REGION=ap-northeast-2
BEDROCK_ROUTER_MODEL_ID=global.anthropic.claude-sonnet-4-6
BEDROCK_WALTER_AGENT_MODEL_ID=global.anthropic.claude-sonnet-4-6
BEDROCK_HOUSE_AGENT_MODEL_ID=global.anthropic.claude-sonnet-4-6
BEDROCK_HARVEY_AGENT_MODEL_ID=global.anthropic.claude-sonnet-4-6
BEDROCK_PUBMED_QUERY_MODEL_ID=global.anthropic.claude-sonnet-4-6

# Optional
NCBI_API_KEY=...
```

## Run

FastAPI:

```bash
./.venv/bin/uvicorn app.api.main:create_runtime_app --factory --host 127.0.0.1 --port 8000
```

Streamlit:

```bash
./.venv/bin/streamlit run app/ui/main.py
```

## Test

```bash
./.venv/bin/pytest -q
```

## Notes

- Secrets and local runtime values are intentionally excluded from git.
- Bedrock Sonnet 4.6 should be configured through an inference profile such as `global.anthropic.claude-sonnet-4-6`, not the raw foundation model ID.
