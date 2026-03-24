# Module Context

- This directory owns the Streamlit user interface for the local PoC.
- UI code should stay thin: it collects user input, calls the FastAPI layer, and renders machine-readable API results.
- The source of truth for workflow behavior remains the API and workflow layers, not the UI.

# Tech Stack & Constraints

- Use `streamlit` for the web UI shell.
- Prefer small pure presenter/helpers that are easy to test outside Streamlit.
- Keep network calls isolated in a UI client module.
- Do not duplicate business logic that already exists in `app/api`, `app/workflows`, or `app/domain`.

# Implementation Patterns

- Put request/response transport in `client.py`.
- Put payload-to-view transformations in `presenters.py`.
- Keep `app.py` focused on layout, interaction flow, and rendering.
- Make the app boot without immediately calling the API; requests should happen only on user submit.

# Testing Strategy

- Add deterministic unit tests for API client error handling and presenter output.
- Use one runtime smoke test to prove `streamlit run ...` can boot locally.
- Keep UI tests offline by mocking HTTP rather than requiring a live API server.

# Local Golden Rules

## Do's
- Always surface API errors clearly in the UI.
- Always preserve machine-readable values from the API before formatting them for display.
- Always keep consult and executive as distinct user flows.

## Don'ts
- Do not move workflow decisions into Streamlit callbacks.
- Do not hardcode secrets or endpoint-specific AWS values in the UI.
- Do not make page load depend on API availability.
