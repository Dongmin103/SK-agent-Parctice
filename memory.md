# Working Memory

This file is a supporting memory for repeated implementation pitfalls and proven fixes.
It is not the architecture task ledger. Use `architecture_progress_checklist_ko.md` for feature status.

## TxGemma Solubility Tuning

Date: 2026-03-24
Scope: `TxGemmaClient` prediction flow, with emphasis on the `solubility` signal.

### Problem Pattern

- The SageMaker-hosted TxGemma endpoint is reachable and returns text, but `solubility` was unstable.
- Prompt-only variants often returned unexplained numeric strings such as `811`, `808`, `793`, `761`, `734`, `621`, `609`, or malformed text like SMILES fragments.
- Those values must not be heuristically mapped to a signal.
- Applying TGI grammar constraints to every property degraded other live signals.

### What Did Not Work Reliably

- Rewording the `solubility` prompt into plain natural language classification only.
- Asking for JSON output by prompt wording alone.
- Asking for numeric `logS` output by prompt wording alone.
- Applying TGI `grammar` to all prediction properties.

### What Worked

- Keep the overall prediction workflow property-by-property.
- Constrain only the `solubility` property with TGI JSON grammar.
- Keep non-solubility properties on the previous prompt path without grammar constraints.
- Use a `solubility` prompt with few-shot examples and plain `Answer:` format:
  - `CO -> (C)`
  - `CCCCCCCC -> (A)`
- Normalize only validated outputs:
  - option tokens such as `(A)`, `(B)`, `(C)`
  - numeric values only if they are proper floats within the allowed `logS` range

### Current Safe Design

- `app/domain/prediction_registry.py`
  - `solubility` uses `few_shot_examples`
  - `solubility` sets `constrain_with_grammar=True`
- `app/clients/txgemma.py`
  - `_build_grammar(...)` is enabled only when `spec.constrain_with_grammar` is true
  - in current design that means `solubility` only

### If This Breaks Again

1. Confirm env wiring, not hardcoded endpoint values:
   - `TXGEMMA_SAGEMAKER_ENDPOINT_NAME` or `TXGEMMA_ENDPOINT_NAME`
   - `TXGEMMA_AWS_REGION` or AWS region fallback vars
2. Run the targeted offline tests:
   - `./.venv/bin/pytest tests/test_txgemma_client.py tests/test_txgemma_registry.py -q`
3. Run the opt-in live test:
   - `TXGEMMA_SAGEMAKER_ENDPOINT_NAME=... TXGEMMA_AWS_REGION=... ./.venv/bin/pytest tests/test_txgemma_client.py -k live -q`
4. If `solubility` is missing again:
   - verify that grammar is still applied only to `solubility`
   - verify that the `solubility` prompt still includes the few-shot examples
   - do not introduce heuristic mappings for `811`-style numeric garbage
5. If other properties regress after a change:
   - first suspect grammar leakage to non-solubility properties

### Last Known Good Outcome

- Offline:
  - `tests/test_txgemma_client.py tests/test_txgemma_registry.py` passed
  - related agent/domain regression slice passed
- Live:
  - `solubility` produced a valid signal
  - full bundle reached `10/10` signals with `missing_signals=[]`

## TxGemma 22-Signal Recovery Tuning

Date: 2026-03-24
Scope: TDC ADMET 22 migration 이후 live에서 missing이던 `Caco2`, `Lipophilicity`, `Solubility`, `PPBR`, `LD50`.

### Problem Pattern

- Exact task-level 22-signal migration 직후 live endpoint는 위 5개 task에 대해 signal을 만들지 못했다.
- Numeric-style prompts에 대해 endpoint는 `692`, `533`, `701`, `940`, `339` 같은 3-digit 문자열을 자주 반환했다.
- 특히 `Solubility`는 기존 memory의 `811`류 패턴과 동일하게 unsafe numeric garbage로 취급해야 했다.

### What Worked

- `Caco2`, `Lipophilicity`, `Solubility`, `PPBR`, `LD50`는 numeric prompt보다 classification prompt가 더 안정적이었다.
- `PPBR`, `LD50`는 classification prompt만으로는 불안정했고, few-shot 예시를 추가해야 안정화됐다.
- `TGI grammar`를 전체 property에 넓게 적용하지 않고, 위 5개 recovery task와 `Solubility`류 classification task에만 제한적으로 적용하면 live recover가 가능했다.
- `PPBR`의 raw `999`류 숫자는 heuristic scaling으로 해석하지 않고, prompt를 few-shot classification으로 바꿔 해결했다.

### Current Safe Design

- `app/domain/prediction_registry.py`
  - `Caco2`, `Lipophilicity`, `Solubility`, `PPBR`, `LD50`는 A/B/C classification answer options를 사용
  - `Solubility`, `Caco2`, `PPBR`, `LD50`는 few-shot examples 포함
  - 위 5개는 `constrain_with_grammar=True`
- `app/clients/txgemma.py`
  - `_build_grammar(...)`는 여전히 `spec.constrain_with_grammar`가 true인 task에만 적용

### If This Breaks Again

1. 먼저 live raw output을 property별로 확인한다.
   - 숫자 3자리 garbage가 보이면 numeric parsing을 늘리지 말고 prompt mismatch를 의심한다.
2. `./.venv/bin/pytest tests/test_txgemma_registry.py tests/test_txgemma_client.py -q`를 먼저 돌린다.
3. live smoke에서 `signals + missing_signals == 22`를 확인한다.
4. `PPBR`가 다시 빠지면 numeric scaling heuristic을 넣지 말고 few-shot classification prompt가 유지되는지부터 확인한다.

### Last Known Good Outcome

- Offline:
  - `./.venv/bin/pytest -q` passed
- Live:
  - direct `TxGemmaClient` invoke reached `22/22`
  - missing signals recovered: `Caco2`, `Lipophilicity`, `Solubility`, `PPBR`, `LD50`
