# Agentic AI 신약개발 PoC 아키텍처 진행 체크리스트

기준 시점: 현재 저장소 상태 기준  
상태 표기:

- `[x]` 완료
- `[~]` 일부 완료 / 외부에서만 확인됨 / 다음 단계 연결 필요
- `[ ]` 미구현

## MVP 운영 원칙

- 현재 목표는 전체 목표 상태를 한 번에 닫는 것이 아니라, `로컬 FastAPI 서버에서 consult/executive 흐름이 실제로 도는 최소 기능 제품`을 먼저 증명하는 것입니다.
- 캐시 통합, CloudWatch, Secrets Manager, Streamlit UI 같은 운영·최적화 항목은 `MVP 이후`로 미룹니다.
- 우선순위는 `TDD -> 최소 구현 -> 로컬 서버 실행 확인 -> 그 다음 최적화`입니다.

---

## 1. 전체 아키텍처 진행률 요약

### 공통 도메인 / 데이터 구조

- `[x]` PubMed 연동용 공통 모델 정의
- `[x]` `PubMedQueryInput`, `PubMedArticleRaw`, `EvidenceItem`, `EvidencePacket` 구현
- `[x]` 전체 시스템 공통 모델 확장
  - `CompoundContext`
  - `PredictionBundle`
  - `AgentFinding`
  - `DecisionDraft`

### PubMed Evidence 파이프라인

- `[x]` Query Builder 구현
- `[x]` PubMed `ESearch` 호출 구현
- `[x]` PubMed `EFetch` 호출 구현
- `[x]` XML 파싱 구현
- `[x]` `EvidenceItem` 정규화 구현
- `[x]` 규칙 기반 점수화 구현
- `[x]` fallback query 검색 구현
- `[x]` `collect_pubmed_evidence(...)` 구현
- `[x]` 임의 query 목록으로 evidence 수집하는 `collect_pubmed_evidence_from_queries(...)` 구현

### PubMed Query Agent

- `[x]` LangChain/LangGraph 기반 Query Planner Agent 구현
- `[x]` Bedrock LLM structured output 기반 query plan 생성 구현
- `[x]` query validation / sanitize 구현
- `[x]` dry-run PubMed search 검증 구현
- `[x]` revision loop 구현
- `[x]` finalize 단계 구현
- `[x]` fallback query 선택 로직 구현
- `[x]` 실제 Bedrock 자격 증명과 연결된 실환경 호출 검증
- `[x]` consult/executive evidence flow에 planner 연결
  - 2026-03-24 `EvidenceCoordinator`가 PubMed 수집 전에 planner를 optional로 호출하고, `selected_query + candidate_queries`를 `collect_pubmed_evidence_from_queries(...)`에 전달하도록 연결
  - planner 미설정/실패 시 규칙 기반 `collect_pubmed_evidence(...)`로 명시적 fallback
  - consult router의 `safety_pk` / `clinical_regulatory` / `multi_expert` 타입을 PubMed/planner용 `safety` / `pk` / `regulatory` / `complex`로 deterministic normalize

### 테스트

- `[x]` 공통 도메인 모델 단위 테스트 작성
- `[x]` PubMed client 단위 테스트 작성
- `[x]` TxGemma client 단위 테스트 작성
- `[x]` TxGemma property registry 단위 테스트 작성
- `[x]` PubMed query agent 단위 테스트 작성
- `[x]` 로컬 `.venv`에서 테스트 통과
- `[ ]` 실제 PubMed API를 대상으로 한 통합 테스트 (MVP 이후)
- `[ ]` Bedrock 실호출 통합 테스트 (MVP 이후)

---

## 2. 아키텍처 컴포넌트별 상태

### A. 입력 / 전처리 계층

- `[x]` SMILES 검증
- `[x]` canonical SMILES 생성
- `[x]` RDKit 기반 분자 SVG 생성
- `[x]` `CompoundContext` 빌더
  - `app/domain/compound.py`의 `CompoundPreprocessor.build_context(...)` 구현
  - consult / executive / stub workflow에서 전처리 공통 적용
  - invalid SMILES 입력 시 API `400 invalid_smiles`로 즉시 반환

### B. 예측 계층

- `[x]` TxGemma SageMaker endpoint 외부에서 배포 및 호출 성공
  - `build_txgemma_client(...)` env wiring: `TXGEMMA_SAGEMAKER_ENDPOINT_NAME` / `TXGEMMA_ENDPOINT_NAME`, `TXGEMMA_AWS_REGION -> AWS_REGION -> AWS_DEFAULT_REGION`
  - 2026-03-24 registry-based live invoke 확인: 초기 `9/10` signals 생성, `solubility` 1개 missing
  - 2026-03-24 solubility 미세조정 후 live invoke 재확인: `10/10` signals 생성, `missing_signals=[]`
  - 2026-03-24 TDC ADMET 22 확장 후 live invoke 재검증: `17/22` signals 생성, missing=`Caco2`, `Lipophilicity`, `Solubility`, `PPBR`, `LD50`
  - 2026-03-24 missing 5-signal tuning 후 live invoke 재검증: `22/22` signals 생성, `missing_signals=[]`
- `[x]` 저장소 내 `TxGemmaClient` 구현
- `[x]` `PredictionBundle` 파서 구현
  - `generated_text/text` 기본 응답 외에 SageMaker-style `predictions` wrapper와 직접 `{"answer": ...}` structured payload도 허용하도록 parser hardening
  - `tests/test_txgemma_client.py`에 wrapper/direct structured payload 회귀 테스트 추가
- `[ ]` 예측 캐시 구현 (MVP 이후)
- `[x]` TDC ADMET 22 속성 레지스트리 구현
  - 2026-03-24 기존 추상 10-signal registry를 exact task-level 22-signal set으로 교체
  - Walter/House regression test를 추가해 새 signal 이름으로도 chemistry/safety fallback 판단이 유지됨을 검증
  - 2026-03-24 live recovery tuning: `Caco2`, `Lipophilicity`, `Solubility`, `PPBR`, `LD50`를 classification/few-shot/targeted grammar 조합으로 전환해 full `22/22` 회복

### C. Evidence Collection 계층

- `[x]` PubMed client 구현
- `[x]` PubChem client 구현
- `[x]` ChEMBL client 구현
- `[x]` ClinicalTrials.gov client 구현
- `[x]` openFDA client 구현
- `[x]` 전체 evidence coordinator 구현
- `[x]` source health / partial failure 통합 처리
  - 2026-03-24 PubChem/ChEMBL/ClinicalTrials/openFDA hardening: `URLError`/`JSONDecodeError`를 source-specific degraded packet으로 분류하고 `diagnostics.error_type`을 추가했으며, score 갱신을 `dataclasses.replace(...)`로 통일
  - 2026-03-24 PubChem/openFDA hardening: `HTTPError 404`는 upstream failure가 아니라 no-hit로 분류하고 다음 fallback query를 계속 시도하도록 수정
  - 2026-03-24 PubChem query normalization hardening: `Axitinib analog` 같은 compound name에서 trailing descriptor를 걷어낸 variant를 fallback candidate로 추가해 strict name lookup hit율을 보강
  - 2026-03-24 PubChem SMILES passthrough hardening: consult/executive가 canonical SMILES를 evidence layer로 전달하고, name query가 모두 miss면 PubChem identity search를 `smiles_identity:<canonical_smiles>`로 재시도하도록 연결

### D. Query Planning / Routing 계층

- `[x]` PubMed용 query planning agent 구현
  - 2026-03-24 query compilation hardening: base term quoting에서 literal `"`/`\`를 escape하도록 수정하고, planner/router/analyzer runnable 타입 힌트를 `Protocol`로 좁혀 broad `Any`를 제거
- `[x]` PubMed planner runtime wiring 구현
  - 2026-03-24 `BEDROCK_PUBMED_QUERY_MODEL_ID` settings/dependency wiring 추가 및 coordinator-level regression test 추가
- `[x]` 전체 consult용 Router 구현
- `[x]` `question_type -> selected_agents` 라우팅 구현
- `[x]` Router confidence/fallback 정책 구현
  - 2026-03-24 Korean mixed-review keyword hardening: fallback router가 `물성`, `안전 프로파일`, `개발 전략` 조합 질문을 `multi_expert`로 올리도록 safety/clinical keyword coverage 보강

### E. 전문가 Agent 계층

- `[x]` Walter agent 구현
- `[x]` House agent 구현
- `[x]` Harvey agent 구현
- `[x]` Agent Registry 구현
- `[x]` Parallel Executor 구현

### F. 결과 합성 계층

- `[x]` consult용 `Answer Composer` 구현
- `[x]` executive용 `CEO Synthesizer` 구현
- `[x]` citation completeness 검증 구현
- `[x]` `review_required=true` 응답 규칙 구현

### G. API / 애플리케이션 계층

- `[x]` FastAPI 앱 생성
- `[x]` `POST /api/reports/consult` 구현
  - `app/api/main.py`, `app/workflows/consult.py`로 consult API/app wiring 추가
  - `tests/test_consult_api.py`로 TDD happy path + request validation 검증 추가
- `[x]` `POST /api/reports/executive` 구현
  - `app/workflows/executive.py`로 3-expert 병렬 실행 + CEO synthesis executive workflow 추가
  - `app/api/main.py`에 `POST /api/reports/executive` wiring 및 typed response 추가
  - `tests/test_executive_api.py`, `tests/test_executive_workflow.py`로 TDD happy path / validation / typed error / orchestration 검증 추가
  - 2026-03-24 live clinical/openFDA evidence의 list-valued metadata를 허용하도록 `EvidenceItem.metadata` / `ExecutiveResponse` contract 확장
- `[x]` 최소 설정 / 환경변수 로더 구현
  - `app/api/settings.py`에 `load_settings(...)` / `AppSettings.from_env(...)` 추가
  - runtime host/port, stub workflow mode, E-utilities identity, TxGemma/Bedrock env resolution 추가
  - 2026-03-24 `.env` / `.env.local` local env file loading 추가, alias-aware override merge 적용
  - 2026-03-24 local `.env.local`에 `BEDROCK_AWS_REGION` 및 Router/Walter/House/Harvey/PubMed planner model ID를 `global.anthropic.claude-sonnet-4-6` inference profile로 연결
  - 2026-03-24 plain `uvicorn --factory` 재시작 경로에서 `.env.local` 기반 live TxGemma consult 재검증: `predictions.source=txgemma`, `10/10` signals
- `[x]` 최소 에러 처리 / 응답 스키마 구현
  - `app/api/schemas.py`로 request/response/error schema 분리
  - `app/api/errors.py`로 request validation / app service / unexpected error handler 추가
  - 2026-03-24 API hardening: request schema에 `max_length` 제한을 추가해 oversized `smiles` / `target` / `question` / `compound_name` 입력을 `422 request_validation_error`로 차단하고, unexpected error 로그 path는 query string을 제거한 값만 남기도록 정리
- `[x]` consult / executive streamed trace endpoint 구현
  - 2026-03-24 `/api/reports/consult/stream`, `/api/reports/executive/stream` NDJSON stream 추가
  - workflow trace event를 background thread + queue로 중계하고, 결과/에러를 같은 stream 프로토콜로 전달
- `[x]` 로컬 FastAPI 서버 실행 검증
  - `app.api.main:create_runtime_app` + `uvicorn --factory` 경로 검증
  - `tests/test_api_runtime.py`로 subprocess 기반 local server startup 검증 추가
- `[x]` 로컬 consult API smoke test
  - `tests/test_api_runtime.py`에서 consult `200` 응답 확인
- `[x]` 로컬 executive API smoke test
  - `tests/test_api_runtime.py`에서 executive `200` 응답 확인
  - 2026-03-24 live runtime 재검증: `TxGemma(ap-southeast-2)` + `Bedrock(ap-northeast-2 inference profile)`로 consult/executive 모두 `200`

### H. UI 계층 (MVP 이후)

- `[x]` Streamlit UI 구현
  - `app/ui/main.py` Streamlit shell + sidebar runtime controls + API 연결 구현
  - `app/ui/client.py`로 FastAPI request/error handling 분리
  - `app/ui/presenters.py`로 consult/executive payload presentation model 분리
  - `app/ui/theme.py`에 sidebar-specific contrast override 추가로 runtime/workspace/runbook 텍스트 가독성 보정
  - live executive latency 대응을 위해 UI API timeout 상향 및 stale cached `UiApiError` 흡수 로직 추가
  - 2026-03-24 UI hardening: executive molecule SVG는 allowed tag/attribute 기반 sanitizer를 거친 뒤에만 `components.html(...)`로 렌더링하도록 변경
- `[x]` 운영 trace / status UI 구현
  - 2026-03-24 Streamlit `st.status` 기반 실시간 진행 상태 표시 추가
  - consult/executive 실행 중 `Selected agents`, `PubMed planner selected query`, `PubMed dry run hits`, source degraded 메시지를 trace panel로 누적 표시
- `[x]` consult 화면 구현
  - consult 입력 form, selected agents / signals / findings / citations 렌더링 추가
  - 2026-03-24 agent finding confidence를 expander 헤더 툴팁/대비에 의존하지 않도록 본문 상단에 항상 visible label로 렌더링
- `[x]` executive 화면 구현
  - executive 입력 form, decision summary / molecule SVG / evidence source / findings 렌더링 추가

### I. 운영 / 관찰 / 캐시 (MVP 이후)

- `[x]` PubMed 로컬 메모리 TTL 캐시 구현
- `[ ]` 전체 evidence 캐시 계층 통합
- `[ ]` prediction 캐시 계층 통합
- `[ ]` CloudWatch/로깅 정리
- `[ ]` Secrets Manager 연동

---

## 3. 현재까지 실제 완료된 파일

- `[x]` `app/domain/models.py`
- `[x]` `app/domain/compound.py`
- `[x]` `app/domain/prediction_registry.py`
- `[x]` `app/clients/pubmed.py`
- `[x]` `app/clients/pubchem.py`
- `[x]` `app/clients/chembl.py`
- `[x]` `app/clients/clinicaltrials.py`
- `[x]` `app/clients/openfda.py`
- `[x]` `app/clients/evidence_coordinator.py`
- `[x]` `app/clients/txgemma.py`
- `[x]` `app/agents/pubmed_query_agent.py`
- `[x]` `app/agents/router_agent.py`
- `[x]` `app/agents/registry.py`
- `[x]` `app/agents/parallel_executor.py`
- `[x]` `app/agents/house_agent.py`
- `[x]` `app/agents/walter_agent.py`
- `[x]` `app/agents/harvey_agent.py`
- `[x]` `app/agents/citation_validator.py`
- `[x]` `app/agents/answer_composer.py`
- `[x]` `app/agents/ceo_synthesizer.py`
- `[x]` `app/agents/review_policy.py`
- `[x]` `app/api/main.py`
- `[x]` `app/api/dependencies.py`
- `[x]` `app/api/errors.py`
- `[x]` `app/api/schemas.py`
- `[x]` `app/api/settings.py`
- `[x]` `app/api/stubs.py`
- `[x]` `app/ui/main.py`
- `[x]` `app/ui/client.py`
- `[x]` `app/ui/presenters.py`
- `[x]` `app/ui/AGENTS.md`
- `[x]` `app/workflows/consult.py`
- `[x]` `app/workflows/executive.py`
- `[x]` `app/workflows/tracing.py`
- `[x]` `tests/test_pubmed_client.py`
- `[x]` `tests/test_pubchem_client.py`
- `[x]` `tests/test_chembl_client.py`
- `[x]` `tests/test_clinicaltrials_client.py`
- `[x]` `tests/test_openfda_client.py`
- `[x]` `tests/test_evidence_coordinator.py`
- `[x]` `tests/test_txgemma_client.py`
- `[x]` `tests/test_txgemma_registry.py`
- `[x]` `tests/test_pubmed_query_agent.py`
- `[x]` `tests/test_router_agent.py`
- `[x]` `tests/test_agent_registry.py`
- `[x]` `tests/test_parallel_executor.py`
- `[x]` `tests/test_house_agent.py`
- `[x]` `tests/test_walter_agent.py`
- `[x]` `tests/test_harvey_agent.py`
- `[x]` `tests/test_consult_api.py`
- `[x]` `tests/test_executive_api.py`
- `[x]` `tests/test_api_runtime.py`
- `[x]` `tests/test_api_settings.py`
- `[x]` `tests/test_consult_workflow.py`
- `[x]` `tests/test_executive_workflow.py`
- `[x]` `tests/test_answer_composer.py`
- `[x]` `tests/test_ceo_synthesizer.py`
- `[x]` `tests/test_citation_validator.py`
- `[x]` `tests/test_review_policy.py`
- `[x]` `tests/test_domain_models.py`
- `[x]` `tests/test_compound_preprocessor.py`
- `[x]` `tests/test_ui_client.py`
- `[x]` `tests/test_ui_presenters.py`
- `[x]` `tests/test_ui_runtime.py`
- `[x]` `pyproject.toml`

---

## 4. 다음 우선순위

### 바로 다음에 구현할 것

- `[ ]` 예측 캐시 구현 (MVP 이후)
- `[ ]` 전체 evidence 캐시 계층 통합 (MVP 이후)

### 그 다음

- `[ ]` prediction 캐시 계층 통합 (MVP 이후)
- `[ ]` CloudWatch/로깅 정리 (MVP 이후)

### MVP 이후

- `[ ]` 예측 캐시 구현
- `[ ]` 전체 evidence 캐시 계층 통합
- `[ ]` prediction 캐시 계층 통합
- `[ ]` CloudWatch/로깅 정리
- `[ ]` Secrets Manager 연동
- `[ ]` Streamlit UI 구현
- `[ ]` 실제 PubMed API 통합 테스트
- `[ ]` Bedrock 실호출 통합 테스트

---

## 5. 현재 진행도 해석

### 저장소 기준

- `PubMed 축`은 기초 구현 완료
- `Evidence Collection 축`은 멀티소스 client + coordinator까지 구현 완료
- `Query planning agent 축`은 기초 구현 완료
- `전체 신약개발 멀티에이전트 시스템`은 아직 초기 단계
- `MVP 기준`으로는 consult/executive API, local FastAPI runtime, consult/executive smoke 검증까지 완료
- 입력 전처리 계층(`SMILES 검증 / canonicalization / SVG / CompoundContext`)의 최소 구현은 완료되었습니다.
- Streamlit UI 계층(consult / executive 화면 포함)도 구현되었습니다.
- 현재 로컬 데모 관점의 다음 큰 갭은 캐시/운영 항목이며, 여전히 MVP 이후 범주입니다.

### 실질적 위치

- 현재는 `PubMed + 멀티소스 Evidence Collection + Query Agent runtime 연결 + TxGemma prediction adapter`까지 기초 구현을 마친 상태
- 전체 목표 상태 기준으로 보면 아직 초기 단계
- 현재 로컬 데모 관점의 주요 잔여 갭은 캐시/운영 항목과 선택적 실환경 통합 테스트입니다.
- 캐시, UI, 운영 연동은 MVP 이후로 뒤로 미룸
