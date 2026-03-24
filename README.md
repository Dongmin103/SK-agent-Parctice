# SK Agent Practice

`consult`와 `executive` 워크플로를 로컬에서 검증하기 위한 Agentic AI 신약개발 PoC 저장소입니다.

이 저장소는 다음 흐름을 하나의 런타임으로 묶습니다.

- TxGemma 기반 ADMET 예측
- PubMed, PubChem, ChEMBL, ClinicalTrials.gov, openFDA 근거 수집
- Bedrock 기반 라우터와 전문가 에이전트
- FastAPI 기반 consult / executive 오케스트레이션
- Streamlit 기반 로컬 워크벤치 UI

## 현재 구현 범위

현재 로컬에서 검증 가능한 최소 수직 슬라이스는 아래와 같습니다.

- SMILES 전처리 및 canonical SMILES 생성
- TDC ADMET 22 신호 예측 계약
- PubMed query planner agent와 fallback 검색 흐름
- Consult router와 Walter / House / Harvey 전문가 에이전트
- Executive synthesis 흐름
- 실시간 runtime trace / status UI

## 주요 문서

- `architecture_progress_checklist_ko.md`
  현재 구현 상태와 작업 진행 체크리스트
- `architecture_overview_ko.md`
  시스템 경계와 데이터 흐름 개요
- `agent_structure_overview_ko.md`
  에이전트 역할과 호출 관계
- `skbiopharm_agentic_ai_poc_implementation_plan_ko.md`
  목표 구현 계획

## 디렉터리 구조

```text
app/
  agents/      Bedrock 에이전트, 라우팅, 합성
  api/         FastAPI 앱, settings, dependency wiring
  clients/     외부 evidence / model client
  domain/      공통 모델과 prediction registry
  ui/          Streamlit 로컬 워크벤치
  workflows/   consult / executive 오케스트레이션
tests/         오프라인 회귀 테스트
```

## 로컬 설치

```bash
python3 -m venv .venv
./.venv/bin/pip install -e .
```

## 환경변수

이 프로젝트는 `.env`와 `.env.local`을 모두 읽습니다. 로컬 개발 환경에서는 `.env.local` 사용을 권장합니다.

예시:

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

# 선택
NCBI_API_KEY=...
```

주의:

- 로컬 비밀값과 런타임 설정은 git에 포함하지 않습니다.
- Bedrock Sonnet 4.6은 raw foundation model ID가 아니라 inference profile 예시인 `global.anthropic.claude-sonnet-4-6`로 연결하는 편이 안전합니다.

## 실행

FastAPI:

```bash
./.venv/bin/uvicorn app.api.main:create_runtime_app --factory --host 127.0.0.1 --port 8000
```

Streamlit:

```bash
./.venv/bin/streamlit run app/ui/main.py
```

## 테스트

```bash
./.venv/bin/pytest -q
```
