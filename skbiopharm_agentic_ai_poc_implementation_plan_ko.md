# SK바이오팜형 Agentic AI 신약개발 PoC 구현 계획

## Summary
- 현재 작업공간은 빈 상태이므로, 기존 코드 증분이 아니라 신규 Python 3.11 프로젝트로 시작합니다.
- 목표 산출물은 `로컬 개발 + AWS 연동` 방식의 PoC 웹앱이며, `Streamlit + FastAPI + Strands Agents + Bedrock + SageMaker(TxGemma)` 조합으로 구현합니다.
- v1에는 두 사용자 흐름을 모두 포함합니다: `3-agent 종합 분석 리포트`와 `자연어 기반 스마트 컨설팅`.
- TxGemma 예측은 실제 SageMaker 엔드포인트를 사용하되, v1에서는 `핵심 10개 속성`만 활성화하고 구조는 `22개까지 확장 가능`하게 설계합니다.

## Public APIs / Interfaces
- `POST /api/reports/executive`
  - 입력: `smiles`, `target`, `compound_name?`
  - 출력: `canonical_smiles`, `molecule_svg`, `predictions`, `evidence_bundle`, `agent_findings`, `executive_decision`, `citations`
- `POST /api/reports/consult`
  - 입력: `smiles`, `target`, `question`, `compound_name?`
  - 출력: `selected_agents`, `routing_reason`, `predictions`, `agent_findings`, `consulting_answer`, `citations`
- 내부 공통 스키마는 Pydantic으로 고정합니다.
  - `PredictionBundle`: 10개 핵심 속성 + 단위 + 신뢰도 + 생성시각
  - `EvidenceItem`: `source`, `title`, `snippet`, `url`, `score`
  - `AgentFinding`: `agent_id`, `summary`, `risks`, `recommendations`, `confidence`, `citations`
  - `ExecutiveDecision`: `go | conditional_go | no_go`, `rationale`, `next_steps`

## Implementation Changes
- 프로젝트는 단일 Python 저장소로 구성합니다.
  - `app/api`: FastAPI 라우트와 요청/응답 스키마
  - `app/ui`: Streamlit 화면
  - `app/workflows`: 종합 분석, 스마트 컨설팅 오케스트레이션
  - `app/agents`: Walter, House, Harvey, CEO/router 프롬프트와 실행기
  - `app/clients`: SageMaker, Bedrock, PubChem, ChEMBL, PubMed, ClinicalTrials.gov, openFDA 클라이언트
  - `app/domain`: SMILES 검증, 속성 레지스트리, 정규화 로직
- 기본 의존성은 `uv`, `fastapi`, `streamlit`, `strands-agents`, `boto3`, `sagemaker`, `httpx`, `pydantic v2`, `rdkit`, `tenacity`, `diskcache`, `pytest`, `respx`로 고정합니다.
- 입력 처리 단계에서 RDKit으로 SMILES를 검증·canonicalize하고 분자 SVG를 생성합니다. 유효하지 않은 SMILES는 에이전트 호출 전에 즉시 차단합니다.
- TxGemma 예측 계층은 SageMaker 엔드포인트 래퍼로 분리합니다. 캐시 키는 `canonical_smiles + target`로 고정하고, 동일 입력 재요청 시 예측과 증거 수집을 재사용합니다.
- 핵심 10개 속성은 `solubility`, `logP/logD`, `permeability`, `plasma protein binding`, `clearance`, `half-life`, `CYP inhibition`, `hERG`, `hepatotoxicity`, `BBB`로 시작합니다. 각 속성은 `prompt template + parser + unit + display metadata`를 갖는 레지스트리로 관리합니다.
- 증거 수집은 병렬 비동기로 수행합니다.
  - Walter: PubChem + ChEMBL 중심의 유사 화합물, SAR, 구조/물성 근거
  - House: PubMed + TxGemma 예측 기반 PK/PD, DDI, 독성 근거
  - Harvey: ClinicalTrials.gov + openFDA 기반 임상/규제/승인 전략 근거
- `종합 분석` 흐름에서는 세 전문가를 항상 병렬 호출하고, CEO 집계 에이전트가 최종 `go/conditional_go/no_go`와 우선 액션을 생성합니다.
- `스마트 컨설팅` 흐름에서는 먼저 Bedrock 라우터가 구조화된 JSON으로 필요한 전문가를 선택한 뒤, 선택된 에이전트만 병렬 실행합니다. 라우터가 확신이 낮거나 파싱 실패 시 기본값은 `3명 모두 호출`입니다.
- 모든 전문가 에이전트는 자유형 Markdown이 아니라 `strict JSON`으로 반환하게 하고, 최종 한국어 문장은 API 레이어에서 렌더링합니다. 이렇게 해야 테스트 가능성과 UI 안정성이 유지됩니다.
- 외부 API 장애는 부분 허용으로 처리합니다. 개별 소스 실패 시 전체 요청은 유지하되, 리포트에 `미수집 소스`를 명시합니다.
- v1에서는 인증, 사용자별 보고서 저장, 벡터DB, 장기 메모리, PDF 생성, 특허/내부 ELN/LIMS 연동은 제외합니다. 세션 상태와 로컬 디스크 캐시만 사용합니다.

## Test Plan
- 단위 테스트: SMILES 검증, 속성 레지스트리, 응답 파서, 라우터 출력 검증, evidence normalizer, citation merge.
- 계약 테스트: SageMaker predictor mock, Bedrock agent mock, PubChem/ChEMBL/PubMed/ClinicalTrials.gov/openFDA 응답 스키마 검증.
- 통합 테스트: `POST /api/reports/executive`, `POST /api/reports/consult`의 정상 경로와 실패 경로를 모두 검증합니다.
- E2E 시나리오:
  - 정상 SMILES로 종합 분석 리포트 생성
  - 동일 SMILES 재실행 시 캐시 hit
  - 질문별 에이전트 동적 선택
  - PubMed 또는 ClinicalTrials.gov 한 소스 장애 시 부분 결과 반환
  - 잘못된 SMILES 입력 시 즉시 오류 반환
- 수용 기준:
  - 모든 최종 주장에는 최소 1개 이상의 출처 URL이 연결됨
  - 스마트 컨설팅은 선택된 에이전트 목록을 응답에 명시함
  - 최종 리포트와 컨설팅 답변은 한국어로 출력됨
  - 블로그의 2025년 하드코딩 모델 ID에 의존하지 않고 환경설정으로 교체 가능함

## Assumptions / Defaults
- AWS 계정에는 Bedrock 모델 접근 권한과 SageMaker 실행 역할이 이미 준비되어 있다고 가정합니다.
- TxGemma 사용을 위해 Hugging Face 접근 권한과 토큰이 준비되어 있다고 가정합니다.
- 현재 날짜는 `2026-03-23`이므로, 블로그의 `2025-11-18` 시점 예제와 달리 Bedrock 모델 버전은 최신 지원 버전으로 환경 변수에서 주입합니다. `anthropic.claude-sonnet-4-20250514-v1:0`는 코드에 직접 박지 않습니다.
- 기본 Bedrock 지역은 사용자 위치를 고려해 `ap-northeast-2`를 우선 검토하되, 선택한 Claude Sonnet 버전 가용성이 없으면 `us-west-2`로 전환합니다.
- v1은 연구용 의사결정 보조 도구이며, 규제 제출용 시스템이나 의료 판단 자동화 시스템으로 취급하지 않습니다.

## References
- [Strands Agents Quickstart](https://strandsagents.com/docs/user-guide/quickstart/)
- [Amazon Bedrock Supported Claude Models](https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-supported-models.html)
- [Amazon Bedrock Model Lifecycle](https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html)
- [PubChem PUG-REST Tutorial](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest-tutorial)
- [ChEMBL REST API Docs](https://www.ebi.ac.uk/chembl/api/data/docs)
- [NCBI Entrez E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/)
- [ClinicalTrials.gov API](https://clinicaltrials.gov/data-api/about-api)
- [openFDA Drug APIs](https://open.fda.gov/apis/drug/)
