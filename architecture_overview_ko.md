# Agentic AI 신약개발 PoC 전체 아키텍처 시각화

## 시스템 개요

```mermaid
flowchart LR
  U["연구자"] --> UI["Streamlit UI"]
  UI --> API["FastAPI API"]

  subgraph RUNTIME["서비스 런타임"]
    API --> PRE["Compound Preprocessor<br/>RDKit 검증 / canonicalization / SVG"]
    PRE --> PRED["Prediction Adapter<br/>TxGemma on SageMaker"]
    PRE --> EV["Evidence Collection Coordinator"]
    PRED --> PB["PredictionBundle"]
    EV --> EP["EvidencePacket"]
    PB --> RT["Bedrock Router"]
    PB --> W["Walter"]
    PB --> H["House"]
    PB --> V["Harvey"]
    EP --> W
    EP --> H
    EP --> V
    RT --> PX["Parallel Executor"]
    PX --> W
    PX --> H
    PX --> V
    W --> SYN["Answer Composer / CEO Synthesizer"]
    H --> SYN
    V --> SYN
    SYN --> RESP["API Response<br/>review_required=true"]
  end

  RESP --> UI

  subgraph SOURCES["공개 데이터 소스"]
    PC["PubChem"]
    CH["ChEMBL"]
    PM["PubMed"]
    CT["ClinicalTrials.gov"]
    FD["openFDA"]
    SM["SageMaker TxGemma Endpoint"]
  end

  EV --> PC
  EV --> CH
  EV --> PM
  EV --> CT
  EV --> FD
  PRED --> SM

  subgraph CACHE["캐시"]
    C1["Prediction Cache<br/>TTL 7d"]
    C2["Evidence Cache<br/>TTL 24h"]
  end

  PRED <--> C1
  EV <--> C2
```

## 스마트 컨설팅 흐름

```mermaid
flowchart TD
  Q["사용자 질문"] --> RT["Bedrock Router"]
  RT --> DEC{"질문 분류 성공?"}

  DEC -->|예| SEL["필요 전문가 선택"]
  DEC -->|아니오| ALL["Walter + House + Harvey 모두 호출"]

  SEL --> A1{"질문 유형"}
  A1 -->|구조 / SAR| W["Walter"]
  A1 -->|독성 / PK / DDI| H["House"]
  A1 -->|임상 / 규제 / 승인| V["Harvey"]
  A1 -->|복합 질문| M["복수 전문가 병렬 호출"]

  ALL --> SYN["Answer Composer"]
  W --> SYN
  H --> SYN
  V --> SYN
  M --> SYN

  SYN --> OUT["consulting_answer<br/>selected_agents<br/>routing_reason<br/>citations<br/>review_required=true"]
```

## 종합 분석 흐름

```mermaid
sequenceDiagram
  participant User as 연구자
  participant UI as Streamlit UI
  participant API as FastAPI
  participant Pre as Preprocessor
  participant Pred as Prediction Adapter
  participant Ev as Evidence Coordinator
  participant W as Walter
  participant H as House
  participant V as Harvey
  participant CEO as CEO Synthesizer

  User->>UI: SMILES + Target 입력
  UI->>API: POST /api/reports/executive
  API->>Pre: SMILES 검증 / canonicalization / SVG 생성

  alt 유효하지 않은 SMILES
    Pre-->>API: validation error
    API-->>UI: 400 error
  else 유효한 SMILES
    API->>Pred: TxGemma 예측 요청
    API->>Ev: 공개 데이터 병렬 수집
    Pred-->>API: PredictionBundle
    Ev-->>API: EvidencePacket

    API->>W: chemistry packet + predictions
    API->>H: safety/literature packet + predictions
    API->>V: clinical/regulatory packet + predictions

    W-->>API: AgentFinding JSON
    H-->>API: AgentFinding JSON
    V-->>API: AgentFinding JSON

    API->>CEO: findings + citations
    CEO-->>API: executive_decision_draft
    API-->>UI: executive report + citations + missing_sources + review_required=true
  end
```

## 모듈 경계

```mermaid
flowchart LR
  subgraph REPO["Python Repository"]
    API["app/api"]
    WF["app/workflows"]
    AG["app/agents"]
    CL["app/clients"]
    DM["app/domain"]
    UI["app/ui"]
  end

  API --> WF
  WF --> AG
  WF --> CL
  WF --> DM
  UI --> API
  AG --> DM
  CL --> DM
```

## 문서 사용 원칙

- 이 문서는 목표 상태 아키텍처를 설명합니다.
- 실제 진행 상태는 `architecture_progress_checklist_ko.md`에서 관리합니다.
- 구조 변경 시 이 문서와 `AGENTS.md`를 함께 업데이트합니다.
