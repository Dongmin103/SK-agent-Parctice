# Agent 구조 시각화

## 전체 Agent 구조

```mermaid
flowchart TD
  IN["사용자 입력<br/>SMILES / Target / Question"] --> PRE["공통 전처리 계층"]
  PRE --> TX["TxGemma Prediction Layer"]
  PRE --> EV["Evidence Collection Layer"]

  EV --> PC["PubChem / ChEMBL"]
  EV --> PM["PubMed"]
  EV --> CT["ClinicalTrials.gov / openFDA"]

  TX --> RT["Router"]
  EV --> RT

  RT --> DEC{"selected_agents"}

  DEC -->|walter| W["Walter"]
  DEC -->|house| H["House"]
  DEC -->|harvey| V["Harvey"]
  DEC -->|복수 선택| PX["Parallel Executor"]

  PX --> W
  PX --> H
  PX --> V

  W --> AC["Answer Composer"]
  H --> AC
  V --> AC

  W --> CEO["CEO Synthesizer"]
  H --> CEO
  V --> CEO

  AC --> COUT["Consult Output<br/>consulting_answer<br/>citations<br/>review_required=true"]
  CEO --> EOUT["Executive Output<br/>go / conditional_go / no_go draft"]
```

## Consult 흐름

```mermaid
flowchart LR
  Q["질문"] --> R["Router"]
  R --> S["selected_agents"]

  S -->|Walter| W["Walter"]
  S -->|House| H["House"]
  S -->|Harvey| V["Harvey"]
  S -->|2개 이상| P["Parallel Executor"]

  P --> W
  P --> H
  P --> V

  W --> A["Answer Composer"]
  H --> A
  V --> A

  A --> O["consulting_answer<br/>selected_agents<br/>routing_reason<br/>citations"]
```

## Executive 흐름

```mermaid
flowchart LR
  X["SMILES / Target"] --> C["공통 컨텍스트 생성"]
  C --> W["Walter"]
  C --> H["House"]
  C --> V["Harvey"]

  W --> CEO["CEO Synthesizer"]
  H --> CEO
  V --> CEO

  CEO --> O["go / conditional_go / no_go draft<br/>rationale<br/>next_steps"]
```

## Agent별 데이터 의존성

```mermaid
flowchart TD
  TX["TxGemma Prediction Summary"] --> W["Walter"]
  TX --> H["House"]
  TX --> V["Harvey"]

  PUB["PubChem / ChEMBL"] --> W
  PMD["PubMed"] --> H
  CTR["ClinicalTrials.gov / openFDA"] --> V

  W --> WF["구조 리스크 / SAR / 구조 개선"]
  H --> HF["독성 / PK/PD / DDI / 모니터링"]
  V --> VF["승인 허들 / 임상 전략 / 개발 우선순위"]
```

## 단순 역할 관계

```mermaid
flowchart TD
  R["Router"] --> W["Walter"]
  R --> H["House"]
  R --> V["Harvey"]

  W --> AC["Answer Composer"]
  H --> AC
  V --> AC

  W --> CEO["CEO Synthesizer"]
  H --> CEO
  V --> CEO
```

## 사용 원칙

- 이 문서는 agent topology와 역할 관계를 설명합니다.
- 모듈 경계와 전체 런타임 데이터 흐름은 `architecture_overview_ko.md`에서 확인합니다.
- Agent를 추가, 제거, 병합하거나 호출 흐름을 바꾸면 이 문서와 `AGENTS.md`를 함께 업데이트합니다.
