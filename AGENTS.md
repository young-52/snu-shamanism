# 🔮 Project SHAMANISM: Agent Team Roles

## 1. Geoman-ji: The Saju Logic Architect
- **Primary Goal**: `korean-lunar-calendar` 라이브러리를 활용한 정교한 사주 분석 엔진 구축.
- **Responsibilities**:
    - 사용자의 생년월일시(Solar/Lunar)를 천간·지지 및 오행으로 변환.
    - 8자(또는 6자) 기반의 오행 분포도 계산 로직 구현.
    - 부족한 기운(용신)을 도출하여 추천 시스템에 전달.

## 2. The Persona: LLM Reasoning & RAG Lead
- **Primary Goal**: `huggingface_hub` Inference API를 활용하여 친근하고 신뢰감 있는 대화 인터페이스 구현.
- **Responsibilities**:
    - `locations_explained.json` 및 `cafes2.json` 데이터를 기반으로 한 RAG(Retrieval-Augmented Generation) 로직 설계.
    - 오행 상생상극(목생화, 토극수 등)에 기반한 추천 사유 생성 프롬프트 엔지니어링.
    - 현대적이고 일상적인 높임체(~합니다, ~드립니다) 톤의 일관성 유지.

## 3. UI Sorcerer: Gradio Layout & UX Designer
- **Primary Goal**: Gradio 6.x 환경에서의 최적화된 사용자 인터페이스 구축 및 커스텀 테마 적용.
- **Responsibilities**:
    - 사이드바(입력부)와 메인 대화창(출력부) 간의 매끄러운 상태 관리(`gr.State`).
    - 모바일 및 웹 반응형 레이아웃 최종 검수.