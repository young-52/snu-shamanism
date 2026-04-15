"""
🔮 The Persona: LLM Reasoning & RAG Lead
RAG 기반 추천 시스템 — JSON 데이터에서 부족한 오행에 맞는 장소/음료를 필터링하고
시스템 프롬프트를 구성하여 응답을 생성한다.
"""

import json
import os
import random
from huggingface_hub import InferenceClient

# ── 데이터 로드 ──
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_BASE_DIR, "locations.json"), "r", encoding="utf-8") as f:
    LOCATIONS = json.load(f)

with open(os.path.join(_BASE_DIR, "cafes.json"), "r", encoding="utf-8") as f:
    CAFES = json.load(f)

# 오행 정규화 매핑 (한자와 한글 혼용 대응)
_ELEMENT_NORMALIZE = {
    "木": "木",
    "목": "木",
    "火": "火",
    "화": "火",
    "土": "土",
    "토": "土",
    "金": "金",
    "금": "金",
    "水": "水",
    "수": "水",
}


def _norm_element(raw: str) -> str:
    """오행 값을 정규화한다 (한글/한자 혼용 대응)."""
    return _ELEMENT_NORMALIZE.get(raw, raw)


def filter_locations(element: str, max_results: int = 3) -> list[dict]:
    """
    부족한 오행에 해당하는 장소를 필터링한다.
    다양한 구역이 나오도록 셔플 후 선택한다.
    """
    matched = [
        loc for loc in LOCATIONS if _norm_element(loc.get("element", "")) == element
    ]
    random.shuffle(matched)

    # 구역(zone) 다양성 확보
    seen_zones = set()
    diverse = []
    for loc in matched:
        zone = loc.get("zone", "")
        if zone not in seen_zones:
            diverse.append(loc)
            seen_zones.add(zone)
        if len(diverse) >= max_results:
            break

    # 다양성이 부족하면 나머지로 채움
    if len(diverse) < max_results:
        for loc in matched:
            if loc not in diverse:
                diverse.append(loc)
            if len(diverse) >= max_results:
                break

    return diverse


def filter_cafes(element: str, max_results: int = 3) -> list[dict]:
    """
    부족한 오행에 해당하는 카페 메뉴를 필터링한다.
    카페 다양성을 확보하면서 추천한다.
    """
    recommendations = []
    seen_cafes = set()

    # 모든 카페에서 해당 오행 메뉴 수집
    all_items = []
    for cafe in CAFES:
        cafe_name = cafe.get("cafe", "")
        zone = cafe.get("zone", "")
        for menu_item in cafe.get("menu", []):
            if _norm_element(menu_item.get("element", "")) == element:
                all_items.append(
                    {
                        "cafe": cafe_name,
                        "zone": zone,
                        "menu": menu_item["name"],
                        "hot": menu_item.get("hot"),
                        "ice": menu_item.get("ice"),
                        "element": element,
                        "reason": menu_item.get("reason", ""),
                    }
                )

    random.shuffle(all_items)

    # 카페 다양성 확보
    for item in all_items:
        if item["cafe"] not in seen_cafes:
            recommendations.append(item)
            seen_cafes.add(item["cafe"])
        if len(recommendations) >= max_results:
            break

    if len(recommendations) < max_results:
        for item in all_items:
            if item not in recommendations:
                recommendations.append(item)
            if len(recommendations) >= max_results:
                break

    return recommendations


def build_system_prompt(saju_result: dict) -> str:
    """
    시스템 프롬프트를 생성한다.
    사주 분석 결과와 추천 데이터를 포함한다.
    """
    weakest = saju_result["weakest"]
    weakest_name = saju_result["weakest_name"]

    # 추천 데이터 필터링
    rec_locations = filter_locations(weakest)
    rec_cafes = filter_cafes(weakest)

    # 장소 정보를 프롬프트에 삽입
    loc_info = ""
    for i, loc in enumerate(rec_locations, 1):
        loc_info += (
            f"  {i}. {loc.get('id', '?')}동 ({loc.get('college', '?')}) "
            f"— 구역: {loc.get('zone', '?')}, 오행: {loc.get('element', '?')}\n"
            f"     해설: {loc.get('reason', '')}\n"
        )

    # 카페 정보를 프롬프트에 삽입
    cafe_info = ""
    for i, cafe in enumerate(rec_cafes, 1):
        price_parts = []
        if cafe.get("hot"):
            price_parts.append(f"HOT {cafe['hot']}원")
        if cafe.get("ice"):
            price_parts.append(f"ICE {cafe['ice']}원")
        price_str = " / ".join(price_parts) if price_parts else "가격 미정"
        cafe_info += (
            f"  {i}. [{cafe.get('cafe', '?')}] {cafe.get('menu', '?')} "
            f"({price_str}) — 구역: {cafe.get('zone', '?')}\n"
            f"     사유: {cafe.get('reason', '')}\n"
        )

    system_prompt = f"""너는 서울대학교 캠퍼스의 장소와 카페에 대해 잘 알고 있는 사주 기반 추천 서비스 '샤:머니즘'의 AI 어시스턴트입니다.
사용자의 사주 오행을 분석하여, 부족한 기운을 보충할 수 있는 캠퍼스 내 장소와 음료를 추천하는 것이 당신의 역할입니다.

## 말투 규칙 (필수)
- 현대적이고 일상적인 높임체를 사용하세요: ~합니다, ~드립니다, ~입니다, ~하세요, ~좋겠습니다
- 무당 말투(~로구나, ~하도다, ~이로다 등)는 절대 사용하지 마세요
- 친근하면서도 신뢰감 있는 톤을 유지하세요
- 답변은 한국어로만 하세요

## 사주 분석 정보
- 간지: {saju_result["gap_ja_str"]}
- 가장 부족한 기운: {weakest_name} ({weakest})
- 가장 강한 기운: {saju_result["strongest_name"]} ({saju_result["strongest"]})
- 오행 분포: {saju_result["element_count"]}

## 오늘의 추천 장소 (부족한 {weakest_name} 기운을 보충하는 곳)
{loc_info}

## 오늘의 추천 음료 (부족한 {weakest_name} 기운을 채우는 한 잔)
{cafe_info}

## 대화 지침
1. 첫 대화에서는 사주 분석 결과를 간결하게 설명하고, 추천 장소 1곳과 음료 1잔을 자연스럽게 안내하세요.
2. 후속 대화에서는 사용자 질문에 따라 추가 장소/음료를 추천하거나, 오행에 기반한 조언을 해주세요.
3. 추천할 때 오행 상생상극 원리를 활용하세요:
   - 상생: 木→火→土→金→水→木 (목생화, 화생토, 토생금, 금생수, 수생목)
   - 상극: 木→土→水→火→金→木 (목극토, 토극수, 수극화, 화극금, 금극목)
4. 장소를 추천할 때는 건물번호와 단과대학 이름을 언급하세요. 구역명은 언급하지 마세요.
5. 음료를 추천할 때는 카페명, 메뉴명, 가격을 언급하세요. 구역명은 언급하지 마세요.
6. 답변은 200자 내외로 간결하고 명확하게 작성하세요.
"""
    return system_prompt


def create_initial_greeting(saju_result: dict) -> str:
    """
    사주 분석 후 챗봇의 첫 인사말을 생성한다.
    LLM을 사용하지 않고 빠르게 표시할 수 있는 정적 인사말.
    """
    weakest = saju_result["weakest"]
    weakest_name = saju_result["weakest_name"]

    from saju_engine import ELEMENT_EMOJI

    rec_locs = filter_locations(weakest, 1)
    rec_cafes = filter_cafes(weakest, 1)

    loc_text = ""
    if rec_locs:
        loc = rec_locs[0]
        loc_text = f"**{loc.get('id', '')}동** ({loc.get('college', '')})"

    cafe_text = ""
    if rec_cafes:
        c = rec_cafes[0]
        price_parts = []
        if c.get("hot"):
            price_parts.append(f"HOT {c['hot']}원")
        if c.get("ice"):
            price_parts.append(f"ICE {c['ice']}원")
        price = " / ".join(price_parts)
        cafe_text = f"**{c['cafe']}**의 **{c['menu']}** ({price})"

    greeting = (
        f"사주 분석이 완료되었습니다! ✨\n\n"
        f"오늘의 오행을 보니, **{weakest_name}** {ELEMENT_EMOJI[weakest]}의 기운이 "
        f"가장 부족합니다.\n\n"
        f"오늘은 {loc_text}에 가시면 부족한 기운을 보충하실 수 있습니다.\n\n"
        f"음료는 {cafe_text}을 추천드립니다. "
        f"{weakest_name}의 기운을 채우기에 딱 좋은 선택이에요! ☕\n\n"
        f"더 궁금한 점이 있으시면 편하게 물어보세요."
    )
    return greeting


def _resolve_hf_token(hf_token: str | None) -> str:
    """
    유효한 HF 토큰을 결정한다.
    1순위: 전달된 OAuth 토큰 (hf_ 접두사 확인)
    2순위: 환경변수 HF_TOKEN
    """
    if hf_token and hf_token.startswith("hf_"):
        return hf_token
    env_token = os.environ.get("HF_TOKEN", "").strip()
    if env_token and env_token.startswith("hf_"):
        return env_token
    return hf_token or ""


def get_llm_response(
    message: str,
    history: list[dict[str, str]],
    system_prompt: str,
    hf_token: str,
):
    """
    HuggingFace InferenceClient를 통해 스트리밍 응답을 생성한다.
    로컬 환경에서는 환경변수 HF_TOKEN을 fallback으로 사용한다.
    """

    token = _resolve_hf_token(hf_token)

    client = InferenceClient(
        token=token,
        model="Qwen/Qwen2.5-72B-Instruct",
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    response = ""
    try:
        for chunk in client.chat_completion(
            messages,
            max_tokens=1024,
            stream=True,
            temperature=0.8,
            top_p=0.95,
        ):
            choices = chunk.choices
            token_text = ""
            if len(choices) and choices[0].delta.content:
                token_text = choices[0].delta.content
            response += token_text
            yield response
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg or "token" in error_msg.lower() or "401" in error_msg:
            yield (
                "🔑 HuggingFace 토큰이 유효하지 않은 것 같습니다.\n\n"
                "`.env` 파일의 `HF_TOKEN`을 확인하시거나, "
                "사이드바에서 HuggingFace 로그인을 다시 시도해 주세요."
            )
        else:
            yield f"⚠️ 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.\n\n오류: {error_msg}"
