# src/rag/llm.py — 채팅 모델 생성과 LLM 출력 파싱
import json
import re

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from langchain_core.rate_limiters import InMemoryRateLimiter

from src.config import CHAT_MODEL, LLM_MAX_RETRIES, LLM_RPM
from src.env import openai_kwargs

# 프로세스 안의 모든 채팅 모델이 하나의 한도를 나눠 쓴다 (스레드 간 공유). 분당 LLM_RPM회를 넘지 않게 호출 전에 기다린다.
_RATE_LIMITER = InMemoryRateLimiter(requests_per_second=LLM_RPM / 60, check_every_n_seconds=0.2,
                                    max_bucket_size=1) if LLM_RPM > 0 else None


def get_chat(temperature: float = 0, model: str = CHAT_MODEL) -> BaseChatModel:
    # max_retries: 429(요청 한도)·일시 오류를 지수 백오프로 다시 시도한다. 무료 등급(Gemini)은 한도가 작아 넉넉히 둔다
    return ChatOpenAI(model=model, temperature=temperature, max_retries=LLM_MAX_RETRIES,
                      rate_limiter=_RATE_LIMITER, **openai_kwargs())


def parse_json_array(text: str) -> list:
    """응답에서 JSON 배열을 꺼낸다. 코드블록·앞뒤 설명이 붙어도 읽고, 못 읽으면 빈 배열.

    처음 '['부터 마지막 ']'까지를 먼저 시도해 객체 배열·중첩 배열도 읽고, 안 되면 가장 짧은 배열을 시도한다.
    """
    spans = []
    if "[" in text and "]" in text:
        spans.append(text[text.index("["):text.rindex("]") + 1])
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        spans.append(match.group(0))
    for span in spans:
        try:
            value = json.loads(span)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return value
    return []


def parse_json_object(text: str) -> dict:
    """응답에서 첫 '{'부터 마지막 '}'까지를 JSON 객체로 읽는다. 못 읽으면 빈 dict."""
    if "{" not in text or "}" not in text:
        return {}
    try:
        value = json.loads(text[text.index("{"):text.rindex("}") + 1])
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}
