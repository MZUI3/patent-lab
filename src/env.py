# src/env.py — .env의 API 키를 읽는 곳은 여기 하나로 모은다
import os

from dotenv import load_dotenv

from src.config import ENV_PATH, GEMINI_BASE_URL, GMS_BASE_URL

load_dotenv(ENV_PATH)


class MissingAPIKeyError(RuntimeError):
    """API 키 누락 — 화면에서 안내 문구로 바꿔 보여준다."""


def openai_kwargs() -> dict:
    """OpenAI 호환 클라이언트 인자. 키 종류로 제공자를 정한다:
    OPENAI_API_KEY → OpenAI 직접, GEMINI_API_KEY → Gemini의 OpenAI 호환 주소, GMS_KEY → SSAFY GMS 프록시.
    OPENAI_BASE_URL을 주면 어느 경우든 그 주소를 쓴다."""
    if os.getenv("OPENAI_API_KEY"):
        return {"api_key": os.environ["OPENAI_API_KEY"], "base_url": os.getenv("OPENAI_BASE_URL")}
    if os.getenv("GEMINI_API_KEY"):
        return {"api_key": os.environ["GEMINI_API_KEY"], "base_url": os.getenv("OPENAI_BASE_URL") or GEMINI_BASE_URL}
    if os.getenv("GMS_KEY"):
        return {"api_key": os.environ["GMS_KEY"], "base_url": os.getenv("OPENAI_BASE_URL") or GMS_BASE_URL}
    raise MissingAPIKeyError(".env에 LLM 키가 없습니다 (GEMINI_API_KEY, OPENAI_API_KEY, GMS_KEY 중 하나).")


def kipris_key() -> str:
    if not os.getenv("KIPRIS_API_KEY"):
        raise MissingAPIKeyError(".env에 KIPRIS_API_KEY가 없습니다.")
    return os.environ["KIPRIS_API_KEY"]


def wiki_user_agent() -> str:
    """위키백과는 연락처가 담긴 User-Agent를 요구한다."""
    if not os.getenv("WIKI_CONTACT"):
        raise MissingAPIKeyError(".env에 WIKI_CONTACT(위키백과 User-Agent에 넣을 연락처)가 없습니다.")
    return f"patent-lab/0.1 ({os.environ['WIKI_CONTACT']}) python-requests"
