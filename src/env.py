# src/env.py — .env의 API 키를 읽는 곳은 여기 하나로 모은다 (NF2003)
import os

from dotenv import load_dotenv

from src.config import ENV_PATH, GMS_BASE_URL

load_dotenv(ENV_PATH)


class MissingAPIKeyError(RuntimeError):
    """API 키 누락 — 화면에서 안내 문구로 바꿔 보여준다 (F2012)."""


def openai_kwargs() -> dict:
    """OpenAI 클라이언트 인자. OPENAI_API_KEY가 있으면 OpenAI에 직접, 없으면 GMS_KEY로 SSAFY GMS를 거친다."""
    if os.getenv("OPENAI_API_KEY"):
        return {"api_key": os.environ["OPENAI_API_KEY"], "base_url": os.getenv("OPENAI_BASE_URL")}
    if os.getenv("GMS_KEY"):
        return {"api_key": os.environ["GMS_KEY"], "base_url": os.getenv("OPENAI_BASE_URL") or GMS_BASE_URL}
    raise MissingAPIKeyError(".env에 GMS_KEY(또는 OPENAI_API_KEY)가 없습니다.")


def kipris_key() -> str:
    if not os.getenv("KIPRIS_API_KEY"):
        raise MissingAPIKeyError(".env에 KIPRIS_API_KEY가 없습니다.")
    return os.environ["KIPRIS_API_KEY"]


def wiki_user_agent() -> str:
    """위키백과는 연락처가 담긴 User-Agent를 요구한다 (NF2006)."""
    if not os.getenv("WIKI_CONTACT"):
        raise MissingAPIKeyError(".env에 WIKI_CONTACT(위키백과 User-Agent에 넣을 연락처)가 없습니다.")
    return f"patent-rag-edu/0.1 ({os.environ['WIKI_CONTACT']}) python-requests"
