# src/security/policy.py — 외부 전송 정책: 어느 등급의 자료를 어느 LLM으로 보낼 수 있나
#
# 경로(Route) = LLM을 부르는 곳. 기본 경로는 .env의 키로 정한 제공자(Gemini·OpenAI·GMS)이고,
# LOCAL_LLM_BASE_URL을 주면 사내 운영 모델(Ollama·vLLM 같은 OpenAI 호환 서버) 경로가 더해진다.
# 경로마다 보낼 수 있는 최고 등급(상한)이 있다. 기본값은 제공자 약관을 근거로 정했다 (docs/decisions/009-send-policy.md):
#   gemini → 공개   무료 등급 약관: 입력을 제품 개선에 쓰고 사람이 검토할 수 있으며, 기밀 정보를 넣지 말라고 적혀 있다
#   openai → 내부   API 입력은 기본적으로 학습에 쓰지 않는다. 기밀은 별도 계약 없이 보내지 않는다
#   gms    → 공개   교육 과정용 제3자 프록시
#   사내 모델 → 기밀
# 계약이 바뀌거나(유료 등급·기업 계약) 가상 데이터만 쓰는 데모라면 LLM_MAX_LEVEL로 기본 경로의 상한을 바꾼다.
#
# 요청마다 보낼 자료 중 가장 높은 등급을 받을 수 있는 경로를 고른다. 없으면 보내지 않는다 (막는 쪽이 기본값).
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import urlparse

from src.config import (
    CHAT_MODEL,
    LLM_MAX_LEVEL,
    LLM_PROVIDER,
    LOCAL_LLM_BASE_URL,
    LOCAL_LLM_MAX_LEVEL,
    LOCAL_LLM_MODEL,
    OPENAI_BASE_URL,
)
from src.security.acl import LEVELS, level_rank

DEFAULT_CEILING = {"gemini": "공개", "gms": "공개", "openai": "내부"}
PROVIDER_LABELS = {"gemini": "Gemini API", "openai": "OpenAI API", "gms": "SSAFY GMS", "local": "사내 모델"}
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}


@dataclass(frozen=True)
class Route:
    name: str          # gemini | openai | gms | local
    model: str
    ceiling: str       # 이 경로로 보낼 수 있는 최고 등급
    external: bool     # 회사 밖 서비스로 나가는가

    @property
    def label(self) -> str:
        return f"{PROVIDER_LABELS.get(self.name, self.name)} · {self.model}"

    def can_carry(self, level: str) -> bool:
        return level_rank(level) <= level_rank(self.ceiling)


class SendBlocked(RuntimeError):
    """보낼 자료의 등급을 받을 수 있는 LLM 경로가 없다 — 보내지 않고 이유를 알린다."""

    def __init__(self, level: str, available: list[Route]):
        self.level, self.available = level, available
        paths = ", ".join(f"{r.label}({r.ceiling}까지)" for r in available)
        super().__init__(f"'{level}' 자료를 받을 수 있는 LLM 경로가 없어 보내지 않았습니다. 현재 경로: {paths}. "
                         "사내 모델(LOCAL_LLM_BASE_URL)을 설정하면 그쪽으로 보냅니다.")


def _is_local(url: str | None) -> bool:
    return bool(url) and (urlparse(url).hostname or "") in _LOCAL_HOSTS


def primary_route(model: str = CHAT_MODEL) -> Route:
    ceiling = LLM_MAX_LEVEL or DEFAULT_CEILING.get(LLM_PROVIDER, LEVELS[0])
    level_rank(ceiling)   # 잘못된 설정값이면 여기서 멈춘다
    return Route(LLM_PROVIDER, model, ceiling, external=not _is_local(OPENAI_BASE_URL))


def local_route() -> Route | None:
    if not LOCAL_LLM_BASE_URL:
        return None
    level_rank(LOCAL_LLM_MAX_LEVEL)
    return Route("local", LOCAL_LLM_MODEL, LOCAL_LLM_MAX_LEVEL, external=False)


def routes(model: str = CHAT_MODEL) -> list[Route]:
    """기본 경로를 먼저 둔다 (품질·속도가 나은 쪽). 상한을 넘는 자료만 사내 모델로 간다."""
    return [r for r in (primary_route(model), local_route()) if r is not None]


def route_for(level: str, model: str = CHAT_MODEL) -> Route | None:
    return next((r for r in routes(model) if r.can_carry(level)), None)


def require_route(level: str, model: str = CHAT_MODEL) -> Route:
    route = route_for(level, model)
    if route is None:
        raise SendBlocked(level, routes(model))
    return route


def chat_on(route: Route, temperature: float = 0):
    """경로에 맞는 채팅 모델."""
    from src.rag.llm import get_chat, get_local_chat
    return get_local_chat(route.model, temperature) if route.name == "local" else get_chat(temperature, route.model)


def split_sendable(evidence: Iterable, ceiling: str) -> tuple[list, list]:
    """(보낼 수 있는 근거, 상한을 넘어 빼야 하는 근거). 근거마다 level 속성이 있다."""
    sendable, withheld = [], []
    for e in evidence:
        (sendable if level_rank(e.level) <= level_rank(ceiling) else withheld).append(e)
    return sendable, withheld
