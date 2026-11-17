# 외부 전송 정책 — 경로 상한, 사내 모델로 넘기기, 막기
import pytest

from src.rag.retriever import Evidence
from src.security import policy
from src.security.policy import SendBlocked


@pytest.fixture
def gemini_only(monkeypatch):
    monkeypatch.setattr(policy, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(policy, "LLM_MAX_LEVEL", "")
    monkeypatch.setattr(policy, "LOCAL_LLM_BASE_URL", "")


def ev(level):
    return Evidence(1, "x", "internal_doc", "t", "", "본문", level=level)


def test_free_tier_default_ceiling_is_public(gemini_only):
    assert policy.primary_route().ceiling == "공개"          # 무료 등급 약관: 기밀 정보를 넣지 말 것
    assert policy.route_for("공개").name == "gemini"
    assert policy.route_for("내부") is None
    with pytest.raises(SendBlocked, match="기밀"):
        policy.require_route("기밀")


def test_confidential_goes_to_local_model_when_configured(gemini_only, monkeypatch):
    monkeypatch.setattr(policy, "LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
    assert policy.route_for("공개").name == "gemini"          # 공개 자료는 기본 경로
    local = policy.route_for("기밀")
    assert local.name == "local" and not local.external


def test_ceiling_override_for_contract_or_demo(gemini_only, monkeypatch):
    monkeypatch.setattr(policy, "LLM_MAX_LEVEL", "내부")
    assert policy.route_for("내부").name == "gemini" and policy.route_for("기밀") is None
    monkeypatch.setattr(policy, "LLM_MAX_LEVEL", "극비")
    with pytest.raises(ValueError):
        policy.primary_route()                                 # 잘못된 설정은 조용히 넘기지 않는다


def test_openai_default_is_internal(monkeypatch):
    monkeypatch.setattr(policy, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(policy, "LLM_MAX_LEVEL", "")
    assert policy.primary_route().ceiling == "내부"


def test_split_sendable_keeps_order():
    items = [ev("공개"), ev("기밀"), ev("내부"), ev("공개")]
    sendable, withheld = policy.split_sendable(items, "내부")
    assert [e.level for e in sendable] == ["공개", "내부", "공개"] and [e.level for e in withheld] == ["기밀"]
