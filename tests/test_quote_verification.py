# 선행기술 조사의 핵심 안전장치 — LLM이 인용한 구절이 청구항 원문에 실제로 있는지 코드로 확인한다.
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.rag.retriever import Evidence
from src.service.prior_art import _squash, compare, quote_found


def claims(text: str) -> list[Evidence]:
    return [Evidence(1, "1020230005129:kipris-1020230005129:0", "patent_claims", "청구항", "https://doi.org/x", text)]


CLAIM = ("1. 스캔 체인(scan chain)에 테스트 패턴을 인가하고, 고장 로그로부터 고장 벡터를 생성하며, "
         "고장 유형별 인공지능 모델이 고장 난 스캔 셀의 위치를 예측하는 스캔 체인 진단 방법.")


def test_squash_removes_whitespace_and_parenthetical():
    assert _squash("스캔 체인(scan chain)에 테스트 패턴") == "스캔체인에테스트패턴"


def test_exact_substring_is_found():
    assert quote_found("고장 로그로부터 고장 벡터를 생성", claims(CLAIM))


def test_parenthetical_dropped_by_llm_still_matches():
    # LLM이 영문 병기를 빼고 옮긴 구절도 원문으로 인정한다
    assert quote_found("스캔 체인에 테스트 패턴을 인가", claims(CLAIM))


def test_small_edit_within_90_percent_is_tolerated():
    # 조사 하나가 바뀐 정도 — "예측하는" → "예측한"
    assert quote_found("인공지능 모델이 고장 난 스캔 셀의 위치를 예측한", claims(CLAIM))


def test_fabricated_quote_is_rejected():
    assert not quote_found("고장 로그로 학습 데이터를 만들어 신경망을 훈련", claims(CLAIM))


def test_too_short_quote_is_rejected():
    assert not quote_found("고장", claims(CLAIM))


def test_compare_downgrades_unverified_disclosure():
    """LLM이 '개시'라고 해도 인용 구절이 원문에 없으면 미개시로, 근거 번호가 없으면 미개시로 바꾼다."""
    llm = FakeListChatModel(responses=["""{"elements": [
        {"no": 1, "quote": "고장 로그로부터 고장 벡터를 생성", "evidence": [1], "status": "개시", "note": "원문에 있음"},
        {"no": 2, "quote": "학습 데이터를 만들어 신경망을 훈련", "evidence": [1], "status": "개시", "note": "지어낸 구절"},
        {"no": 3, "quote": "스캔 체인에 테스트 패턴을 인가", "evidence": [], "status": "일부", "note": "근거 번호 없음"}
      ], "summary": "요약"}"""])
    elements = ["고장 로그로 고장 벡터를 만든다", "학습 데이터로 신경망을 훈련한다", "스캔 체인에 패턴을 넣는다", "결과를 검증한다"]
    names = ["고장 벡터", "학습", "패턴 인가", "검증"]
    meta = {"item_id": "1020230005129", "title": "T", "applicant": "A", "application_date": "2023-01-01",
            "status": "공개", "source_url": "https://doi.org/x"}
    c = compare(elements, names, meta, "이유", claims(CLAIM), llm)

    assert [r.status for r in c.rows] == ["개시", "미개시", "미개시", "미개시"]
    assert "찾지 못해" in c.rows[1].note
    assert "근거 번호가 없어" in c.rows[2].note
    assert c.rows[3].note == "판정 없음"           # 응답에 없는 구성요소
    assert c.rows[0].quote and not c.rows[1].quote   # 미개시에는 구절·근거를 붙이지 않는다
    assert c.overlap == 0.25
